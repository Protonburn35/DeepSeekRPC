"""Real owned-stage execution and dependency/commit boundaries.

The service owns TCP dispatch and the cluster coordinator owns scheduling.
This component binds their validated operations to the actual CUDA graph,
independent local slot maps, original Engram history and phase ledger.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict, dataclass
import threading

from .attention_state import AttentionWindow
from .bootstrap import LoadedStage
from .history import TokenHistory
from .jsonutil import ValidationError, fingerprint, integer
from .phases import PhaseLedger
from .planner import Role
from .protocol import hex_id
from .state import Operation


@dataclass(frozen=True)
class StepInput:
    tokens: tuple[int, ...]
    text_mask: tuple[bool, ...]

    def __post_init__(self):
        if (not isinstance(self.tokens, tuple) or not 1 <= len(self.tokens) <= 64 or
                not isinstance(self.text_mask, tuple) or len(self.text_mask) != len(self.tokens) or
                any(type(v) is not bool for v in self.text_mask)):
            raise ValidationError("execution needs immutable bounded token/image rows")
        for token in self.tokens:
            integer(token, "execution token ID", 0, 129279)

    def digest(self, window):
        operation = asdict(window.operation)
        del operation["digest"]
        return fingerprint({"operation": operation, "phase": window.phase,
                            "history_digest": window.history_digest, "encoder_tokens": window.encoder_tokens,
                            "tokens": self.tokens, "text_mask": self.text_mask})

    def make_window(self, *, session, request, epoch, number, transaction, start,
                    phase, history_digest, encoder_tokens):
        operation = Operation(session, request, epoch, number, transaction, "0" * 64, start, len(self.tokens))
        provisional = AttentionWindow(operation, history_digest, phase, encoder_tokens)
        operation = Operation(session, request, epoch, number, transaction, self.digest(provisional), start, len(self.tokens))
        return AttentionWindow(operation, history_digest, phase, encoder_tokens)


def source_closures(window: AttentionWindow, ratio: int):
    integer(ratio, "source ratio", 1, 2)
    if window.phase not in {"encoder", "decode"}:
        return (), ()
    positions = tuple(p for p in range(window.operation.start, window.operation.start + window.operation.rows)
                      if (p + 1) % ratio == 0)
    return tuple((p + 1) // ratio - 1 for p in positions), positions


class StageExecution:
    def __init__(self, stage: LoadedStage):
        if not isinstance(stage, LoadedStage) or stage.closed:
            raise ValidationError("execution requires the actual completed native Windows stage loader")
        self.stage = stage
        self.ledger = PhaseLedger(stage.role)
        self.history = TokenHistory(stage.identity)
        self._mutex = threading.RLock()
        self._window = self._inputs = self._result = self._prepared = None
        self._hashes = self._last_logits = None
        self._hash_pending = self._forwarded = False
        self._imported_sources, self._imported_selections = set(), set()

    @contextmanager
    def _compute(self):
        import torch
        with self._mutex, torch.cuda.device(self.stage.attention.device), torch.cuda.stream(self.stage.stream), torch.inference_mode():
            if self.stage.closed:
                raise ValidationError("the persistent stage is closed")
            yield

    def request_cancel(self):
        # Control thread never waits for the model execution mutex.
        self.stage.cancellation.set()

    def fault(self):
        self.stage.cancellation.set()
        self.stage.attention.fail()
        self.ledger.fault()

    def reset(self, *, session, request, epoch, tokens, text_mask=None, images=(), expected_history):
        hex_id(expected_history, 64, "full prompt history")
        with self._compute():
            minimum = max(self.ledger.epoch + 1, self.stage.attention.next_reset_epoch)
            integer(epoch, "coherent new session epoch", minimum, (1 << 63) - 1)
            self.stage.stream.synchronize()
            # A temporary bounded history validates the incoming identity
            # before any live cache is invalidated. No model bytes are copied.
            incoming = TokenHistory(self.stage.identity)
            if incoming.set_prompt(tokens, text_mask, images) != expected_history:
                raise ValidationError("prompt token/image history differs from the all-stage request identity")
            try:
                if self.ledger.phase == "faulted":
                    self.stage.attention.discard_session(session, epoch)
                else:
                    self.stage.attention.reset(session, epoch)
                if self.stage.engram_history is not None:
                    self.stage.engram_history.reset()
                self.ledger.reset(session=session, request=request, epoch=epoch,
                                  history_digest=expected_history, encoder_tokens=incoming.length)
                self.history = incoming
                self._window = self._inputs = self._result = self._prepared = None
                self._hashes = self._last_logits = None
                self._hash_pending = self._forwarded = False
                self.stage.cancellation.clear()
                self.stage.guard.check()
            except BaseException:
                self.fault()
                raise
            return {"history_digest": self.history.digest, "epoch": epoch,
                    "minimum_next_epoch": epoch + 1, "weights_retained": True}

    def begin(self, window: AttentionWindow, inputs: StepInput):
        import torch
        if not isinstance(inputs, StepInput) or inputs.digest(window) != window.operation.digest:
            raise ValidationError("execution semantic inputs differ from their operation digest")
        with self._compute():
            cached = self.ledger.admit(window)
            if cached is not None:
                return cached
            self._window, self._inputs = window, inputs
            self._prepared = self._result = self._hashes = None
            self._hash_pending = self._forwarded = False
            self._imported_sources.clear()
            self._imported_selections.clear()
            try:
                if window.phase in {"encoder", "target"}:
                    if (inputs.tokens, inputs.text_mask) != self.history.rows(window.operation.start, window.operation.rows):
                        raise ValidationError("CED window differs from immutable original token/image history")
                elif window.phase == "decode":
                    if inputs.tokens[0] != self.ledger.pending_token or not all(inputs.text_mask):
                        raise ValidationError("verification must begin with the published pending seed")
                elif inputs.tokens != (self.ledger.pending_token, 128799, 128799, 128799, 128799) or not all(inputs.text_mask):
                    raise ValidationError("DSpark input differs from its seed plus four original noise tokens")
                self.stage.guard.check()
                self.stage.attention.begin(window, cancelled=self.stage.cancellation.is_set)
                if self.stage.role == Role.HOST and window.phase != "draft":
                    history = self.stage.engram_history
                    if window.phase == "target":
                        hashes = history.replay(list(inputs.tokens), start=window.operation.start,
                                                text_mask=list(inputs.text_mask))
                    else:
                        hashes = history.begin(list(inputs.tokens), list(inputs.text_mask))
                        self._hash_pending = True
                    self._hashes = torch.tensor(hashes, dtype=torch.int64, device=self.stage.attention.device)
            except BaseException:
                self.stage.attention.fail()
                raise
            return None

    def _current(self):
        if self._window is None or self.ledger.phase == "faulted":
            raise ValidationError("worker has no valid active execution window")
        return self._window

    def import_source(self, source: int, kv, index):
        import torch
        with self._compute():
            window, state = self._current(), self.stage.attention
            if (self._forwarded or source in self._imported_sources or source not in state.sources or
                    state.plan.owner(source) == self.stage.role or window.phase not in {"encoder", "decode"}):
                raise ValidationError("source delta is duplicate, unneeded, unowned or stale")
            logical, positions = source_closures(window, state.plan.geometry.compress_ratios[source])
            if (not logical or kv.device != state.device or index.device != state.device or
                    kv.dtype != torch.bfloat16 or index.dtype != torch.bfloat16 or
                    kv.shape != (len(logical), 512) or index.shape != (len(logical), 128) or
                    not kv.is_contiguous() or not index.is_contiguous() or
                    not bool(torch.isfinite(kv).all()) or not bool(torch.isfinite(index).all())):
                raise ValidationError("source delta does not cover the current exact BF16 closures")
            state.sources[source].write(logical, positions, kv)
            state.index_keys[source].write(logical, positions, index)
            self._imported_sources.add(source)

    def import_selection(self, source, selection):
        with self._compute():
            window, state = self._current(), self.stage.attention
            required = {d.source_layer for d in state.plan.dependencies()
                        if d.recipient == self.stage.role and d.field == "topk"}
            if (self._forwarded or source not in required or source in self._imported_selections or
                    window.phase == "draft" or window.phase == "encoder" and source >= 20):
                raise ValidationError("index selection is duplicate, unneeded or stale")
            state.publish_selection(source, selection)
            self._imported_selections.add(source)

    def _require_dependencies(self):
        window, state = self._current(), self.stage.attention
        if window.phase == "draft":
            return
        if window.phase in {"encoder", "decode"}:
            required = {source for source in state.sources if state.plan.owner(source) != self.stage.role and
                        source_closures(window, state.plan.geometry.compress_ratios[source])[0]}
            if not required <= self._imported_sources:
                raise ValidationError("forward cannot use a missing current source/shadow delta")
        required = {d.source_layer for d in state.plan.dependencies() if d.recipient == self.stage.role and
                    d.field == "topk" and (window.phase != "encoder" or d.source_layer < 20)}
        if not required <= self._imported_selections:
            raise ValidationError("forward cannot use a missing current source top-k/candidate selection")

    def forward(self, *, incoming=None, image_rows=None):
        import torch
        with self._compute():
            window = self._current()
            if self._forwarded:
                raise ValidationError("the current CUDA graph window was already executed")
            try:
                self._require_dependencies()
                if window.phase == "draft":
                    if self.stage.role == Role.TERMINAL:
                        self._result = self.stage.graph.run_draft(self.ledger.pending_token)
                elif window.phase != "encoder" or self.stage.plan.bounds(self.stage.role)[0] <= 20:
                    tokens = torch.tensor(self._inputs.tokens, dtype=torch.int64, device=self.stage.attention.device)
                    mask = torch.tensor(self._inputs.text_mask, dtype=torch.bool, device=self.stage.attention.device)
                    self._result = self.stage.graph.run_target(tokens, incoming=incoming, engram_hashes=self._hashes,
                                                               image_rows=image_rows, text_mask=mask)
                    if self.stage.role == Role.TERMINAL and window.phase == "target" and self._result.logits is not None:
                        self._last_logits = self._result.logits.clone()
                elif incoming is not None or image_rows is not None:
                    raise ValidationError("a stage beyond source 20 receives only encoder source deltas")
                self._forwarded = True
                self.stage.guard.check()
                return self._result
            except BaseException:
                self.stage.attention.fail()
                raise

    def export_source(self, source):
        import torch
        with self._compute():
            window, state = self._current(), self.stage.attention
            if not self._forwarded or state.plan.owner(source) != self.stage.role or source not in state.sources:
                raise ValidationError("source export requires its actual completed owner graph")
            logical, positions = source_closures(window, state.plan.geometry.compress_ratios[source])
            if not logical:
                return None
            values = []
            for pool in (state.sources[source], state.index_keys[source]):
                with pool.read_window(window.operation):
                    slots = torch.tensor(pool.physical(logical), dtype=torch.int64, device=state.device)
                    values.append(pool.data.index_select(0, slots).contiguous())
            return logical, positions, *values

    def export_selection(self, source):
        with self._compute():
            self._current()
            if not self._forwarded or self.stage.plan.owner(source) != self.stage.role:
                raise ValidationError("selection export requires its actual completed owner graph")
            return self.stage.attention.selection(source)

    def prepare(self, *, accepted, result_digest, pending_token=None, aborted=False):
        with self._compute():
            window = self._current()
            if type(aborted) is not bool or aborted and (accepted or pending_token is not None) or not aborted and not self._forwarded:
                raise ValidationError("execution commit is incomplete or has an invalid abort decision")
            decision = fingerprint({"result_digest": hex_id(result_digest, 64, "terminal result"), "aborted": aborted})
            digest = self.ledger.prepare(window, accepted=accepted, result_digest=decision, pending_token=pending_token)
            prepared = accepted, result_digest, pending_token, aborted, digest
            if self._prepared is not None and self._prepared != prepared:
                raise ValidationError("execution received a conflicting prepared commit")
            self._prepared = prepared
            return digest

    def finish(self, *, commit_digest):
        with self._compute():
            window = self._current()
            if self._prepared is None or self._prepared[-1] != commit_digest:
                raise ValidationError("execution finish lacks the matching prepared decision")
            if self.stage.attention.active_operation is None:
                return self.ledger.finish(window, commit_digest=commit_digest)
            try:
                accepted = self._prepared[0]
                self.stage.attention.finish(window.operation, accepted)
                if self._hash_pending:
                    self.stage.engram_history.commit(accepted)
                    self._hash_pending = False
                receipt = self.ledger.finish(window, commit_digest=commit_digest)
                self._result = self._hashes = None
                self.stage.guard.check()
                return receipt
            except BaseException:
                self.fault()
                raise

    def seal(self, receipts, *, commit_digest):
        with self._compute():
            receipt = self.ledger.seal(receipts, commit_digest=commit_digest)
            if self._prepared is not None and self._prepared[-1] == commit_digest:
                self._window = self._inputs = self._prepared = None
            return receipt

    def activate_target(self, receipts, *, commit_digest):
        with self._compute():
            try:
                self.ledger.encoder_barrier(receipts, commit_digest=commit_digest)
                self.stage.attention.rewind_for_exact_target(self.history.digest, self.history.length)
                self.ledger.activate_target()
            except BaseException:
                self.fault()
                raise

    @property
    def target_tail_logits(self):
        if (self.stage.role != Role.TERMINAL or self._last_logits is None or self.ledger.phase != "decode" or
                self._window is not None):
            raise ValidationError("stage C has no completed exact target-tail logits")
        return self._last_logits
