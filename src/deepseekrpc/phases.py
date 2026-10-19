"""Metadata-only production cursors for exact CED and bounded decode windows.

GPU pools own their bytes/undo. This ledger advances only after their local
finish succeeds. A coordinator still needs all three matching commit receipts
before publishing tokens. An uncertain partial finish faults the session.
"""
from __future__ import annotations

from collections import OrderedDict
from dataclasses import asdict, dataclass

from .attention_state import AttentionWindow
from .jsonutil import ValidationError, fingerprint, integer
from .planner import Role
from .protocol import hex_id


@dataclass(frozen=True)
class PhaseReceipt:
    role: Role
    operation_digest: str
    commit_digest: str
    phase: str
    encoder_cursor: int
    target_cursor: int
    processed: int
    pending_token: int | None

    def __post_init__(self):
        if not isinstance(self.role, Role) or self.phase not in {"encoder", "target", "decode"}:
            raise ValidationError("invalid stage receipt role or phase")
        hex_id(self.operation_digest, 64, "receipt operation")
        hex_id(self.commit_digest, 64, "receipt commit")
        for name in ("encoder_cursor", "target_cursor", "processed"):
            integer(getattr(self, name), name, 0, 1048576)
        if self.pending_token is not None:
            integer(self.pending_token, "receipt pending token", 0, 129279)

    def as_dict(self):
        result = asdict(self)
        result["role"] = int(self.role)
        return result


def unanimous_receipts(receipts, *, expected_digest):
    """Validate the actual stage acknowledgments without inventing peers."""
    if (not isinstance(receipts, (tuple, list)) or len(receipts) != 3 or
            any(not isinstance(r, PhaseReceipt) for r in receipts) or
            {r.role for r in receipts} != set(Role)):
        raise ValidationError("commit requires exactly one receipt from every actual stage")
    hex_id(expected_digest, 64, "unanimous commit")
    bodies = []
    for receipt in receipts:
        if receipt.commit_digest != expected_digest:
            raise ValidationError("stage commit acknowledgment has a different decision digest")
        body = receipt.as_dict()
        del body["role"]
        bodies.append(body)
    if any(body != bodies[0] for body in bodies[1:]):
        raise ValidationError("stage acknowledgments disagree on phase, processed or pending state")
    return receipts[0]


class PhaseLedger:
    def __init__(self, role: Role, *, replay_limit=128):
        if not isinstance(role, Role):
            raise ValidationError("phase ledger needs an explicit stage role")
        self.role = role
        self.replay_limit = integer(replay_limit, "phase receipt history", 1, 4096)
        self.epoch, self.phase = -1, "empty"
        self.session = self.request = self.history_digest = None
        self.encoder_tokens = self.encoder_cursor = self.target_cursor = self.processed = 0
        self.pending_token = None
        self._next, self._active, self._prepared = 1, None, None
        self._receipts = OrderedDict()
        self._encoder_barrier = None
        self._awaiting, self._sealed_number = None, 0
        self._seeded = False

    def reset(self, *, session, request, epoch, history_digest, encoder_tokens):
        hex_id(session, 32, "phase session")
        hex_id(request, 32, "phase request")
        hex_id(history_digest, 64, "immutable full prompt/image history")
        integer(epoch, "phase epoch", 0, (1 << 63) - 1)
        integer(encoder_tokens, "full encoded prompt extent", 1, 1048576)
        if (self._active is not None or epoch <= self.epoch or
                self._awaiting is not None and self.phase != "faulted"):
            raise ValidationError("phase reset requires quiescence and a newer epoch")
        self.session, self.request, self.epoch = session, request, epoch
        self.history_digest, self.encoder_tokens = history_digest, encoder_tokens
        self.encoder_cursor = self.target_cursor = self.processed = 0
        self.pending_token, self.phase = None, "encoder"
        self._next, self._prepared, self._encoder_barrier = 1, None, None
        self._awaiting, self._sealed_number = None, 0
        self._seeded = False
        self._receipts.clear()

    def admit(self, window: AttentionWindow):
        """Return an old commit receipt on exact replay; otherwise admit once."""
        if not isinstance(window, AttentionWindow):
            raise ValidationError("phase admission needs an explicit attention window")
        operation = window.operation
        if (self.phase in {"empty", "faulted"} or operation.session != self.session or
                operation.request != self.request or operation.epoch != self.epoch):
            raise ValidationError("operation belongs to an empty, faulted or different session")
        old = self._receipts.get(operation.number)
        if old is not None:
            if old[0] != window:
                raise ValidationError("replayed operation changed its phase, history or semantic inputs")
            return old[2]
        if (self._active is not None or self._awaiting is not None or operation.number != self._next or
                window.history_digest != self.history_digest):
            raise ValidationError("phase operation is active, stale, out of order or has different history")
        if self.phase in {"encoder", "target"}:
            cursor = self.encoder_cursor if self.phase == "encoder" else self.target_cursor
            expected_rows = min(64, self.encoder_tokens - cursor)
            if (window.phase != self.phase or window.encoder_tokens != self.encoder_tokens or
                    operation.start != cursor or operation.rows != expected_rows or not expected_rows):
                raise ValidationError("exact CED must traverse the whole prompt in identical fixed 64-row chunks")
        else:
            if self.pending_token is None or operation.start != self.processed:
                raise ValidationError("decode requires the published pending seed at the processed cursor")
            if window.phase == "draft":
                if operation.rows != 5:
                    raise ValidationError("the pinned DSpark proposal graph has five rows")
            elif window.phase != "decode" or not 1 <= operation.rows <= 6:
                raise ValidationError("decode must verify the seed plus at most five proposals")
            if window.encoder_tokens != operation.start + operation.rows:
                raise ValidationError("incremental visibility must match this bounded tentative window")
        self._active = window
        return None

    def prepare(self, window, *, accepted, result_digest, pending_token=None):
        """Validate a decision before any irreversible local GPU finish."""
        hex_id(result_digest, 64, "prepared compact result")
        if pending_token is not None:
            integer(pending_token, "new pending token", 0, 129279)
        old = self._receipts.get(window.operation.number)
        if old is not None:
            proposal = self._decision(window, accepted, result_digest, pending_token)
            if old[:2] != (window, proposal):
                raise ValidationError("replayed commit changed the accepted prefix or terminal result")
            return old[2].commit_digest
        if self._active != window:
            raise ValidationError("prepare refers to a window that is not active")
        decision = self._decision(window, accepted, result_digest, pending_token)
        if self._prepared is not None and self._prepared != decision:
            raise ValidationError("active window has a conflicting prepared commit")
        self._prepared = decision
        return fingerprint(decision)

    def _decision(self, window, accepted, result_digest, pending_token):
        integer(accepted, "accepted target input rows", 0, window.operation.rows)
        if window.phase in {"encoder", "target"}:
            if accepted not in (0, window.operation.rows) or pending_token is not None:
                raise ValidationError("prefill commits whole fixed chunks and cannot sample a pending seed")
        elif window.phase == "draft":
            if accepted or pending_token is not None:
                raise ValidationError("draft proposal rows never enter persistent target context")
        elif not accepted and pending_token is not None:
            raise ValidationError("an aborted decode cannot publish a new pending output")
        if window.phase == "decode" and pending_token is not None and self.processed + accepted >= 1048576:
            raise ValidationError("no pending output token fits beyond the full context")
        return {"window": asdict(window), "accepted": accepted, "result_digest": result_digest,
                "pending_token": pending_token}

    def finish(self, window, *, commit_digest):
        """Call only after every local GPU/hash resource has finished successfully."""
        old = self._receipts.get(window.operation.number)
        if old is not None:
            if old[0] != window or old[2].commit_digest != commit_digest:
                raise ValidationError("repeated commit does not match the cached acknowledgment")
            return old[2]
        if self._active != window or self._prepared is None or fingerprint(self._prepared) != commit_digest:
            raise ValidationError("local finish lacks the matching prepared commit")
        accepted = self._prepared["accepted"]
        if window.phase == "encoder":
            self.encoder_cursor += accepted
        elif window.phase == "target":
            self.target_cursor += accepted
            if self.target_cursor == self.encoder_tokens:
                self.processed, self.phase = self.encoder_tokens, "decode"
        elif window.phase == "decode" and accepted:
            self.processed += accepted
            self.pending_token = self._prepared["pending_token"]
        receipt = PhaseReceipt(self.role, window.operation.digest, commit_digest, self.phase,
                               self.encoder_cursor, self.target_cursor, self.processed, self.pending_token)
        self._receipts[window.operation.number] = window, self._prepared, receipt
        while len(self._receipts) > self.replay_limit:
            self._receipts.popitem(last=False)
        self._next += 1
        self._active = self._prepared = None
        self._awaiting = receipt
        return receipt

    def seal(self, receipts, *, commit_digest):
        """Require all real GPU finish receipts before another window can run."""
        receipt = unanimous_receipts(receipts, expected_digest=commit_digest)
        own, = [value for value in receipts if value.role == self.role]
        if self._awaiting is None:
            if not any(number <= self._sealed_number and entry[2] == own
                       for number, entry in self._receipts.items()):
                raise ValidationError("seal does not refer to a retained globally acknowledged commit")
            return receipt
        if own != self._awaiting:
            raise ValidationError("seal differs from this stage's actual local finish receipt")
        self._sealed_number = self._next - 1
        self._awaiting = None
        return receipt

    def encoder_barrier(self, receipts, *, commit_digest):
        receipt = unanimous_receipts(receipts, expected_digest=commit_digest)
        if (self.phase != "encoder" or self._active is not None or self._awaiting is not None or
                self.encoder_cursor != self.encoder_tokens or
                receipt.phase != "encoder" or receipt.encoder_cursor != self.encoder_tokens or
                not self._receipts or next(reversed(self._receipts.values()))[2].commit_digest != commit_digest):
            raise ValidationError("exact target replay requires the complete matching all-stage encoder barrier")
        self._encoder_barrier = commit_digest

    def activate_target(self):
        """Call after successful SWA/pair rewind preserving all full-N sources."""
        if (self._encoder_barrier is None or self.phase != "encoder" or self._active is not None or
                self._awaiting is not None):
            raise ValidationError("target replay has no complete encoder barrier")
        self.phase, self.target_cursor = "target", 0
        self._encoder_barrier = None

    def publish_seed(self, *, token, history_digest):
        """Record a globally acknowledged seed without pretending it was forwarded."""
        integer(token, "published pending seed", 0, 129279)
        hex_id(history_digest, 64, "published token/image history")
        if (self.phase != "decode" or self._active is not None or self._awaiting is not None or
                self.pending_token is not None or self._seeded):
            raise ValidationError("initial seed publication requires a complete idle target pass")
        if self.processed >= 1048576:
            raise ValidationError("context is full; no new pending token may be emitted")
        self.pending_token, self.history_digest = token, history_digest
        self._seeded = True

    def publish_history(self, *, history_digest):
        hex_id(history_digest, 64, "published token/image history")
        if self.phase != "decode" or self._active is not None or self._awaiting is not None:
            raise ValidationError("history publication requires a committed idle decode boundary")
        self.history_digest = history_digest

    def fault(self):
        # Keep the uncertain cursor/operation for diagnostics. Only a coherent
        # newer reset can permit another model operation; weights stay loaded.
        self.phase = "faulted"
        self._active = self._prepared = None
