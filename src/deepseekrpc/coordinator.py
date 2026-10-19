"""Stage A's real three-stage CED/speculative scheduler and publication barrier."""
from dataclasses import dataclass
import threading
import uuid

from .execution import StepInput
from .history import TokenHistory
from .jsonutil import ValidationError, fingerprint, integer
from .phases import unanimous_receipts
from .planner import Role
from .prompt_wire import PromptSpec, send_prompt
from .protocol import Frame, Kind
from .publication import OutputPlan, unanimous_outputs
from .request_settings import RequestSettings
from .runtime_messages import receipt_from, wire
from .stage_service import COMMAND_KINDS, StageService


@dataclass(frozen=True)
class TokenBatch:
    output_ids: tuple[int, ...]
    stop_reason: str | None
    processed: int
    pending_token: int | None
    history_digest: str
    mode: str


class ClusterCoordinator:
    """One request at a time; all model work goes through persistent services.

    Peers begin a window before A sends dependencies. Every local GPU finish
    and published history commit is acknowledged by all three actual services
    before a yielded token can become visible to the API client.
    """
    def __init__(self, host: StageService, *, operation_timeout=300):
        if not isinstance(host, StageService) or host.role != Role.HOST:
            raise ValidationError("cluster coordinator requires the actual persistent stage A service")
        integer(operation_timeout, "cluster operation deadline", 1, 3600)
        self.host, self.timeout = host, operation_timeout
        self._request_lock = threading.Lock()
        self._cancelled, self._cancel_lock = threading.Event(), threading.Lock()
        self._spec = self._window = None
        self._number, self._cancel_sent = 1, False

    def _frame(self, role, action, *, kind=None):
        spec, window, zero = self._spec, self._window, "0" * 32
        return Frame(kind or COMMAND_KINDS[action], self.host.data.run, Role.HOST, role,
                     spec.session if spec else zero, spec.request if spec else zero,
                     spec.epoch if spec else 0, window.operation.number if window else 0, 0,
                     window.operation.transaction if window else spec.transaction if spec else zero,
                     window.operation.start if window else 0, window.operation.rows if window else 0)

    def _check_cancel(self):
        if self._cancelled.is_set():
            raise InterruptedError("client request was cancelled; its session requires coherent recovery")

    def _all(self, body, *, prompt=None, allow_cancelled=False):
        if not allow_cancelled:
            self._check_cancel()
        action = body["action"]
        futures = {role: self.host.controls[role].submit(self._frame(role, action), body)
                   for role in (Role.MIDDLE, Role.TERMINAL)}
        local = self.host.submit_local(body, prompt=prompt)
        if prompt is not None:
            for role in (Role.MIDDLE, Role.TERMINAL):
                send_prompt(self.host.data.connections[role], self._spec, prompt, cancelled=self._cancelled.is_set)
        results = {role: self.host.controls[role].wait(future, timeout=self.timeout) for role, future in futures.items()}
        results[Role.HOST] = local.result(timeout=self.timeout)
        if not allow_cancelled:
            self._check_cancel()
        return results

    def _terminal(self, action):
        self._check_cancel()
        rpc = self.host.controls[Role.TERMINAL]
        result = rpc.call(self._frame(Role.TERMINAL, action), {"action": action}, timeout=self.timeout)
        self._check_cancel()
        return result

    def admit(self):
        peers = [self.host.status()]
        futures = {role: self.host.controls[role].submit(self._frame(role, "status"), {"action": "status"})
                   for role in (Role.MIDDLE, Role.TERMINAL)}
        for role, future in futures.items():
            peer = self.host.controls[role].wait(future, timeout=self.timeout)
            if peer.get("stage") != int(role):
                raise ValidationError("loaded peer status differs from its authenticated control role")
            peers.append(peer)
        identities = [{key: peer[key] for key in ("stage", "run", "identity")} for peer in peers]
        results = self._all({"action": "admit", "peers": identities})
        if any(result != {"ready": True, "identity": self.host.execution.stage.identity} for result in results.values()):
            raise ValidationError("not every actual stage admitted the complete runtime/plan identity")
        return peers

    def _begin_window(self, inputs, *, phase, start, encoder_tokens):
        history = self.host.execution.history
        self._window = inputs.make_window(session=self._spec.session, request=self._spec.request,
            epoch=self._spec.epoch, number=self._number, transaction=uuid.uuid4().hex, start=start,
            phase=phase, history_digest=history.digest, encoder_tokens=encoder_tokens)
        results = self._all({"action": "step", "window": wire(self._window), "inputs": wire(inputs)})
        for role, result in results.items():
            if (set(result) != {"operation_digest", "proposal_ids"} or
                    result["operation_digest"] != self._window.operation.digest or
                    (phase != "draft" or role != Role.TERMINAL) and result["proposal_ids"] is not None):
                raise ValidationError("actual stage did not finish the exact dispatched model window")
        return results

    def _commit_window(self, *, accepted, result_digest, pending_token=None):
        decisions = self._all({"action": "prepare", "accepted": accepted, "result_digest": result_digest,
                               "pending_token": pending_token, "aborted": False})
        digests = [decision.get("commit_digest") for decision in decisions.values()]
        if len(set(digests)) != 1 or digests[0] is None:
            raise ValidationError("actual stages disagree on the exact GPU commit decision")
        digest = digests[0]
        finished = self._all({"action": "finish", "commit_digest": digest})
        receipts = []
        for role in Role:
            receipt = receipt_from(finished[role]["receipt"])
            if receipt.role != role:
                raise ValidationError("GPU finish receipt differs from its actual stage channel")
            receipts.append(receipt)
        unanimous_receipts(receipts, expected_digest=digest)
        self._all({"action": "seal", "receipts": [wire(r) for r in receipts], "commit_digest": digest})
        self._number += 1
        self._window = None
        return receipts, digest

    def _prefill(self):
        history, extent = self.host.execution.history, self._spec.tokens
        for phase in ("encoder", "target"):
            last = None
            for start in range(0, extent, 64):
                self._check_cancel()
                tokens, mask = history.rows(start, min(64, extent - start))
                self._begin_window(StepInput(tokens, mask), phase=phase, start=start, encoder_tokens=extent)
                result = fingerprint({"window": wire(self._window), "completed_phase_chunk": True})
                last = self._commit_window(accepted=len(tokens), result_digest=result)
            if phase == "encoder":
                receipts, digest = last
                self._all({"action": "target", "receipts": [wire(r) for r in receipts], "commit_digest": digest})

    def _publish(self, decision, *, initial, mode):
        history, ledger = self.host.execution.history, self.host.execution.ledger
        if set(decision) != {"digest", "accepted_target_rows", "output_ids", "pending_token", "stop_reason"}:
            raise ValidationError("stage C returned unknown compact sampling decision fields")
        ids = decision["output_ids"]
        if not isinstance(ids, list) or not 1 <= len(ids) <= 6:
            raise ValidationError("stage C returned an invalid bounded output token batch")
        plan = OutputPlan(uuid.uuid4().hex, self._spec.session, self._spec.request, self._spec.epoch,
                          ledger.processed, history.digest, history.preview_append(ids), decision["digest"],
                          tuple(ids), decision["pending_token"], initial, decision["stop_reason"])
        receipts = None
        for action, state in (("output_prepare", "prepared"), ("output_commit", "committed"), ("output_finalize", "finalized")):
            body = {"action": action, "plan": wire(plan)}
            if receipts is not None:
                body["receipts"] = [wire(r) for r in receipts]
            results = self._all(body)
            receipts = []
            for role in Role:
                receipt = receipt_from(results[role]["receipt"], output=True)
                if receipt.role != role:
                    raise ValidationError("published output receipt differs from its authenticated stage channel")
                receipts.append(receipt)
            unanimous_outputs(receipts, plan, state=state)
        return TokenBatch(plan.output_ids, plan.stop_reason, plan.processed, plan.pending_token, plan.history_digest, mode)

    def cancel(self):
        """Safe from the API/disconnect thread; does not wait for model work."""
        self._cancelled.set()
        with self._cancel_lock:
            if self._cancel_sent or self._spec is None:
                return
            self._cancel_sent = True
            spec = self._spec
            self.host.cancel(session=spec.session, request=spec.request, epoch=spec.epoch)
            for role in (Role.MIDDLE, Role.TERMINAL):
                try:
                    self.host.controls[role].submit(self._frame(role, "fault", kind=Kind.CANCEL), {"action": "cancel"})
                except (ConnectionError, ValidationError):
                    pass  # Channel failure itself invalidates that peer service.

    def generate(self, prompt: TokenHistory, settings: RequestSettings):
        if not isinstance(prompt, TokenHistory) or not isinstance(settings, RequestSettings):
            raise ValidationError("generation needs validated full original token/image history and immutable settings")
        if prompt.identity != self.host.execution.stage.identity or not 1 <= prompt.length < prompt.capacity:
            raise ValidationError("prompt identity/context leaves no admitted output capacity")
        if not self._request_lock.acquire(blocking=False):
            raise RuntimeError("the three-stage runtime already has an active request")
        complete = False
        try:
            self._cancelled.clear()
            self._cancel_sent, self._window, self._number = False, None, 1
            self._spec = None
            peers = self.admit()
            epoch = max(peer["minimum_next_epoch"] for peer in peers)
            self._spec = PromptSpec.from_history(prompt, session=uuid.uuid4().hex, request=uuid.uuid4().hex,
                                                epoch=epoch, transaction=uuid.uuid4().hex)
            resets = self._all({"action": "reset", "prompt": wire(self._spec), "settings": settings.as_dict()}, prompt=prompt)
            if any((value.get("history_digest"), value.get("epoch"), value.get("settings_digest")) !=
                   (prompt.digest, epoch, settings.digest) for value in resets.values()):
                raise ValidationError("stage reset did not admit the same original history and request controls")
            self._prefill()
            decision = self._terminal("sample_seed")
            if decision.get("accepted_target_rows") != 0:
                raise ValidationError("initial sampling cannot claim a forwarded target input")
            batch = self._publish(decision, initial=True, mode="seed")
            while True:
                complete = batch.stop_reason is not None
                yield batch
                if complete:
                    return
                self._check_cancel()
                ledger = self.host.execution.ledger
                seed, start = ledger.pending_token, ledger.processed
                self._terminal("sample_begin")
                proposals = ()
                mode = "speculative"
                if settings.speculative and prompt.capacity - start >= 6:
                    results = self._begin_window(StepInput((seed, 128799, 128799, 128799, 128799), (True,) * 5),
                        phase="draft", start=start, encoder_tokens=start + 5)
                    proposed = results[Role.TERMINAL]["proposal_ids"]
                    if not isinstance(proposed, list) or not 1 <= len(proposed) <= 5:
                        raise ValidationError("stage C returned an invalid DSpark proposal prefix")
                    proposals = tuple(integer(token, "draft proposal", 0, 129279) for token in proposed)
                    self._commit_window(accepted=0, result_digest=fingerprint({"proposals": proposals, "window": wire(self._window)}))
                else:
                    mode = "normal_disabled" if not settings.speculative else "normal_context_tail"
                    self._terminal("normal_decode")
                inputs = StepInput((seed, *proposals), (True,) * (1 + len(proposals)))
                self._begin_window(inputs, phase="decode", start=start, encoder_tokens=start + len(inputs.tokens))
                decision = self._terminal("verify")
                accepted = integer(decision.get("accepted_target_rows"), "accepted verification inputs", 1, len(inputs.tokens))
                self._commit_window(accepted=accepted, result_digest=decision["digest"], pending_token=decision["pending_token"])
                batch = self._publish(decision, initial=False, mode=mode)
        finally:
            if not complete:
                self.cancel()
            self._window = None
            self._request_lock.release()
