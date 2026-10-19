"""All-stage published history and Stage C RNG finalization before streaming."""
from collections import OrderedDict
from dataclasses import asdict, dataclass

from .history import TokenHistory
from .jsonutil import ValidationError, fingerprint, integer
from .phases import PhaseLedger
from .planner import Role
from .protocol import hex_id


@dataclass(frozen=True)
class OutputPlan:
    publication: str
    session: str
    request: str
    epoch: int
    processed: int
    parent_history: str
    history_digest: str
    sampling_digest: str
    output_ids: tuple[int, ...]
    pending_token: int | None
    initial_seed: bool
    stop_reason: str | None

    def __post_init__(self):
        for name in ("publication", "session", "request"):
            hex_id(getattr(self, name), 32, name)
        for name in ("parent_history", "history_digest", "sampling_digest"):
            hex_id(getattr(self, name), 64, name)
        integer(self.epoch, "publication epoch", 0, (1 << 63) - 1)
        integer(self.processed, "processed target rows", 1, 1048576)
        if (type(self.initial_seed) is not bool or not isinstance(self.output_ids, tuple) or
                not 1 <= len(self.output_ids) <= 6 or
                self.stop_reason is not None and not isinstance(self.stop_reason, str) or
                self.stop_reason not in {None, "eos", "length", "stop"}):
            raise ValidationError("invalid bounded terminal output publication")
        for token in self.output_ids:
            integer(token, "published output token", 0, 129279)
        if self.pending_token is not None:
            integer(self.pending_token, "published pending token", 0, 129279)
            if self.pending_token != self.output_ids[-1]:
                raise ValidationError("pending token must be the newly emitted correction/bonus or seed")
        elif self.stop_reason is None:
            raise ValidationError("a continuing decode requires its new pending token")
        if self.initial_seed and (len(self.output_ids) != 1 or self.pending_token is None):
            raise ValidationError("initial sampling emits exactly one unprocessed seed")

    @property
    def digest(self):
        return fingerprint(asdict(self))


@dataclass(frozen=True)
class OutputReceipt:
    role: Role
    plan_digest: str
    state: str
    history_digest: str
    processed: int
    pending_token: int | None

    def __post_init__(self):
        if not isinstance(self.role, Role) or self.state not in {"prepared", "committed", "finalized"}:
            raise ValidationError("invalid output acknowledgment role or state")
        hex_id(self.plan_digest, 64, "output acknowledgment")
        hex_id(self.history_digest, 64, "acknowledged published history")
        integer(self.processed, "acknowledged processed rows", 1, 1048576)
        if self.pending_token is not None:
            integer(self.pending_token, "acknowledged pending token", 0, 129279)


def unanimous_outputs(receipts, plan, *, state):
    if (not isinstance(receipts, (tuple, list)) or len(receipts) != 3 or
            any(not isinstance(r, OutputReceipt) for r in receipts) or {r.role for r in receipts} != set(Role)):
        raise ValidationError("output publication requires exactly one acknowledgment from every stage")
    for receipt in receipts:
        if (receipt.plan_digest, receipt.state, receipt.history_digest, receipt.processed, receipt.pending_token) != (
                plan.digest, state, plan.history_digest, plan.processed, plan.pending_token):
            raise ValidationError("stage output acknowledgment differs from the exact publication plan")
    return True


class PublicationLedger:
    def __init__(self, phase: PhaseLedger, history: TokenHistory, *, validate_sampling=None, finalize_sampling=None,
                 replay_limit=128):
        if not isinstance(phase, PhaseLedger) or not isinstance(history, TokenHistory):
            raise ValidationError("publication requires real phase and token/image history owners")
        callbacks = callable(validate_sampling) and callable(finalize_sampling)
        if phase.role == Role.TERMINAL:
            if not callbacks:
                raise ValidationError("stage C must validate/finalize its actual prepared sampling transaction")
        elif validate_sampling is not None or finalize_sampling is not None:
            raise ValidationError("only stage C owns sampling/RNG finalization")
        self.phase, self.history = phase, history
        if phase.session is None or phase.epoch < 0:
            raise ValidationError("publication owner must already have its admitted request/session epoch")
        self._owner_epoch = phase.session, phase.request, phase.epoch
        self.validate_sampling, self.finalize_sampling = validate_sampling, finalize_sampling
        self.replay_limit = integer(replay_limit, "published receipt history", 1, 4096)
        self._recent = OrderedDict()

    def _receipt(self, plan, state):
        return OutputReceipt(self.phase.role, plan.digest, state, plan.history_digest, plan.processed, plan.pending_token)

    def _current_epoch(self):
        if (self.phase.session, self.phase.request, self.phase.epoch) != self._owner_epoch or self.phase.phase == "faulted":
            raise ValidationError("output ledger belongs to an invalidated request/session epoch")

    def prepare(self, plan: OutputPlan):
        self._current_epoch()
        if not isinstance(plan, OutputPlan):
            raise ValidationError("output publication needs its typed immutable plan")
        cached = self._recent.get(plan.digest)
        if cached is not None:
            return cached["prepared"]
        if ((plan.session, plan.request, plan.epoch, plan.processed) != (
                self.phase.session, self.phase.request, self.phase.epoch, self.phase.processed) or
                plan.parent_history != self.history.digest or plan.parent_history != self.phase.history_digest or
                not self.phase.quiescent):
            raise ValidationError("output publication does not match the current committed session/history")
        if plan.initial_seed:
            self.phase.validate_seed(token=plan.pending_token, history_digest=plan.history_digest)
        elif self.phase.pending_token != plan.pending_token:
            raise ValidationError("output pending token differs from the committed GPU verification decision")
        expected_length = plan.processed + (plan.pending_token is not None)
        if (self.history.length + len(plan.output_ids) != expected_length or
                self.history.preview_append(plan.output_ids) != plan.history_digest):
            raise ValidationError("published output rows do not match processed-versus-pending history")
        if self.validate_sampling is not None and self.validate_sampling(plan) is not True:
            raise ValidationError("stage C did not validate its exact prepared sampling decision")
        if len(self._recent) >= self.replay_limit:
            key, entry = next(iter(self._recent.items()))
            if "finalized" not in entry:
                raise ValidationError("previous output publication has not finalized")
            del self._recent[key]
        self.phase.prepare_publication(plan.digest)
        prepared = self._receipt(plan, "prepared")
        self._recent[plan.digest] = {"plan": plan, "prepared": prepared}
        return prepared

    def commit(self, plan, receipts):
        self._current_epoch()
        unanimous_outputs(receipts, plan, state="prepared")
        cached = self._recent.get(plan.digest)
        if cached is None or cached["plan"] != plan:
            raise ValidationError("output commit has no matching local prepared decision")
        if "committed" in cached:
            return cached["committed"]
        try:
            self.history.append(plan.output_ids, expected_digest=plan.history_digest)
            if plan.initial_seed:
                self.phase.publish_seed(token=plan.pending_token, history_digest=plan.history_digest, stop_reason=plan.stop_reason)
            else:
                self.phase.publish_history(history_digest=plan.history_digest, stop_reason=plan.stop_reason)
            receipt = self._receipt(plan, "committed")
            cached["committed"] = receipt
            return receipt
        except BaseException:
            self.phase.fault()
            raise

    def finalize(self, plan, receipts):
        self._current_epoch()
        unanimous_outputs(receipts, plan, state="committed")
        cached = self._recent.get(plan.digest)
        if cached is None or "committed" not in cached:
            raise ValidationError("output finalization lacks this stage's actual history commit")
        if "finalized" in cached:
            return cached["finalized"]
        try:
            if self.finalize_sampling is not None:
                self.finalize_sampling(plan)
            self.phase.seal_publication(plan.digest)
            receipt = self._receipt(plan, "finalized")
            cached["finalized"] = receipt
            return receipt
        except BaseException:
            self.phase.fault()
            raise
