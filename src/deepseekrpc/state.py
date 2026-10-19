"""Bounded transaction journal and explicit session timeline.

This is state machinery, not a model executor. Committing local state alone does
not establish an all-rank commit; the coordinator must collect all acknowledgments.
"""
from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from enum import Enum

from .jsonutil import ValidationError, integer
from .protocol import U64, hex_id


class Lifecycle(str, Enum):
    IDLE = "idle"
    ACTIVE = "active"
    FAULTED = "faulted_session"


@dataclass(frozen=True)
class Operation:
    session: str
    request: str
    epoch: int
    number: int
    transaction: str
    digest: str
    start: int
    rows: int

    def __post_init__(self):
        for name in ("session", "request", "transaction"):
            hex_id(getattr(self, name), 32, name)
        hex_id(self.digest, 64, "operation payload")
        integer(self.epoch, "epoch", 0, U64)
        integer(self.number, "operation number", 1, U64)
        integer(self.start, "start position", 0, 1048576)
        integer(self.rows, "operation rows", 1, min(4096, 1048576 - self.start))


@dataclass(frozen=True)
class Row:
    logical_position: int
    generation: int
    data: bytes


@dataclass(frozen=True)
class Mutation:
    token_position: int
    key: tuple[str, int]
    previous: Row | None


class Session:
    def __init__(self, *, capacity: int, max_journal_bytes: int,
                 sources: dict[str, tuple[int, int]], replay_limit: int = 128,
                 state_budget_bytes: int = 64 * 1024 * 1024):
        self.capacity = integer(capacity, "capacity", 1, 1048576)
        self.max_journal_bytes = integer(max_journal_bytes, "journal budget", 1, 1 << 30)
        self.replay_limit = integer(replay_limit, "replay bound", 1, 4096)
        if not 1 <= len(sources) <= 64:
            raise ValidationError("a bounded admitted state-source registry is required")
        self.sources = dict(sources)
        integer(state_budget_bytes, "state byte budget", 1, 1 << 40)
        reservation = 0
        for source, (slots, row_bytes) in self.sources.items():
            if not isinstance(source, str) or not source or len(source) > 128:
                raise ValidationError("invalid state-source name")
            integer(slots, "source slot count", 1, capacity)
            integer(row_bytes, "source row byte count", 1, 1024 * 1024)
            # Conservative Python row/dict bookkeeping allowance is separate
            # from the actual payload and the tentative-write journal.
            reservation += slots * (row_bytes + 512)
        if reservation > state_budget_bytes:
            raise ValidationError("state source registry exceeds admitted persistent budget")
        self.state = Lifecycle.FAULTED
        self.session = None
        self.epoch = -1
        self.position = self.emitted = 0
        self.pending_token: int | None = None
        self.slots: dict[tuple[str, int], Row] = {}
        self._journal: list[Mutation] = []
        self._journal_bytes = 0
        self.active: Operation | None = None
        self._next = 1
        self._replays: OrderedDict[int, tuple[Operation, int]] = OrderedDict()

    def reset(self, session: str, epoch: int):
        hex_id(session, 32, "session")
        integer(epoch, "epoch", 0, U64)
        if epoch <= self.epoch:
            raise ValidationError("reset requires a strictly newer generation epoch")
        self.session, self.epoch = session, epoch
        self.position = self.emitted = 0
        self.pending_token = None
        self.slots.clear()
        self._journal.clear()
        self._journal_bytes = 0
        self.active = None
        self._next = 1
        self._replays.clear()
        self.state = Lifecycle.IDLE

    def begin(self, operation: Operation) -> int | None:
        if self.state == Lifecycle.FAULTED:
            raise ValidationError("uncertain session needs coordinated reset or restore")
        if operation.session != self.session or operation.epoch != self.epoch:
            raise ValidationError("stale or mismatched session epoch")
        replay = self._replays.get(operation.number)
        if replay is not None:
            if replay[0] != operation:
                raise ValidationError("conflicting replay identity or payload")
            return replay[1]
        if self.active is not None:
            if self.active == operation:
                raise ValidationError("operation is already tentative; wait for its outcome")
            raise ValidationError("only one active operation is allowed")
        if operation.number != self._next or operation.start != self.position:
            raise ValidationError("out-of-order operation or timeline position")
        if operation.start + operation.rows > self.capacity:
            raise ValidationError("context capacity exceeded; no truncation performed")
        self.active = operation
        self.state = Lifecycle.ACTIVE
        return None

    def write(self, source: str, slot: int, *, token_position: int,
              logical_position: int, generation: int, data: bytes):
        if self.active is None:
            raise ValidationError("state writes require an active transaction")
        if source not in self.sources:
            raise ValidationError("state source was not admitted")
        slots, row_bytes = self.sources[source]
        integer(slot, "physical slot", 0, slots - 1)
        integer(logical_position, "logical position", 0, self.capacity - 1)
        integer(generation, "slot generation", 0, U64)
        integer(token_position, "mutation token position", self.active.start,
                self.active.start + self.active.rows - 1)
        if not isinstance(data, bytes) or len(data) != row_bytes:
            raise ValidationError("invalid state row identity or payload")
        if self._journal and token_position < self._journal[-1].token_position:
            raise ValidationError("journal mutations must follow token timeline order")
        key = (source, slot)
        previous = self.slots.get(key)
        if previous is not None and generation < previous.generation:
            raise ValidationError("stale physical slot mapping")
        cost = len(data) + (len(previous.data) if previous else 0) + 128
        if self._journal_bytes + cost > self.max_journal_bytes:
            raise ValidationError("transaction journal exceeds admitted budget")
        self._journal.append(Mutation(token_position, key, previous))
        self._journal_bytes += cost
        self.slots[key] = Row(logical_position, generation, data)

    def commit(self, operation: Operation, accepted: int) -> int:
        integer(accepted, "accepted prefix length", 0, operation.rows)
        if self.state == Lifecycle.FAULTED or operation.session != self.session or operation.epoch != self.epoch:
            raise ValidationError("commit targets a faulted or stale session")
        replay = self._replays.get(operation.number)
        if replay is not None:
            if replay != (operation, accepted):
                raise ValidationError("conflicting commit replay")
            return operation.start + accepted
        if self.state != Lifecycle.ACTIVE or self.active != operation:
            raise ValidationError("commit does not match active operation")
        boundary = operation.start + accepted
        for mutation in reversed(self._journal):
            if mutation.token_position < boundary:
                continue
            if mutation.previous is None:
                self.slots.pop(mutation.key, None)
            else:
                self.slots[mutation.key] = mutation.previous
        self.position = boundary
        self._replays[operation.number] = (operation, accepted)
        while len(self._replays) > self.replay_limit:
            self._replays.popitem(last=False)
        self._next += 1
        self._journal.clear()
        self._journal_bytes = 0
        self.active = None
        self.state = Lifecycle.IDLE
        return self.position

    def abort(self, operation: Operation):
        return self.commit(operation, 0)

    def fault(self):
        # Keep model ownership outside this object; uncertain inference state
        # cannot be reused even if weights and CUDA allocations remain healthy.
        self.state = Lifecycle.FAULTED

    def row(self, source: str, slot: int, logical_position: int, generation: int) -> bytes:
        if self.state == Lifecycle.FAULTED:
            raise ValidationError("faulted session state cannot be consumed")
        row = self.slots[(source, slot)]
        if (row.logical_position, row.generation) != (logical_position, generation):
            raise ValidationError("stale logical-to-physical row mapping")
        return row.data
