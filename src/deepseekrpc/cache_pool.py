"""Eager BF16 row pools with independent slot maps and bounded CUDA undo logs."""
from __future__ import annotations

from dataclasses import dataclass
from contextlib import contextmanager
import threading

import numpy as np

from .jsonutil import ValidationError, integer
from .state import Operation

I64_MAX = (1 << 63) - 1


@dataclass(frozen=True)
class SlotMutation:
    token_position: int
    slot: int
    previous_logical: int
    previous_generation: int


class SlotMap:
    """Fixed host metadata, shared by actual CUDA storage and its state tests."""
    def __init__(self, slots: int, *, context: int, offset: int, max_mutations: int = 64):
        self.slots = integer(slots, "cache slots", 1, 1048576)
        self.context = integer(context, "cache logical capacity", slots, 1048576)
        self.offset = integer(offset, "local physical slot offset", 0, slots - 1)
        self.max_mutations = integer(max_mutations, "undo row capacity", 1, 4096)
        self.logical = np.full(slots, -1, dtype=np.int64)
        self.generations = np.zeros(slots, dtype=np.int64)
        self.epoch, self.session, self.active = -1, None, None
        self.journal: list[SlotMutation] = []

    @property
    def metadata_bytes(self):
        return self.logical.nbytes + self.generations.nbytes

    def reset(self, session: str, epoch: int):
        from .protocol import hex_id
        hex_id(session, 32, "cache session")
        integer(epoch, "cache epoch", 0, I64_MAX)
        if self.active is not None or epoch <= self.epoch:
            raise ValidationError("cache reset requires quiescence and a newer epoch")
        self.session, self.epoch = session, epoch
        self.logical.fill(-1)
        self.generations.fill(0)

    def discard_for_reset(self, session: str, epoch: int):
        """Invalidate an uncertain old window under a strictly newer epoch."""
        from .protocol import hex_id
        hex_id(session, 32, "cache session")
        integer(epoch, "cache epoch", 0, I64_MAX)
        if epoch <= self.epoch:
            raise ValidationError("discarded cache recovery requires a newer epoch")
        self.active = None
        self.journal.clear()
        self.reset(session, epoch)

    def begin(self, operation: Operation):
        if (self.active is not None or operation.epoch != self.epoch or operation.session != self.session or
                operation.rows > 64 or operation.start + operation.rows > self.context):
            raise ValidationError("cache window is active, stale or exceeds its admitted bounds")
        self.active = operation

    def prepare(self, logical_positions, token_positions):
        if self.active is None:
            raise ValidationError("cache mutation requires an active window")
        if len(logical_positions) != len(token_positions) or not 1 <= len(logical_positions) <= 64:
            raise ValidationError("cache row batch is empty or outside the admitted bound")
        if len(self.journal) + len(logical_positions) > self.max_mutations:
            raise ValidationError("cache undo row capacity exhausted")
        mutations, slots, generations = [], [], []
        previous_token = self.journal[-1].token_position if self.journal else self.active.start
        seen = set()
        for logical, position in zip(logical_positions, token_positions):
            integer(logical, "cache logical position", 0, self.context - 1)
            integer(position, "cache producing token", previous_token, self.active.start + self.active.rows - 1)
            previous_token = position
            slot = (logical + self.offset) % self.slots
            if slot in seen:
                raise ValidationError("one scatter batch repeats a physical cache slot")
            seen.add(slot)
            generation = int(self.generations[slot])
            if generation == I64_MAX:
                raise ValidationError("cache generation exhausted; coordinated new epoch required")
            mutations.append(SlotMutation(position, slot, int(self.logical[slot]), generation))
            slots.append(slot)
            generations.append(generation)
        return mutations, slots, generations

    def reserve(self, logical_positions, token_positions):
        mutations, slots, generations = self.prepare(logical_positions, token_positions)
        # Validate the entire batch before modifying any host metadata.
        for logical, mutation in zip(logical_positions, mutations):
            self.logical[mutation.slot] = logical
            self.generations[mutation.slot] = mutation.previous_generation + 1
        self.journal.extend(mutations)
        return slots, generations

    def physical(self, logical_positions):
        result = []
        for logical in logical_positions:
            if logical == -1:
                result.append(-1)
                continue
            integer(logical, "visible cache logical position", 0, self.context - 1)
            slot = (logical + self.offset) % self.slots
            if self.logical[slot] != logical or self.generations[slot] == 0:
                raise ValidationError("cache row is absent or its physical slot was overwritten")
            result.append(slot)
        return result

    def rejected(self, operation: Operation, accepted: int):
        integer(accepted, "cache accepted prefix", 0, operation.rows)
        if self.active != operation:
            raise ValidationError("cache commit targets another operation")
        boundary = operation.start + accepted
        return [(index, item) for index, item in reversed(list(enumerate(self.journal)))
                if item.token_position >= boundary]

    def finish(self, operation: Operation, accepted: int):
        for _, mutation in self.rejected(operation, accepted):
            self.logical[mutation.slot] = mutation.previous_logical
            self.generations[mutation.slot] = mutation.previous_generation
        self.journal.clear()
        self.active = None


class DeviceRowPool:
    """One eager unquantized CUDA allocation, with no quantized/CPU fallback.

    Each source or shadow owns this map independently. Register these pools in
    the stage's complete memory admission ledger before construction. The fixed
    host metadata, GPU generations, undo rows and transfer scratch count in
    addition to BF16 payload bytes; this is not proof of WDDM residency.
    """
    def __init__(self, slots: int, width: int, *, context: int, offset: int,
                 admitted_bytes: int, device="cuda:0", max_mutations: int = 64):
        import torch
        integer(width, "BF16 cache width", 1, 512)
        integer(admitted_bytes, "BF16 cache pool admission", 1, 1 << 40)
        self.map = SlotMap(slots, context=context, offset=offset, max_mutations=max_mutations)
        self.width, self.device = width, torch.device(device)
        if self.device.type != "cuda" or not torch.cuda.is_available():
            raise RuntimeError("BF16 cache pools require a real CUDA device")
        if self.device.index is None:
            self.device = torch.device("cuda", torch.cuda.current_device())
        # Worst-case 64-row gather/scatter metadata and output staging, plus the
        # persistent map and journal bookkeeping allowance, are explicit.
        self.payload_bytes = slots * width * 2
        self.allocation_bytes = self.payload_bytes + slots * 8 + max_mutations * (width * 2 + 256) + 64 * (width * 2 + 32)
        self.admission_bytes = self.allocation_bytes + self.map.metadata_bytes
        if self.admission_bytes > admitted_bytes:
            raise ValidationError("eager BF16 payload, map, generations and undo exceed admission")
        self._mutex, self._needs_rollback, self.closed = threading.RLock(), False, False
        with torch.cuda.device(self.device):
            self.data = torch.zeros((slots, width), dtype=torch.bfloat16, device=self.device)
            self.generations = torch.zeros(slots, dtype=torch.int64, device=self.device)
            self._undo = torch.empty((max_mutations, width), dtype=torch.bfloat16, device=self.device)
            self._undo.zero_()
            torch.cuda.synchronize(self.device)

    def _open(self):
        if self.closed:
            raise ValidationError("BF16 cache pool is closed")

    def reset(self, session: str, epoch: int):
        import torch
        with self._mutex, torch.cuda.device(self.device):
            self._open()
            self.map.reset(session, epoch)
            self.generations.zero_()
            torch.cuda.synchronize(self.device)
            self._needs_rollback = False

    def begin(self, operation: Operation):
        with self._mutex:
            self._open()
            if self._needs_rollback:
                raise ValidationError("cache requires rollback before another window")
            self.map.begin(operation)

    def discard_for_reset(self, session: str, epoch: int):
        """Quiesce CUDA and invalidate all cache validity, retaining allocation."""
        import torch
        with self._mutex, torch.cuda.device(self.device):
            self._open()
            torch.cuda.synchronize(self.device)
            self.map.discard_for_reset(session, epoch)
            self.generations.zero_()
            torch.cuda.synchronize(self.device)
            self._needs_rollback = False

    def rewind_ring_for_replay(self, session: str, epoch: int):
        """Clear a quiescent SWA ring for the full exact target pass.

        The phase coordinator must first validate the immutable token/image
        history. Compressed source/shadow pools must never use this operation.
        """
        import torch
        with self._mutex, torch.cuda.device(self.device):
            self._open()
            if (self.map.slots != 128 or self.width != 512 or self.map.active is not None or
                    self._needs_rollback or session != self.map.session or epoch != self.map.epoch):
                raise ValidationError("SWA rewind requires the current quiescent 128-row ring")
            self.map.logical.fill(-1)
            self.map.generations.fill(0)
            self.generations.zero_()
            torch.cuda.synchronize(self.device)

    def write(self, logical_positions, token_positions, values):
        import torch
        from .cuda_ops import scatter_slots
        with self._mutex, torch.cuda.device(self.device):
            self._open()
            if (self._needs_rollback or values.device != self.device or values.dtype != torch.bfloat16 or
                    values.shape != (len(logical_positions), self.width) or not values.is_contiguous()):
                raise ValidationError("cache scatter state, shape or BF16 storage mismatch")
            start = len(self.map.journal)
            _, slots, previous_generations = self.map.prepare(logical_positions, token_positions)
            indices = torch.tensor(slots, dtype=torch.int32, device=self.device)
            previous = torch.tensor(previous_generations, dtype=torch.int64, device=self.device)
            torch.index_select(self.data, 0, indices.long(), out=self._undo[start:start + len(slots)])
            # Finish the backup before recording a mutation. An allocation or
            # gather failure must never leave an undo entry with uninitialized
            # bytes that a later abort could copy into an intact cache row.
            torch.cuda.synchronize(self.device)
            self.map.reserve(logical_positions, token_positions)
            try:
                scatter_slots(self.data, self.generations, indices, previous, values)
                self.generations.index_copy_(0, indices.long(), previous + 1)
                torch.cuda.synchronize(self.device)
            except BaseException:
                self._needs_rollback = True
                raise

    def physical(self, logical_positions):
        with self._mutex:
            self._open()
            if self._needs_rollback:
                raise ValidationError("failed tentative cache writes cannot be consumed")
            return self.map.physical(logical_positions)

    @contextmanager
    def read_window(self, operation: Operation):
        import torch
        with self._mutex, torch.cuda.device(self.device):
            self._open()
            if self._needs_rollback or self.map.active != operation:
                raise ValidationError("cache consumer targets a failed or stale window")
            try:
                yield self.data
            finally:
                # A control thread cannot restore or reset slots while queued
                # readers still use them. Cancellation signals are separate.
                torch.cuda.synchronize(self.device)

    def finish(self, operation: Operation, accepted: int):
        import torch
        with self._mutex, torch.cuda.device(self.device):
            self._open()
            if self._needs_rollback and accepted != 0:
                raise ValidationError("failed cache operation requires complete abort")
            rejected = self.map.rejected(operation, accepted)
            try:
                for index, mutation in rejected:
                    self.data[mutation.slot].copy_(self._undo[index])
                    self.generations[mutation.slot] = mutation.previous_generation
                torch.cuda.synchronize(self.device)
            except BaseException:
                self._needs_rollback = True
                raise
            self.map.finish(operation, accepted)
            self._needs_rollback = False

    def close(self):
        import torch
        with self._mutex:
            if not self.closed:
                torch.cuda.synchronize(self.device)
                self.data = self.generations = self._undo = None
                self.closed = True
