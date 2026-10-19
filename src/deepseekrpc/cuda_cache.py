"""Bounded real CUDA expert tier over desktop RAM or B's locked resident store.

The caller admits the whole process working set separately. This pool reserves
one fixed allocation, uses synchronous bounded uploads, and retains each slot
until the recorded use event completes. It does not reconstruct expert weights.
"""
from __future__ import annotations

from collections import OrderedDict
from contextlib import contextmanager
from dataclasses import dataclass
import threading

from .inventory import MAX_READ
from .jsonutil import ValidationError, integer
from .packed import DevicePackedLinear, PackedBundleLayout
from .storage import ExpertKey, RAMCache


class DesktopUploader:
    """Two fixed MiB plus a bounded one-MiB transient read/slice reservation."""
    staging_bytes = 3 * MAX_READ

    def __init__(self, source, cache: RAMCache):
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError("desktop upload tier requires a real CUDA device")
        self.source, self.cache = source, cache
        self._scratch = bytearray(MAX_READ)
        self._scratch_tensor = torch.frombuffer(self._scratch, dtype=torch.uint8)
        self._pinned = torch.empty(MAX_READ, dtype=torch.uint8, pin_memory=True)
        self._mutex = threading.Lock()

    def _read(self, layout):
        # RAMCache reserves payload plus this whole-bundle construction peak.
        data = bytearray(layout.nbytes)
        for _, base, matrix in layout.matrices:
            for part in matrix.parts:
                for offset in range(0, part.info.nbytes, MAX_READ):
                    chunk = self.source.read_range(part.info.name, offset, min(MAX_READ, part.info.nbytes - offset))
                    start = base + part.offset + offset
                    data[start:start + len(chunk)] = chunk
        return bytes(data)

    def upload(self, key, layout, destination):
        with self._mutex, self.cache.lease(key, nbytes=layout.nbytes, loader=lambda: self._read(layout)) as (data, _):
            for offset in range(0, layout.nbytes, MAX_READ):
                count = min(MAX_READ, layout.nbytes - offset)
                self._scratch[:count] = data[offset:offset + count]
                self._pinned[:count].copy_(self._scratch_tensor[:count])
                destination[offset:offset + count].copy_(self._pinned[:count], non_blocking=False)


class ResidentUploader:
    """No inference file-loader path and no pageable tensor payload copies."""
    staging_bytes = MAX_READ

    def __init__(self, source):
        from .resident_weights import LockedOwnedWeights
        if not isinstance(source, LockedOwnedWeights) or source.closed:
            raise ValidationError("B's GPU uploader requires its completed locked resident preload")
        self._transfer = source.transfer()
        self.source = source

    def upload(self, key, layout, destination):
        for _, base, matrix in layout.matrices:
            for part in matrix.parts:
                for offset in range(0, part.info.nbytes, MAX_READ):
                    count = min(MAX_READ, part.info.nbytes - offset)
                    start = base + part.offset + offset
                    self._transfer.copy(part.info.name, offset, destination[start:start + count])


@dataclass
class _Entry:
    key: ExpertKey
    slot: int
    generation: int
    layout: PackedBundleLayout
    matrices: dict
    event: object = None
    leases: int = 0


class _Lease:
    def __init__(self, entry, caller_stream, use_stream):
        self._entry, self._generation = entry, entry.generation
        self._caller_stream, self._open = caller_stream, True
        self._use_stream = use_stream

    def linear(self, name, inputs):
        import torch
        if not self._open or self._entry.generation != self._generation:
            raise ValidationError("GPU expert lease is closed or has a stale generation")
        if name not in self._entry.matrices:
            raise ValidationError("unknown expert projection")
        if not inputs.is_cuda or torch.cuda.current_stream(inputs.device) != self._use_stream:
            raise ValidationError("expert execution must use the lease's tracked CUDA stream")
        inputs.record_stream(torch.cuda.current_stream(inputs.device))
        output = self._entry.matrices[name].forward(inputs)
        output.record_stream(self._caller_stream)
        return output


class DeviceExpertCache:
    def __init__(self, *, admitted_bytes: int, slot_bytes: int, uploader, device="cuda:0"):
        import torch
        integer(admitted_bytes, "GPU expert pool bytes", 256, 1 << 40)
        integer(slot_bytes, "GPU expert slot bytes", 256, admitted_bytes)
        if slot_bytes % 256:
            raise ValidationError("GPU expert slots require 256-byte alignment")
        self.slots = min(admitted_bytes // slot_bytes, 65536)
        self.allocation_bytes = self.slots * slot_bytes
        self.slot_bytes, self.uploader, self.device = slot_bytes, uploader, torch.device(device)
        if self.device.type != "cuda" or not torch.cuda.is_available():
            raise RuntimeError("GPU expert cache requires a real CUDA device")
        self._mutex = threading.RLock()
        self._entries = OrderedDict()
        self._free = list(range(self.slots - 1, -1, -1))
        self._generations = [0] * self.slots
        self.closed = False
        self.hits = self.loads = self.evictions = 0
        with torch.cuda.device(self.device):
            self.stream = torch.cuda.Stream(device=self.device)
            with torch.cuda.stream(self.stream):
                self._arena = torch.zeros(self.allocation_bytes, dtype=torch.uint8, device=self.device)
            self.stream.synchronize()

    @contextmanager
    def lease(self, key: ExpertKey, layout: PackedBundleLayout):
        import torch
        if not isinstance(key, ExpertKey) or key.component != "bundle" or layout.nbytes > self.slot_bytes:
            raise ValidationError("expert identity/layout exceeds the admitted GPU slot")
        # Serialize CPU ownership; control/cancellation use independent threads.
        with self._mutex, torch.cuda.device(self.device):
            if self.closed:
                raise ValidationError("GPU expert cache is closed")
            caller = torch.cuda.current_stream(self.device)
            self.stream.wait_stream(caller)
            entry = self._entries.get(key)
            with torch.cuda.stream(self.stream):
                if entry is not None:
                    if entry.layout != layout:
                        raise ValidationError("GPU expert identity has a conflicting packed layout")
                    self.hits += 1
                    self._entries.move_to_end(key)
                else:
                    if not self._free:
                        victim_key = next((candidate for candidate, value in self._entries.items()
                                           if value.leases == 0 and (value.event is None or value.event.query())), None)
                        if victim_key is None:
                            raise ValidationError("GPU slots are in use; apply bounded backpressure")
                        victim = self._entries.pop(victim_key)
                        for matrix in victim.matrices.values():
                            matrix.close()
                        self._free.append(victim.slot)
                        self.evictions += 1
                    slot = self._free.pop()
                    self._generations[slot] += 1
                    storage = self._arena.narrow(0, slot * self.slot_bytes, layout.nbytes)
                    matrices = {}
                    try:
                        storage.zero_()
                        self.uploader.upload(key, layout, storage)
                        for name, offset, matrix in layout.matrices:
                            matrices[name] = DevicePackedLinear(matrix, storage.narrow(0, offset, matrix.nbytes))
                    except BaseException:
                        self.stream.synchronize()
                        for matrix in matrices.values():
                            matrix.close()
                        self._free.append(slot)
                        raise
                    entry = _Entry(key, slot, self._generations[slot], layout, matrices)
                    self._entries[key] = entry
                    self.loads += 1
                entry.leases += 1
                handle = _Lease(entry, caller, self.stream)
                try:
                    yield handle
                finally:
                    handle._open = False
                    entry.leases -= 1
                    event = torch.cuda.Event()
                    event.record(self.stream)
                    entry.event = event
                    caller.wait_event(event)

    def close(self):
        with self._mutex:
            if not self.closed:
                self.stream.synchronize()
                for entry in self._entries.values():
                    for matrix in entry.matrices.values():
                        matrix.close()
                self._entries.clear()
                self._arena = None
                self.closed = True
