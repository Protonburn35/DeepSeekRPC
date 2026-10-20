"""Exact bounded host storage and leased RAM cache.

GPU promotion and physical residency are separate mechanisms. This module never
labels RAM bytes as GPU-ready or assumes pageable storage proves residency.
"""
from __future__ import annotations

from collections import OrderedDict
from contextlib import contextmanager
from dataclasses import dataclass
import hashlib
import threading
import time
from . import metrics
from typing import Callable

from .jsonutil import ValidationError, integer


@dataclass(frozen=True)
class ExpertKey:
    model: str
    layer: int
    expert: int
    component: str
    quantization: str

    def __post_init__(self):
        from .protocol import hex_id
        hex_id(self.model, 64, "model")
        hex_id(self.quantization, 64, "quantization")
        integer(self.layer, "expert layer", 0, 42)
        integer(self.expert, "expert index", 0, 383)
        if self.component not in {"bundle", "w1", "w2", "w3"}:
            raise ValidationError("unknown expert component")


class ResidentStore:
    """Preload-owned bytes, seal once, then expose no inference file-loader path.

    Bytes in this portable store are not claimed to be physically locked. Native
    Windows admission must lock/register the real backing allocation separately.
    """
    def __init__(self, budget: int):
        self.budget = integer(budget, "resident byte budget", 1, 1 << 40)
        self.used = 0
        self._items: dict[ExpertKey, bytes] = {}
        self.sealed = False

    def preload(self, key: ExpertKey, chunks, *, expected_bytes: int, sha256: str | None = None):
        if self.sealed or key in self._items:
            raise ValidationError("resident store is sealed or expert already loaded")
        integer(expected_bytes, "expert byte length", 1, self.budget - self.used)
        # Account for both construction buffer and immutable final copy at boot.
        if self.used + 2 * expected_bytes > self.budget:
            raise ValidationError("resident boot staging peak exceeds budget")
        data = bytearray(expected_bytes)
        offset = 0
        digest = hashlib.sha256()
        for chunk in chunks:
            if not isinstance(chunk, bytes) or not 0 < len(chunk) <= 1024 * 1024:
                raise ValidationError("resident preload requires bounded nonempty byte chunks")
            if offset + len(chunk) > expected_bytes:
                raise ValidationError("resident payload exceeds declared expert length")
            data[offset:offset + len(chunk)] = chunk
            digest.update(chunk)
            offset += len(chunk)
        if offset != expected_bytes or (sha256 is not None and digest.hexdigest() != sha256):
            raise ValidationError("resident expert short read or checksum mismatch")
        self._items[key] = bytes(data)
        self.used += expected_bytes

    def seal(self, expected: set[ExpertKey]):
        if self.sealed or set(self._items) != expected:
            raise ValidationError("resident preload inventory incomplete or seal repeated")
        self.sealed = True

    def get(self, key: ExpertKey) -> memoryview:
        if not self.sealed:
            raise ValidationError("resident-only inference requires completed preload")
        if key not in self._items:
            raise ValidationError("resident expert missing; inference disk reads are forbidden")
        return memoryview(self._items[key])


@dataclass
class _Entry:
    data: bytes
    generation: int
    leases: int = 0


class RAMCache:
    """A bounded host tier with deduplicated loads and non-evictable leases.

    A loader must read exactly the declared bundle using its own bounded I/O
    buffers. Loader/staging bytes are reserved in advance by this cache. GPU/DMA
    users must keep the lease until their event completes; returning a lease
    early is an adapter bug and is not hidden by a CUDA-ready flag here.
    """
    def __init__(self, budget: int, staging_budget: int, *, max_entries: int = 4096, max_inflight: int = 8):
        self.budget = integer(budget, "RAM cache budget", 1, 1 << 40)
        self.staging_budget = integer(staging_budget, "I/O staging budget", 1, 1 << 40)
        self.used = self.staging_used = 0
        self.max_entries = integer(max_entries, "entry count bound", 1, 65536)
        self.max_inflight = integer(max_inflight, "in-flight read bound", 1, 256)
        self._entries: OrderedDict[ExpertKey, _Entry] = OrderedDict()
        self._loading: set[ExpertKey] = set()
        self._condition = threading.Condition()
        self._generation = 0
        self.reads = self.hits = self.evictions = 0

    @contextmanager
    def lease(self, key: ExpertKey, *, nbytes: int, loader: Callable[[], bytes], timeout: float = 10):
        integer(nbytes, "expert bytes", 1, min(self.budget, self.staging_budget))
        if not 0 < timeout <= 300:
            raise ValidationError("invalid cache wait timeout")
        metrics.current.add("ram_bundle_requests")
        deadline = time.monotonic() + timeout
        with self._condition:
            while key in self._loading:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError("expert in-flight read deadline exceeded")
                self._condition.wait(timeout=remaining)
            entry = self._entries.get(key)
            if entry is not None:
                if len(entry.data) != nbytes:
                    raise ValidationError("expert cache identity has conflicting byte lengths")
                self.hits += 1
                metrics.current.add("ram_bundle_hits")
                self._entries.move_to_end(key)
            else:
                if len(self._loading) >= self.max_inflight:
                    raise ValidationError("bounded I/O queue is full")
                if self.staging_used + nbytes > self.staging_budget:
                    raise ValidationError("bounded I/O staging is exhausted")
                while (self.used + nbytes > self.budget or
                       len(self._entries) + len(self._loading) >= self.max_entries):
                    victim = next((k for k, v in self._entries.items() if not v.leases), None)
                    if victim is None:
                        raise ValidationError("all cache slots are leased; backpressure required")
                    self.used -= len(self._entries.pop(victim).data)
                    self.evictions += 1
                    metrics.current.add("ram_evictions")
                # Reservation counts against cache capacity even before publication.
                self.used += nbytes
                self.staging_used += nbytes
                self._loading.add(key)
            if entry is not None:
                entry.leases += 1
        if entry is None:
            try:
                data = loader()
                if not isinstance(data, bytes) or len(data) != nbytes:
                    raise ValidationError("failed or short expert read; entry not published")
                with self._condition:
                    self._generation += 1
                    entry = _Entry(data, self._generation, 1)
                    self._entries[key] = entry
                    self.reads += 1
                    metrics.current.add("ram_bundle_loads")
            except BaseException:
                with self._condition:
                    self.used -= nbytes
                raise
            finally:
                with self._condition:
                    self.staging_used -= nbytes
                    self._loading.remove(key)
                    self._condition.notify_all()
        try:
            yield memoryview(entry.data), entry.generation
        finally:
            with self._condition:
                entry.leases -= 1
                self._condition.notify_all()
