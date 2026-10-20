"""Boot-only loading into one checked Win32 locked host allocation.

This closes B's disk source permanently. CUDA copies, their staging buffers,
device allocations and WDDM admission still require separate qualification.
"""
from __future__ import annotations

import threading
from . import metrics

from .inventory import MAX_READ
from .jsonutil import ValidationError, integer
from .native_windows import LockedBuffer
from .planner import Role
from .weights import SelectiveWeights


class LockedOwnedWeights:
    def __init__(self, source: SelectiveWeights, *, admitted_bytes: int):
        if source.role != Role.MIDDLE or source.closed:
            raise ValidationError("resident preload requires B's open owned boot source")
        integer(admitted_bytes, "locked resident admission", 1, 1 << 40)
        self._layout = {}
        self._buffer = None
        self._transfer = None
        self._transfer_mutex = threading.Lock()
        self.closed = True
        try:
            offset = 0
            for name in sorted(source.names):
                info = source.info(name)
                self._layout[name] = (offset, info)
                offset += info.nbytes
                if offset > admitted_bytes:
                    raise ValidationError("owned resident tensors exceed admitted host bytes")
            self._buffer = LockedBuffer(offset, admitted_bytes=admitted_bytes)
            for name, (base, info) in self._layout.items():
                written = 0
                for chunk in source.chunks(name):
                    self._buffer.write(base + written, chunk)
                    written += len(chunk)
                if written != info.nbytes:
                    raise ValidationError("resident tensor preload was incomplete")
            self.residency = self._buffer.verify_locked_pages()
            self.payload_bytes, self.allocation_bytes = offset, self._buffer.size
            self.names = frozenset(self._layout)
            self.closed = False
        except BaseException:
            if self._buffer is not None:
                self._buffer.close()
            raise
        finally:
            source.close()

    def info(self, name: str):
        if self.closed or name not in self._layout:
            raise ValidationError("resident weights are closed or tensor is not owned")
        return self._layout[name][1]

    def revalidate_residency(self):
        if self.closed:
            raise ValidationError("resident weights are closed")
        self.residency = self._buffer.verify_locked_pages()
        return self.residency

    def read_range(self, name: str, offset: int, length: int) -> bytes:
        info = self.info(name)
        integer(offset, "resident tensor offset", 0, info.nbytes)
        integer(length, "bounded resident copy", 0, min(MAX_READ, info.nbytes - offset))
        return self._buffer.read(self._layout[name][0] + offset, length)

    def copy_to_pinned(self, name: str, offset: int, destination):
        """Direct locked-to-pinned copy, never a whole-store CUDA registration."""
        import ctypes
        import torch
        info = self.info(name)
        if (destination.device.type != "cpu" or destination.dtype != torch.uint8 or
                destination.ndim != 1 or not destination.is_contiguous() or not destination.is_pinned()):
            raise ValidationError("resident transfer requires actual CUDA-pinned CPU bytes")
        integer(offset, "resident transfer tensor offset", 0, info.nbytes)
        count = integer(destination.numel(), "resident transfer bytes", 0, min(MAX_READ, info.nbytes - offset))
        if count:
            # The destination Tensor remains owned by this call. Only its
            # bounded writable address is borrowed; no source pointer escapes.
            owner = (ctypes.c_ubyte * count).from_address(destination.data_ptr())
            self._buffer.copy_into(self._layout[name][0] + offset, owner)

    def transfer(self):
        """One shared pinned tile for boot/native/expert copies, never all weights."""
        with self._transfer_mutex:
            if self.closed:
                raise ValidationError("resident weights are closed")
            if self._transfer is None:
                self._transfer = _ResidentTransfer(self)
            return self._transfer

    def close(self):
        with self._transfer_mutex:
            if self._buffer is not None:
                self._buffer.close()
            self.closed = True
            self._transfer = None

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


class _ResidentTransfer:
    def __init__(self, source):
        import torch
        self.source = source
        self._mutex = threading.Lock()
        self._pinned = torch.empty(MAX_READ, dtype=torch.uint8, device="cpu", pin_memory=True)

    def copy(self, name, offset, destination):
        import torch
        count = destination.numel()
        if (destination.device.type != "cuda" or destination.dtype != torch.uint8 or
                destination.ndim != 1 or not destination.is_contiguous() or not 1 <= count <= MAX_READ):
            raise ValidationError("resident upload requires one bounded CUDA byte range")
        with self._mutex:
            self.source.copy_to_pinned(name, offset, self._pinned[:count])
            with metrics.current.measure("pcie_upload"):
                destination.copy_(self._pinned[:count], non_blocking=False)
            metrics.current.add("pcie_upload_bytes", count)
