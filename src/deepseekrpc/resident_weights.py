"""Boot-only loading into one checked Win32 locked host allocation.

This closes B's disk source permanently. CUDA copies, their staging buffers,
device allocations and WDDM admission still require separate qualification.
"""
from __future__ import annotations

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

    def read_range(self, name: str, offset: int, length: int) -> bytes:
        info = self.info(name)
        integer(offset, "resident tensor offset", 0, info.nbytes)
        integer(length, "bounded resident copy", 0, min(MAX_READ, info.nbytes - offset))
        return self._buffer.read(self._layout[name][0] + offset, length)

    def register_cuda(self):
        if self.closed:
            raise ValidationError("resident weights are closed")
        self._buffer.register_cuda()

    def copy_to_cuda(self, name: str, offset: int, destination):
        info = self.info(name)
        integer(offset, "resident DMA tensor offset", 0, info.nbytes)
        integer(destination.numel(), "resident DMA bytes", 0, min(MAX_READ, info.nbytes - offset))
        self._buffer.copy_to_cuda(self._layout[name][0] + offset, destination)

    def close(self):
        if self._buffer is not None:
            self._buffer.close()
        self.closed = True

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
