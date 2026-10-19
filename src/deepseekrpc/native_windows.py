"""Checked Win32 host locking and sparse-file primitives.

This is not a CUDA/WDDM residency guarantee. Production laptop admission must
also qualify its device allocations and every other model-dependent buffer.
"""
from __future__ import annotations

import ctypes
from ctypes import wintypes
import os
import mmap
import threading

from .inventory import MAX_READ
from .jsonutil import ValidationError, integer


def _kernel32():
    if os.name != "nt":
        raise OSError("native Windows memory admission is unavailable on this platform")
    dll = ctypes.WinDLL("kernel32", use_last_error=True)
    dll.VirtualAlloc.argtypes = [ctypes.c_void_p, ctypes.c_size_t, wintypes.DWORD, wintypes.DWORD]
    dll.VirtualAlloc.restype = ctypes.c_void_p
    dll.VirtualFree.argtypes = [ctypes.c_void_p, ctypes.c_size_t, wintypes.DWORD]
    dll.VirtualFree.restype = wintypes.BOOL
    for name in ("VirtualLock", "VirtualUnlock"):
        function = getattr(dll, name)
        function.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
        function.restype = wintypes.BOOL
    return dll


class LockedBuffer:
    """Private committed and locked host pages, with no pageable fallback.

    Read/write use bounded copies. No memoryview or raw address is exposed that
    could outlive the lock. Kernel32 failure means admission fails explicitly.
    """
    def __init__(self, size: int, *, admitted_bytes: int):
        integer(admitted_bytes, "admitted host bytes", 1, 1 << 40)
        requested = integer(size, "locked allocation bytes", 1, admitted_bytes)
        self.size = ((requested + mmap.PAGESIZE - 1) // mmap.PAGESIZE) * mmap.PAGESIZE
        if self.size > admitted_bytes:
            raise ValidationError("page-rounded locked allocation exceeds admitted host budget")
        self._dll = _kernel32()
        self._mutex = threading.Lock()
        self._pointer = self._dll.VirtualAlloc(None, self.size, 0x3000, 0x04)
        if not self._pointer:
            raise ctypes.WinError(ctypes.get_last_error())
        if not self._dll.VirtualLock(self._pointer, self.size):
            error = ctypes.get_last_error()
            self._dll.VirtualFree(self._pointer, 0, 0x8000)
            self._pointer = None
            raise ctypes.WinError(error)
        self._locked = True
        self._cuda_runtime = None

    def register_cuda(self):
        """Register existing locked pages; never allocate a second host mirror."""
        import torch
        with self._mutex:
            self._bounds(0, 0)
            if self._cuda_runtime is not None:
                return
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA host registration requires a real CUDA device")
            runtime = torch.cuda.cudart()
            result = runtime.cudaHostRegister(self._pointer, self.size, 0)
            if result.value != 0:
                raise RuntimeError(f"CUDA host registration refused admission: {result.value}")
            self._cuda_runtime = runtime

    def copy_to_cuda(self, offset: int, destination):
        """Synchronous bounded DMA; no raw pointer or borrowed view escapes."""
        import torch
        with self._mutex:
            if (not destination.is_cuda or destination.dtype != torch.uint8 or
                    destination.ndim != 1 or not destination.is_contiguous()):
                raise ValidationError("locked transfer destination must be contiguous CUDA bytes")
            length = destination.numel()
            self._bounds(offset, length)
            if self._cuda_runtime is None:
                raise ValidationError("locked host pages require CUDA registration before DMA")
            if not length:
                return
            with torch.cuda.device(destination.device):
                owner = (ctypes.c_ubyte * length).from_address(self._pointer + offset)
                source = torch.frombuffer(owner, dtype=torch.uint8, count=length)
                if not source.is_pinned():
                    raise ValidationError("CUDA did not recognize registered locked host pages")
                destination.copy_(source, non_blocking=False)

    def _bounds(self, offset, length):
        if not self._pointer or not self._locked:
            raise ValidationError("locked allocation is closed")
        integer(offset, "host byte offset", 0, self.size)
        integer(length, "bounded host copy", 0, min(MAX_READ, self.size - offset))

    def read(self, offset: int, length: int) -> bytes:
        with self._mutex:
            self._bounds(offset, length)
            return ctypes.string_at(self._pointer + offset, length)

    def write(self, offset: int, data: bytes):
        if not isinstance(data, bytes):
            raise ValidationError("locked host write requires bounded bytes")
        with self._mutex:
            self._bounds(offset, len(data))
            ctypes.memmove(self._pointer + offset, data, len(data))

    def close(self):
        with self._mutex:
            if self._pointer:
                if self._cuda_runtime is not None:
                    result = self._cuda_runtime.cudaHostUnregister(self._pointer)
                    if result.value != 0:
                        raise RuntimeError(f"CUDA host unregister failed: {result.value}")
                    self._cuda_runtime = None
                if self._locked:
                    if not self._dll.VirtualUnlock(self._pointer, self.size):
                        raise ctypes.WinError(ctypes.get_last_error())
                    self._locked = False
                if not self._dll.VirtualFree(self._pointer, 0, 0x8000):
                    raise ctypes.WinError(ctypes.get_last_error())
                self._pointer = None

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


def mark_sparse(file):
    """Enable sparse allocation before creating large-offset synthetic fixtures."""
    if os.name != "nt":
        return
    import msvcrt
    dll = _kernel32()
    dll.DeviceIoControl.argtypes = [wintypes.HANDLE, wintypes.DWORD, ctypes.c_void_p,
                                   wintypes.DWORD, ctypes.c_void_p, wintypes.DWORD,
                                   ctypes.POINTER(wintypes.DWORD), ctypes.c_void_p]
    dll.DeviceIoControl.restype = wintypes.BOOL
    returned = wintypes.DWORD()
    if not dll.DeviceIoControl(msvcrt.get_osfhandle(file.fileno()), 0x900C4,
                               None, 0, None, 0, ctypes.byref(returned), None):
        raise ctypes.WinError(ctypes.get_last_error())
