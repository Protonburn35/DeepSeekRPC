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

_working_set_mutex = threading.Lock()
_working_set_users = {}
_working_set_baseline = None
_working_set_applied = None


class _ProcessMemoryCounters(ctypes.Structure):
    _fields_ = [("cb", wintypes.DWORD), ("faults", wintypes.DWORD),
                ("peak_working_set", ctypes.c_size_t), ("working_set", ctypes.c_size_t),
                ("peak_paged_pool", ctypes.c_size_t), ("paged_pool", ctypes.c_size_t),
                ("peak_nonpaged_pool", ctypes.c_size_t), ("nonpaged_pool", ctypes.c_size_t),
                ("pagefile", ctypes.c_size_t), ("peak_pagefile", ctypes.c_size_t)]


def _working_set_limits(dll):
    process = dll.GetCurrentProcess()
    minimum, maximum, flags = ctypes.c_size_t(), ctypes.c_size_t(), wintypes.DWORD()
    if not dll.GetProcessWorkingSetSizeEx(process, ctypes.byref(minimum), ctypes.byref(maximum), ctypes.byref(flags)):
        raise ctypes.WinError(ctypes.get_last_error())
    return minimum.value, maximum.value, flags.value


def _reserve_working_set(dll, size):
    global _working_set_baseline, _working_set_applied
    with _working_set_mutex:
        current = _working_set_limits(dll)
        if _working_set_baseline is None or current != _working_set_applied:
            _working_set_baseline = current
        counters = _ProcessMemoryCounters()
        counters.cb = ctypes.sizeof(counters)
        if not dll.K32GetProcessMemoryInfo(dll.GetCurrentProcess(), ctypes.byref(counters), counters.cb):
            raise ctypes.WinError(ctypes.get_last_error())
        minimum = max(current[0], sum(_working_set_users.values()) + size + 4 * 1024**2)
        # Preserve the existing resident runtime as well as new locked pages.
        maximum = max(current[1], minimum + 4 * 1024**2, counters.working_set + size + 4 * 1024**2)
        if not dll.SetProcessWorkingSetSizeEx(dll.GetCurrentProcess(), minimum, maximum, current[2]):
            raise ctypes.WinError(ctypes.get_last_error())
        _working_set_applied = _working_set_limits(dll)
        token = object()
        _working_set_users[token] = size
        return token


def _release_working_set(dll, token):
    global _working_set_baseline, _working_set_applied
    with _working_set_mutex:
        _working_set_users.pop(token, None)
        if not _working_set_users and _working_set_baseline is not None:
            if _working_set_limits(dll) == _working_set_applied:
                if not dll.SetProcessWorkingSetSizeEx(dll.GetCurrentProcess(), *_working_set_baseline):
                    raise ctypes.WinError(ctypes.get_last_error())
            # Preserve an independent caller's changes to the process limits.
            _working_set_baseline = _working_set_applied = None


def _kernel32():
    if os.name != "nt":
        raise OSError("native Windows memory admission is unavailable on this platform")
    dll = ctypes.WinDLL("kernel32", use_last_error=True)
    dll.VirtualAlloc.argtypes = [ctypes.c_void_p, ctypes.c_size_t, wintypes.DWORD, wintypes.DWORD]
    dll.VirtualAlloc.restype = ctypes.c_void_p
    dll.VirtualFree.argtypes = [ctypes.c_void_p, ctypes.c_size_t, wintypes.DWORD]
    dll.VirtualFree.restype = wintypes.BOOL
    dll.GetCurrentProcess.argtypes = []
    dll.GetCurrentProcess.restype = wintypes.HANDLE
    dll.GetProcessWorkingSetSizeEx.argtypes = [wintypes.HANDLE, ctypes.POINTER(ctypes.c_size_t),
                                              ctypes.POINTER(ctypes.c_size_t), ctypes.POINTER(wintypes.DWORD)]
    dll.GetProcessWorkingSetSizeEx.restype = wintypes.BOOL
    dll.SetProcessWorkingSetSizeEx.argtypes = [wintypes.HANDLE, ctypes.c_size_t, ctypes.c_size_t, wintypes.DWORD]
    dll.SetProcessWorkingSetSizeEx.restype = wintypes.BOOL
    dll.K32GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(_ProcessMemoryCounters), wintypes.DWORD]
    dll.K32GetProcessMemoryInfo.restype = wintypes.BOOL
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
        self._working_set_token = _reserve_working_set(self._dll, self.size)
        self._pointer = None
        try:
            self._pointer = self._dll.VirtualAlloc(None, self.size, 0x3000, 0x04)
            if not self._pointer or not self._dll.VirtualLock(self._pointer, self.size):
                raise ctypes.WinError(ctypes.get_last_error())
        except BaseException:
            if self._pointer:
                self._dll.VirtualFree(self._pointer, 0, 0x8000)
                self._pointer = None
            _release_working_set(self._dll, self._working_set_token)
            self._working_set_token = None
            raise
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
            if self._working_set_token is not None:
                _release_working_set(self._dll, self._working_set_token)
                self._working_set_token = None

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
