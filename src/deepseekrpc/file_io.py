"""Read handles that keep admitted Windows source bytes stable until close."""
import os


def open_read(path, *, buffering=0, sequential=False):
    """Share reads, but deny Windows writers and deletion for this handle's life.

    Taking the handle before validation also refuses an existing writer. POSIX
    development readers retain ordinary file I/O and their caller's stat checks.
    No file is mapped and no process-global filesystem setting is changed.
    """
    if os.name != "nt":
        return path.open("rb", buffering=buffering)
    import ctypes
    from ctypes import wintypes
    import msvcrt
    dll = ctypes.WinDLL("kernel32", use_last_error=True)
    dll.CreateFileW.argtypes = [wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD, ctypes.c_void_p,
                               wintypes.DWORD, wintypes.DWORD, wintypes.HANDLE]
    dll.CreateFileW.restype = wintypes.HANDLE
    dll.CloseHandle.argtypes = [wintypes.HANDLE]
    dll.CloseHandle.restype = wintypes.BOOL
    # GENERIC_READ, FILE_SHARE_READ, OPEN_EXISTING; preserve snapshots' hint.
    handle = dll.CreateFileW(str(path), 0x80000000, 1, None, 3,
                             0x08000000 if sequential else 0x80, None)
    if handle == ctypes.c_void_p(-1).value:
        raise ctypes.WinError(ctypes.get_last_error())
    try:
        descriptor = msvcrt.open_osfhandle(handle, os.O_RDONLY | os.O_BINARY)
    except BaseException:
        dll.CloseHandle(handle)
        raise
    return os.fdopen(descriptor, "rb", buffering=buffering)
