import os

import pytest

from deepseekrpc.jsonutil import ValidationError
from deepseekrpc.native_windows import LockedBuffer


@pytest.mark.skipif(os.name != "nt", reason="requires the real Win32 VirtualAlloc/VirtualLock APIs")
def test_native_locked_host_pages():
    with LockedBuffer(4096, admitted_bytes=4096) as buffer:
        buffer.write(7, b"resident")
        assert buffer.read(7, 8) == b"resident"
        with pytest.raises(ValidationError):
            buffer.write(4095, b"xx")
    with pytest.raises(ValidationError):
        buffer.read(0, 1)


@pytest.mark.skipif(os.name != "nt", reason="requires real Win32 working-set/locking APIs")
def test_large_locked_buffer_reserves_process_quota_and_releases_it():
    from deepseekrpc.native_windows import _kernel32, _working_set_limits
    dll = _kernel32()
    before = _working_set_limits(dll)
    size = 64 * 1024**2
    with LockedBuffer(size, admitted_bytes=size) as buffer:
        buffer.write(size - 8, b"resident")
        assert buffer.read(size - 8, 8) == b"resident"
        assert _working_set_limits(dll)[0] >= size
        # Closing one allocation must preserve another live allocation's quota.
        with LockedBuffer(4096, admitted_bytes=4096) as second:
            second.write(0, b"ok")
        assert buffer.read(size - 8, 8) == b"resident"
    assert _working_set_limits(dll) == before


@pytest.mark.skipif(os.name == "nt", reason="this check exercises explicit refusal on non-Windows hosts")
def test_other_platform_does_not_claim_windows_residency():
    with pytest.raises(OSError, match="unavailable"):
        LockedBuffer(4096, admitted_bytes=4096)


def test_host_admission_budget_rejected_before_os_call():
    with pytest.raises(ValidationError):
        LockedBuffer(8192, admitted_bytes=4096)
