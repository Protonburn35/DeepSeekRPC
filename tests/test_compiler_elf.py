"""CUDA ELF layout fixtures; these do not execute a compiler or GPU."""
import struct

import pytest

from deepseekrpc.compiler_selftest import cubin_architecture
from deepseekrpc.jsonutil import ValidationError


def _elf(header, flags):
    result = bytearray(64)
    result[:16] = bytes.fromhex(header)
    struct.pack_into("<H", result, 18, 190)
    struct.pack_into("<I", result, 48, flags)
    return result


def test_blackwell_abi8_uses_the_actual_captured_shifted_architecture_field():
    # Exact header/flags observed from pinned ptxas in Windows run 39410453853.
    binary = _elf("7f454c46020101410800000000000000", 100694026)
    assert binary[8] == 8
    assert cubin_architecture(binary) == 120
    assert 100694026 & 255 == 10  # The previous low-byte inspector was wrong.


def test_pre_blackwell_abi7_retains_the_low_byte_architecture_field():
    # Protocol fixture from the documented ABI 7 mask, not a captured cubin.
    assert cubin_architecture(_elf("7f454c46020101330700000000000000", 0x590559)) == 89


@pytest.mark.parametrize("offset,value", [(0, 0), (4, 1), (5, 2), (8, 9), (18, 0)])
def test_unknown_or_non_cuda_elf_layouts_are_rejected(offset, value):
    binary = _elf("7f454c46020101410800000000000000", 100694026)
    binary[offset] = value
    with pytest.raises(ValidationError):
        cubin_architecture(binary)


def test_truncated_header_cannot_claim_an_architecture():
    with pytest.raises(ValidationError):
        cubin_architecture(b"\x7fELF")
