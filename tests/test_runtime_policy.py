import json
import os
from pathlib import Path
import subprocess
import sys

import pytest


def test_compiler_child_has_no_inherited_sdk_selector_but_retains_unrelated_controls():
    from deepseekrpc.triton_policy import compiler_qualification_environment
    original = {"SystemRoot": str(Path('/windows')), "PATH": "ambient developer tools", "CC": "external compiler",
        "INCLUDE": "sdk headers", "LIB": "sdk libraries", "VCINSTALLDIR": "msvc", "WindowsSDKVersion": "external",
        "TRITON_CACHE_MANAGER": "external.module", "TRITON_INTERPRET": "1", "CUDA_PATH_V13_2": "external cuda",
        "PTXAS_OPTIONS": "external flags", "PYTHONPATH": "ambient modules", "unrelated": "preserved"}
    actual = compiler_qualification_environment(original)
    assert all(name not in actual for name in original if name not in {"SystemRoot", "PATH", "unrelated"})
    assert actual["unrelated"] == "preserved" and actual["CUDA_VISIBLE_DEVICES"] == "-1"
    assert actual["PATH"] == os.pathsep.join([str(Path('/windows/System32')), str(Path('/windows'))])
    assert original["TRITON_INTERPRET"] == "1"  # The parent is not modified.


def test_exact_policy_overrides_int8_defaults_before_import_and_detects_changes():
    code = '''
import json, os
from deepseekrpc.runtime_policy import configure_exact_exl3, assert_exact_exl3
from deepseekrpc.jsonutil import ValidationError
assert configure_exact_exl3() == configure_exact_exl3()
assert_exact_exl3()
print(json.dumps({k:os.environ[k] for k in ['EXL3_INT8_GEMV','EXL3_GR_INT8','EXL3_GEMV','EXL3_GEMM_LEGACY_TILES']}))
os.environ['EXL3_INT8_GEMV']='2'
try:
    assert_exact_exl3()
except ValidationError:
    pass
else:
    raise AssertionError('changed fidelity policy was accepted')
'''
    env = dict(os.environ, EXL3_INT8_GEMV="2", EXL3_GR_INT8="1", EXL3_GEMV="2", EXL3_GEMM_LEGACY_TILES="1")
    env["PYTHONPATH"] = str(Path(__file__).parents[1] / "src")
    result = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True, timeout=10)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout) == {"EXL3_INT8_GEMV": "0", "EXL3_GR_INT8": "0", "EXL3_GEMV": "0", "EXL3_GEMM_LEGACY_TILES": "0"}


@pytest.mark.reference
def test_auxiliary_cuda_precision_policy_refuses_a_later_tf32_shortcut():
    pytest.importorskip("torch")
    code = '''
from deepseekrpc.runtime_policy import configure_exact_cuda_math, assert_exact_cuda_math
from deepseekrpc.jsonutil import ValidationError
import torch
configure_exact_cuda_math()
assert_exact_cuda_math()
torch.backends.cuda.matmul.allow_tf32 = True
try:
    assert_exact_cuda_math()
except ValidationError:
    pass
else:
    raise AssertionError("changed auxiliary CUDA math policy was accepted")
'''
    env = dict(os.environ)
    env["PYTHONPATH"] = str(Path(__file__).parents[1] / "src")
    # This is a precision-policy assertion, not a cold Torch startup speed
    # benchmark. Shared cloud/Windows runners can stall during native import.
    result = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stderr
