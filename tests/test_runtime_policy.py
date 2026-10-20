import json
import os
from pathlib import Path
import subprocess
import sys

import pytest


def test_private_metadata_cache_is_owned_and_cannot_be_redirected_after_start(tmp_path):
    code = '''
from pathlib import Path
import os, sys
from deepseekrpc.runtime_policy import configure_owned_vllm_cache, assert_owned_vllm_cache
from deepseekrpc.jsonutil import ValidationError
root = Path(sys.argv[1]) / 'worker cache \u0394'
configure_owned_vllm_cache(root)
configure_owned_vllm_cache(root)
assert_owned_vllm_cache()
assert Path(os.environ['VLLM_CACHE_ROOT']) == root / 'cache'
assert Path(os.environ['VLLM_CONFIG_ROOT']) == root / 'config'
assert all((root / name).is_dir() for name in ('cache','config'))
try:
    configure_owned_vllm_cache(root.parent / 'other worker')
except ValidationError:
    pass
else:
    raise AssertionError('a live worker switched to another cache')
os.environ['VLLM_CACHE_ROOT'] = str(root.parent / 'external cache')
try:
    assert_owned_vllm_cache()
except ValidationError:
    pass
else:
    raise AssertionError('a redirected metadata cache was accepted')
'''
    env = dict(os.environ, VLLM_CACHE_ROOT="inherited shared cache", VLLM_CONFIG_ROOT="inherited shared config",
               PYTHONPATH=str(Path(__file__).parents[1] / "src"))
    result = subprocess.run([sys.executable, "-c", code, str(tmp_path)], env=env, capture_output=True, text=True, timeout=10)
    assert result.returncode == 0, result.stderr
    assert not (tmp_path / 'external cache').exists()


def test_private_metadata_cache_refuses_an_existing_file(tmp_path):
    path = tmp_path / 'not a directory'
    path.write_bytes(b'original')
    from deepseekrpc.runtime_policy import configure_owned_vllm_cache
    from deepseekrpc.jsonutil import ValidationError
    with pytest.raises(ValidationError, match='not a directory'):
        configure_owned_vllm_cache(path)
    assert path.read_bytes() == b'original'


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
names = ['EXL3_INT8_GEMV','EXL3_GR_INT8','EXL3_GEMV','EXL3_GEMM_LEGACY_TILES','EXL3_GEMM_SMALLM_SHARED_TUNE','VLLM_USE_DEEP_GEMM']
print(json.dumps({k:os.environ[k] for k in names}))
offline = ['VLLM_DO_NOT_TRACK','VLLM_NO_USAGE_STATS','DO_NOT_TRACK','HF_HUB_OFFLINE','TRANSFORMERS_OFFLINE','HF_HUB_DISABLE_TELEMETRY']
assert all(os.environ[name] == '1' for name in offline)
for name in names + offline:
    previous = os.environ[name]
    os.environ[name] = '0' if previous == '1' else '2'
    try:
        assert_exact_exl3()
    except ValidationError:
        pass
    else:
        raise AssertionError('changed fidelity policy was accepted')
    os.environ[name] = previous
'''
    env = dict(os.environ, EXL3_INT8_GEMV="2", EXL3_GR_INT8="1", EXL3_GEMV="2", EXL3_GEMM_LEGACY_TILES="1",
               EXL3_GEMM_SMALLM_SHARED_TUNE="0", VLLM_USE_DEEP_GEMM="1", VLLM_DO_NOT_TRACK="0", VLLM_NO_USAGE_STATS="0",
               DO_NOT_TRACK="0", HF_HUB_OFFLINE="0", TRANSFORMERS_OFFLINE="0", HF_HUB_DISABLE_TELEMETRY="0")
    env["PYTHONPATH"] = str(Path(__file__).parents[1] / "src")
    result = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True, timeout=10)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout) == {"EXL3_INT8_GEMV": "0", "EXL3_GR_INT8": "0", "EXL3_GEMV": "0",
                                       "EXL3_GEMM_LEGACY_TILES": "0", "EXL3_GEMM_SMALLM_SHARED_TUNE": "1", "VLLM_USE_DEEP_GEMM": "0"}


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
