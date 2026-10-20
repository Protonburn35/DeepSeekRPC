"""Explicit fidelity settings established before importing ExLlama kernels."""
import os
from pathlib import Path
import sys
import threading

from .jsonutil import ValidationError

_mutex = threading.Lock()
_configured = False
_cuda_math_configured = False
_vllm_cache_root = None
_vllm_cache_environment = None
_EXACT_ENV = {"EXL3_INT8_GEMV": "0", "EXL3_GR_INT8": "0",
              "EXL3_GEMV": "0", "EXL3_GEMM_LEGACY_TILES": "0",
              "EXL3_GEMM_SMALLM_SHARED_TUNE": "1",
              "VLLM_DISABLE_SHARED_EXPERTS_STREAM": "1",
              "VLLM_USE_DEEP_GEMM": "0",
              "VLLM_DO_NOT_TRACK": "1", "VLLM_NO_USAGE_STATS": "1", "DO_NOT_TRACK": "1",
              "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "HF_HUB_DISABLE_TELEMETRY": "1"}


def configure_owned_vllm_cache(root=None):
    """Select private paths once; an omitted root rechecks the current owner."""
    global _vllm_cache_root, _vllm_cache_environment
    with _mutex:
        root = (Path(root).absolute() if root is not None else
                _vllm_cache_root if _vllm_cache_root is not None else Path(sys.prefix) / ".deepseekrpc-vllm")
        if _vllm_cache_root is not None:
            if root != _vllm_cache_root:
                raise ValidationError("vLLM cache ownership cannot change within a worker process")
            assert_owned_vllm_cache()
        else:
            if any(name == "vllm" or name.startswith("vllm.") for name in sys.modules):
                raise ValidationError("vLLM was imported before its private cache policy; start a fresh worker")
            paths = {"VLLM_CACHE_ROOT": root / "cache", "VLLM_CONFIG_ROOT": root / "config"}
            if root.is_symlink() or root.exists() and not root.is_dir():
                raise ValidationError("private vLLM state is redirected or is not a directory")
            for path in paths.values():
                if path.is_symlink() or path.exists() and not path.is_dir():
                    raise ValidationError("private vLLM state is redirected or is not a directory")
                path.mkdir(parents=True, exist_ok=True)
            _vllm_cache_environment = {name: str(path) for name, path in paths.items()}
            os.environ.update(_vllm_cache_environment)
            _vllm_cache_root = root
    return {"private_metadata_cache": True, "private_configuration": True}


def assert_owned_vllm_cache():
    if _vllm_cache_environment is None:
        return
    if any(os.environ.get(name) != value for name, value in _vllm_cache_environment.items()):
        raise ValidationError("private vLLM cache policy changed after configuration")
    envs = sys.modules.get("vllm.envs")
    if envs is not None and any(getattr(envs, name) != value for name, value in _vllm_cache_environment.items()):
        raise ValidationError("vLLM cached a different metadata/config directory; start a fresh worker")


def configure_exact_exl3():
    """Disable activation/int8 gate shortcuts before their cached first use.

    The pinned Python and CUDA sources disagree on the unset GEMV default.
    An explicit zero before either import is required. No native execution or
    precision qualification is claimed merely by setting these variables.
    """
    global _configured
    with _mutex:
        if _configured:
            assert_exact_exl3()
            return dict(_EXACT_ENV)
        if any(name == "exllamav3_ext" or name == "exllamav3" or name.startswith("exllamav3.")
               for name in sys.modules):
            raise ValidationError("ExLlama was imported before exact policy; start a fresh worker process")
        os.environ.update(_EXACT_ENV)
        _configured = True
        return dict(_EXACT_ENV)


def assert_exact_exl3():
    assert_owned_vllm_cache()
    if not _configured or any(os.environ.get(name) != value for name, value in _EXACT_ENV.items()):
        raise ValidationError("exact ExLlama policy is missing or changed after configuration")
    usage = sys.modules.get("vllm.usage.usage_lib")
    if usage is not None and usage.is_usage_stats_enabled():
        raise ValidationError("vLLM cached usage reporting before offline startup policy; start a fresh worker process")


def configure_exact_cuda_math():
    """Require full FP32 products/reductions for Torch's auxiliary CUDA graph."""
    configure_exact_exl3()
    import torch
    global _cuda_math_configured
    with _mutex:
        if not _cuda_math_configured:
            torch.set_float32_matmul_precision("highest")
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
            torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
            torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
            _cuda_math_configured = True
        assert_exact_cuda_math()


def assert_exact_cuda_math():
    import torch
    from .triton_policy import assert_windows_triton
    assert_windows_triton(required=False)
    assert_exact_exl3()
    vllm_envs = sys.modules.get("vllm.envs")
    if (vllm_envs is not None and (not vllm_envs.VLLM_DISABLE_SHARED_EXPERTS_STREAM or vllm_envs.VLLM_USE_DEEP_GEMM) or
            not _cuda_math_configured or torch.get_float32_matmul_precision() != "highest" or
            torch.backends.cuda.matmul.allow_tf32 or torch.backends.cudnn.allow_tf32 or
            torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction or
            torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction):
        raise ValidationError("exact auxiliary CUDA math policy is missing or changed")
