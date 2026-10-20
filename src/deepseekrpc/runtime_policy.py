"""Explicit fidelity settings established before importing ExLlama kernels."""
import os
import sys
import threading

from .jsonutil import ValidationError

_mutex = threading.Lock()
_configured = False
_cuda_math_configured = False
_EXACT_ENV = {"EXL3_INT8_GEMV": "0", "EXL3_GR_INT8": "0",
              "EXL3_GEMV": "0", "EXL3_GEMM_LEGACY_TILES": "0",
              "VLLM_DISABLE_SHARED_EXPERTS_STREAM": "1"}


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
    if not _configured or any(os.environ.get(name) != value for name, value in _EXACT_ENV.items()):
        raise ValidationError("exact ExLlama policy is missing or changed after configuration")


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
    assert_exact_exl3()
    vllm_envs = sys.modules.get("vllm.envs")
    if (vllm_envs is not None and not vllm_envs.VLLM_DISABLE_SHARED_EXPERTS_STREAM or
            not _cuda_math_configured or torch.get_float32_matmul_precision() != "highest" or
            torch.backends.cuda.matmul.allow_tf32 or torch.backends.cudnn.allow_tf32 or
            torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction or
            torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction):
        raise ValidationError("exact auxiliary CUDA math policy is missing or changed")
