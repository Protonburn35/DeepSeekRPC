"""Checked native CUDA operations. There is no CPU inference fallback."""
from __future__ import annotations

from functools import lru_cache
import math

from .jsonutil import ValidationError
from .runtime_policy import assert_exact_exl3, configure_exact_exl3


@lru_cache(maxsize=1)
def extension():
    configure_exact_exl3()
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("a real CUDA device is required; CPU inference fallback is not supported")
    import deepseekrpc_cuda
    if deepseekrpc_cuda.abi_version() != 2:
        raise RuntimeError("DeepSeekRPC native extension ABI does not match this source")
    return deepseekrpc_cuda


def _result(result):
    value, status = result
    _status(status)
    return value


def _status(status):
    # Correctness baseline: complete the operation before publishing its state.
    # Later overlap must preserve the same event/transaction dependency.
    code = int(status.item())
    if code:
        raise ValidationError(f"native CUDA operation rejected input/state (code {code}); tentative writes require rollback")


def sparse_attention(query, window_kv, source_kv, indices, sink, *, scale: float):
    if not math.isfinite(scale) or scale <= 0:
        raise ValidationError("invalid attention scale")
    operation = extension()
    assert_exact_exl3()
    return _result(operation.sparse_attention(query, window_kv, source_kv, indices, sink, scale))


def index_scores(query, keys, weights, indices, visible):
    operation = extension()
    assert_exact_exl3()
    return _result(operation.index_scores(query, keys, weights, indices, visible))


def decode_engram(encoded):
    return _result(extension().engram_decode(encoded))


def scatter_slots(pool, generations, slots, expected_generations, values):
    _status(extension().scatter_slots(pool, generations, slots, expected_generations, values))
