"""Independent CPU equations for native-operation fixtures and later self-tests.

These implement the official October sparse-attention/sink and index-score
equations. They do not execute the model or provide a CPU inference fallback.
"""
import math

import numpy as np

from .jsonutil import ValidationError


def sparse_attention_reference(query, window_kv, source_kv, indices, sink, *, scale):
    if (query.ndim != 3 or query.shape[0] > 64 or query.shape[2] > 512 or
            window_kv.ndim != 2 or source_kv.ndim != 2 or len(source_kv) > 8192 or
            query.shape[2] != window_kv.shape[1] or query.shape[2] != source_kv.shape[1] or
            indices.shape[0] != query.shape[0] or sink.shape != (query.shape[1],) or
            not math.isfinite(scale) or scale <= 0):
        raise ValidationError("invalid bounded attention reference fixture")
    kv = np.concatenate([window_kv, source_kv]).astype(np.float32)
    if indices.dtype != np.int32 or np.any(indices < -1) or np.any(indices >= len(kv)):
        raise ValidationError("invalid sparse reference slot")
    output = np.zeros_like(query, dtype=np.float32)
    for row in range(len(query)):
        selected = kv[indices[row][indices[row] >= 0]]
        for head in range(query.shape[1]):
            logits = selected @ query[row, head].astype(np.float32) * np.float32(scale)
            maximum = max(float(sink[head]), float(np.max(logits, initial=-np.inf)))
            exp = np.exp(logits - maximum)
            output[row, head] = (exp @ selected) / (exp.sum() + np.exp(np.float32(sink[head] - maximum)))
    return output


def index_scores_reference(query, keys, weights, indices, visible):
    if (query.ndim != 3 or query.shape[0] > 64 or query.shape[1] > 32 or query.shape[2] != 128 or
            keys.ndim != 2 or keys.shape[1] != 128 or len(keys) > 8192 or
            weights.shape != query.shape[:2] or indices.ndim != 2 or indices.shape[0] != len(query) or
            visible.shape != (len(query),) or np.any(visible < 0) or np.any(visible > len(keys))):
        raise ValidationError("invalid bounded index reference fixture")
    if indices.dtype != np.int32 or np.any(indices < -1) or np.any(indices >= len(keys)):
        raise ValidationError("invalid index reference slot")
    out = np.full(indices.shape, -np.inf, dtype=np.float32)
    for row in range(len(query)):
        valid = (indices[row] >= 0) & (indices[row] < visible[row])
        dot = query[row].astype(np.float32) @ keys[indices[row, valid]].astype(np.float32).T
        out[row, valid] = (np.maximum(dot, 0) * weights[row, :, None]).sum(axis=0)
    return out
