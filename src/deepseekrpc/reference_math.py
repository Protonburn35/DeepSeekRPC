"""Portable CPU operation baselines for diagnostics and independent fixtures.

These operations are not a complete DeepSeek graph or an EXL3 inference backend.
GPU execution paths must be compared against the frozen reference fixtures;
passing these CPU checks cannot establish CUDA or full-model correctness.
"""
import numpy as np

from .jsonutil import ValidationError, integer


def route(x, weight, bias, *, image_mask=None, image_bias=None, topk=6, scale=1.5):
    x, weight, bias = [np.asarray(v, dtype=np.float32) for v in (x, weight, bias)]
    if x.ndim != 2 or weight.ndim != 2 or x.shape[1] != weight.shape[1] or bias.shape != (weight.shape[0],):
        raise ValidationError("router input/weight/bias shape mismatch")
    integer(topk, "router top-k", 1, weight.shape[0])
    scores = np.sqrt(np.logaddexp(np.float32(0), x @ weight.T))
    correction = bias
    if image_mask is not None:
        image_mask = np.asarray(image_mask)
        if image_mask.dtype != np.bool_ or image_mask.shape != (len(x),) or image_bias is None:
            raise ValidationError("image routing requires a boolean mask and image bias")
        image_bias = np.asarray(image_bias, dtype=np.float32)
        if image_bias.shape != bias.shape:
            raise ValidationError("image router bias shape mismatch")
        correction = np.where(image_mask[:, None], image_bias, bias)
    ranked = scores + correction
    # Equal-score expert tie behavior must be qualified against the actual GPU
    # router. This CPU reference chooses lower IDs and makes no CUDA tie claim.
    indices = np.argsort(-ranked, axis=-1, kind="stable")[:, :topk]
    selected = np.take_along_axis(scores, indices, axis=1)
    if topk > 1:
        selected = selected / (selected.sum(axis=-1, keepdims=True) + np.float32(1e-20))
    return (selected * np.float32(scale)).astype(np.float32), indices.astype(np.int64)


def expert(x, w1, w2, w3, *, routing_weight=None, limit=10.0):
    x = np.asarray(x, dtype=np.float32)
    w1, w2, w3 = [np.asarray(v, dtype=np.float32) for v in (w1, w2, w3)]
    if (x.ndim != 2 or w1.ndim != 2 or w3.shape != w1.shape or
            w2.shape != (w1.shape[1], w1.shape[0]) or x.shape[1] != w1.shape[1]):
        raise ValidationError("expert projection geometry mismatch")
    gate, up = x @ w1.T, x @ w3.T
    if limit > 0:
        gate, up = np.minimum(gate, np.float32(limit)), np.clip(up, -limit, limit)
    intermediate = gate * np.exp(-np.logaddexp(np.float32(0), -gate)) * up
    if routing_weight is not None:
        routing_weight = np.asarray(routing_weight, dtype=np.float32)
        if routing_weight.shape != (len(x), 1):
            raise ValidationError("expert routing weights must be [rows,1]")
        intermediate *= routing_weight
    return intermediate @ w2.T


def hc_pre(x, pre_mix):
    x, pre_mix = np.asarray(x, dtype=np.float32), np.asarray(pre_mix, dtype=np.float32)
    if x.ndim != 4 or pre_mix.shape != x.shape[:-1]:
        raise ValidationError("mHC pre-mix shape mismatch")
    return (x * pre_mix[..., None]).sum(axis=2, dtype=np.float32)


def hc_post(x, residual, post, comb):
    x, residual, post, comb = [np.asarray(v, dtype=np.float32) for v in (x, residual, post, comb)]
    if (residual.ndim != 4 or x.shape != residual.shape[:2] + residual.shape[3:] or
            post.shape != residual.shape[:-1] or comb.shape != residual.shape[:2] + (residual.shape[2],) * 2):
        raise ValidationError("mHC post-mix shape mismatch")
    # The source's first comb stream axis is the reduction (input) stream.
    mixed = np.sum(comb[..., None] * residual[..., None, :], axis=2, dtype=np.float32)
    return post[..., None] * x[..., None, :] + mixed


def mul1_codebook(windows):
    """EXL3 mul1 procedural codebook, returning exact binary16 table values.

    Formula is audited against the pinned ExLlama/plugin reference and tested
    for all 65536 16-bit windows. This does not decode trellis layouts/rotations.
    """
    windows = np.asarray(windows)
    if windows.dtype != np.uint16:
        raise ValidationError("mul1 windows must use unsigned 16-bit storage")
    product = (windows.astype(np.uint64) * np.uint64(2212286765)) & np.uint64(0xFFFFFFFF)
    total = np.full(windows.shape, 0x6400, dtype=np.uint16)
    for shift in (0, 8, 16, 24):
        total += ((product >> shift) & 255).astype(np.uint16)
    a = np.array(0x1EEE, dtype=np.uint16).view(np.float16).astype(np.float64)
    b = np.array(0xC931, dtype=np.uint16).view(np.float16).astype(np.float64)
    return (total.view(np.float16).astype(np.float64) * a + b).astype(np.float16)
