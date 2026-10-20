# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Explicit Windows CUDA mHC backend for the real delayed vLLM layer graph.

PyTorch dispatches the products and reductions to its native CUDA kernels. The
public layer entry points require CUDA, BF16 residuals and the pinned geometry;
the arithmetic helpers are separately compared on CPU as operation diagnostics.
Adapted from the pinned vLLM mhc/torch.py; see _vendor/vllm_windows/PROVENANCE.json.
"""
from __future__ import annotations

import math

import torch

from .jsonutil import ValidationError
from .runtime_policy import assert_exact_cuda_math


def _coefficients(residual, fn, scale, base, rms_eps, pre_eps, sinkhorn_eps, post_multiplier, iterations, x=None):
    hc = residual.shape[1]
    flattened = (residual.flatten(1) if x is None else x).float()
    mixes = (flattened @ fn.T) * torch.rsqrt(flattened.square().mean(-1, keepdim=True) + rms_eps)
    pre = torch.sigmoid(mixes[:, :hc] * scale[0] + base[:hc]) + pre_eps
    post = torch.sigmoid(mixes[:, hc:2 * hc] * scale[1] + base[hc:2 * hc]) * post_multiplier
    comb = mixes[:, 2 * hc:].view(-1, hc, hc) * scale[2] + base[2 * hc:].view(1, hc, hc)
    comb = torch.softmax(comb, dim=-1) + sinkhorn_eps
    comb = comb / (comb.sum(dim=-2, keepdim=True) + sinkhorn_eps)
    for _ in range(iterations - 1):
        comb = comb / (comb.sum(dim=-1, keepdim=True) + sinkhorn_eps)
        comb = comb / (comb.sum(dim=-2, keepdim=True) + sinkhorn_eps)
    return post.unsqueeze(-1), comb, pre


def _collapse(residual, pre_mix):
    if pre_mix is None:
        return residual[:, 0].contiguous()
    return (pre_mix.unsqueeze(-1) * residual.float()).sum(1).to(residual.dtype)


def _post(x, residual, post_layer_mix, comb_res_mix):
    # The owned MoE runner returns the FP32 sum of routed and shared experts.
    # The reference rounds the complete FFN output before its HC expansion.
    x = x.to(residual.dtype)
    mixed = torch.einsum("rij,rid->rjd", comb_res_mix.float(), residual.float())
    return (mixed + post_layer_mix.float() * x.unsqueeze(1).float()).to(residual.dtype)


def _check_residual(residual):
    assert_exact_cuda_math()
    if (not residual.is_cuda or residual.dtype != torch.bfloat16 or residual.ndim != 3 or
            not 1 <= len(residual) <= 64 or residual.shape[1:] != (4, 5120) or not residual.is_contiguous()):
        raise ValidationError("Windows mHC requires bounded contiguous BF16 CUDA streams [rows,4,5120]")


def mhc_pre_delayed(residual, fn, hc_scale, hc_base, rms_eps, hc_pre_eps, hc_sinkhorn_eps,
                    hc_post_mult_value, sinkhorn_repeat, pre_mix=None, x=None,
                    norm_weight=None, norm_eps=1e-6):
    _check_residual(residual)
    width = 4 * 5120 if x is None else 5120
    if (fn.device != residual.device or fn.dtype != torch.float32 or fn.shape != (24, width) or
            hc_scale.device != residual.device or hc_scale.dtype != torch.float32 or hc_scale.shape != (3,) or
            hc_base.device != residual.device or hc_base.dtype != torch.float32 or hc_base.shape != (24,) or
            x is not None and (x.device != residual.device or x.dtype != torch.bfloat16 or x.shape != (len(residual), 5120)) or
            pre_mix is not None and (pre_mix.device != residual.device or pre_mix.dtype != torch.float32 or pre_mix.shape != (len(residual), 4)) or
            type(sinkhorn_repeat) is not int or not 1 <= sinkhorn_repeat <= 64 or
            any(not math.isfinite(value) or value <= 0 for value in (rms_eps, hc_pre_eps, hc_sinkhorn_eps, norm_eps)) or
            hc_post_mult_value != 2.0):
        raise ValidationError("Windows mHC coefficient inputs differ from the admitted graph")
    post, comb, pre = _coefficients(residual, fn, hc_scale, hc_base, rms_eps, hc_pre_eps,
                                    hc_sinkhorn_eps, hc_post_mult_value, sinkhorn_repeat, x)
    layer_input = _collapse(residual, pre_mix)
    if norm_weight is not None:
        if (norm_weight.device != residual.device or norm_weight.shape != (5120,) or
                norm_weight.dtype not in (torch.bfloat16, torch.float32)):
            raise ValidationError("Windows mHC RMSNorm parameter mismatch")
        value = layer_input.float()
        layer_input = (value * torch.rsqrt(value.square().mean(-1, keepdim=True) + norm_eps) * norm_weight.float()).to(residual.dtype)
    if any(not bool(torch.isfinite(tensor).all()) for tensor in (post, comb, pre, layer_input)):
        raise ValidationError("Windows mHC produced nonfinite coefficients or layer input")
    return post, comb, layer_input, pre


def mhc_post(x, residual, post_layer_mix, comb_res_mix):
    _check_residual(residual)
    if (x.device != residual.device or x.dtype not in (torch.bfloat16, torch.float32) or x.shape != (len(residual), 5120) or
            post_layer_mix.device != residual.device or post_layer_mix.dtype != torch.float32 or post_layer_mix.shape != (len(residual), 4, 1) or
            comb_res_mix.device != residual.device or comb_res_mix.dtype != torch.float32 or comb_res_mix.shape != (len(residual), 4, 4)):
        raise ValidationError("Windows mHC post-mix input mismatch")
    result = _post(x, residual, post_layer_mix, comb_res_mix)
    if not bool(torch.isfinite(result).all()):
        raise ValidationError("Windows mHC post-mix output is nonfinite")
    return result


def mhc_shifted_post_pre(x, residual, post_layer_mix, comb_res_mix, fn, hc_scale, hc_base,
                         rms_eps, hc_pre_eps, hc_sinkhorn_eps, hc_post_mult_value, sinkhorn_repeat,
                         pre_mix=None, norm_weight=None, norm_eps=1e-6, capture_aux=False, *,
                         stream=None, reduce_results=False):
    if stream is not None or reduce_results or not isinstance(x, torch.Tensor):
        raise ValidationError("Windows layer-RPC mHC requires its explicit TP=1 eager backend")
    residual = mhc_post(x, residual, post_layer_mix, comb_res_mix)
    aux = residual.mean(1) if capture_aux else residual.new_empty((0, 5120))
    outputs = mhc_pre_delayed(residual, fn, hc_scale, hc_base, rms_eps, hc_pre_eps, hc_sinkhorn_eps,
                              hc_post_mult_value, sinkhorn_repeat, pre_mix=pre_mix,
                              norm_weight=norm_weight, norm_eps=norm_eps)
    return residual, *outputs, aux
