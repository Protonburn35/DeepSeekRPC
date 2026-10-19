# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Owned BF16 attention backend for the pinned vLLM V4.1 decoder graph.

Projection construction adapts the pinned attention.py; provenance is packaged
under _vendor/vllm_windows. Real vLLM linears, Q/KV norms, RoPE and index-query
preparation remain in use. Explicit source/shadow pools replace paged quantized
cache backends. This module requires the real native stack and CUDA device.
"""
from __future__ import annotations

from contextlib import ExitStack

import torch
from torch import nn

from vllm.distributed import get_tensor_model_parallel_world_size
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import (
    ColumnParallelLinear, MergedColumnParallelLinear, ReplicatedLinear, RowParallelLinear,
)
from vllm.model_executor.models.utils import extract_layer_index
from vllm.models.deepseek_v41.attention import DeepseekV4Attention
from vllm.models.deepseek_v41.common.ops import fused_indexer_q_rope_quant
from vllm.models.deepseek_v41.common.rope import build_deepseek_v4_rope

from .attention_state import AttentionState
from .compression_state import DeviceCompressionState
from .cuda_ops import sparse_attention
from .jsonutil import ValidationError
from .runtime_policy import assert_exact_cuda_math
from .vllm_experts import FixedPackedMethod, OwnedExl3Config


class OwnedCompressorProjection(nn.Module):
    def __init__(self, config, quant_config, ratio, prefix):
        super().__init__()
        self.fused_wkv_wgate = MergedColumnParallelLinear(
            config.hidden_size, [config.head_dim] * ratio, bias=False,
            return_bias=False, quant_config=quant_config, disable_tp=True,
            params_dtype=torch.bfloat16, prefix=f"{prefix}.fused_wkv_wgate")
        self.norm = RMSNorm(config.head_dim, config.rms_norm_eps, dtype=torch.bfloat16)


class OwnedIndexerProjection(nn.Module):
    def __init__(self, config, quant_config, owns_k, prefix):
        super().__init__()
        self.wq_b = ReplicatedLinear(
            config.q_lora_rank, config.index_n_heads * config.index_head_dim,
            bias=False, return_bias=False, quant_config=quant_config,
            params_dtype=torch.bfloat16, prefix=f"{prefix}.wq_b")
        self.weights_proj = ReplicatedLinear(
            config.hidden_size, config.index_n_heads, bias=False, return_bias=False,
            quant_config=None, params_dtype=torch.bfloat16, prefix=f"{prefix}.weights_proj")
        if owns_k:
            self.wk = ReplicatedLinear(
                config.head_dim, config.index_head_dim, bias=False, return_bias=False,
                quant_config=quant_config, params_dtype=torch.bfloat16, prefix=f"{prefix}.wk")
            self.k_norm = RMSNorm(config.index_head_dim, config.rms_norm_eps, dtype=torch.bfloat16)


class OwnedBF16Attention(DeepseekV4Attention):
    """Actual vLLM attention subclass with bounded source-aware CUDA execution."""
    use_fp8_ds_mla_layout = False

    def __init__(self, vllm_config, prefix, *, state: AttentionState,
                 topk_indices_buffer=None, aux_stream_list=None, candidate_block_buffer=None,
                 compression_admitted_bytes=512 * 1024):
        # The stock constructor registers quantized paged cache/index backends.
        # Build its learned projection/norm graph explicitly over admitted pools.
        nn.Module.__init__(self)
        assert_exact_cuda_math()
        if (not isinstance(state, AttentionState) or state.device.type != "cuda" or
                not torch.cuda.is_available() or get_tensor_model_parallel_world_size() != 1 or
                not isinstance(vllm_config.quant_config, OwnedExl3Config) or
                aux_stream_list is not None or topk_indices_buffer is not None or candidate_block_buffer is not None):
            raise ValidationError("owned BF16 attention requires its real TP=1 packed graph and explicit state")
        config = vllm_config.model_config.hf_config
        expected = (5120, 64, 512, 64, 1280, 1024, 8, 128, 32, 128, 512)
        actual = tuple(getattr(config, name) for name in (
            "hidden_size", "num_attention_heads", "head_dim", "qk_rope_head_dim",
            "q_lora_rank", "o_lora_rank", "o_groups", "sliding_window",
            "index_n_heads", "index_head_dim", "index_topk"))
        if actual != expected or config.max_position_embeddings != state.plan.geometry.context:
            raise ValidationError("owned attention is not the pinned October model geometry")
        self.layer_id = extract_layer_index(prefix)
        if self.layer_id not in state.windows:
            raise ValidationError("vLLM attempted to construct an unowned attention layer")
        self.prefix, self.state = prefix, state
        self.hidden_size, self.q_lora_rank, self.o_lora_rank = 5120, 1280, 1024
        self.n_heads = self.n_local_heads = self.padded_heads = 64
        self.head_dim, self.rope_head_dim, self.nope_head_dim = 512, 64, 448
        self.n_groups = self.n_local_groups = 8
        self.window_size, self.eps, self.scale = 128, config.rms_norm_eps, 512 ** -.5
        g = state.plan.geometry
        self.compress_ratio = g.compress_ratios[self.layer_id] if self.layer_id < g.layers else 0
        self.is_kv_source, self.is_index_source = self.layer_id in g.kv_sources, self.layer_id in g.index_sources
        self.kv_source_layer_id = max(s for s in g.kv_sources if s <= self.layer_id) if self.compress_ratio else None
        self.index_source_layer_id = max(s for s in g.index_sources if s <= self.layer_id) if self.compress_ratio else None
        self._compression_admitted_bytes = compression_admitted_bytes
        self._ready = False
        self.gemm_rs = None
        self.aux_stream_list = None
        self._prepare_and_attn_fn = self._prepare_and_attn
        quant = vllm_config.quant_config
        with torch.device(state.device):
            self.attn_sink = nn.Parameter(torch.full((64,), float("nan"), dtype=torch.float32), requires_grad=False)
            self.fused_wqa_wkv = MergedColumnParallelLinear(
                5120, [1280, 512], bias=False, quant_config=quant, disable_tp=True,
                params_dtype=torch.bfloat16, prefix=f"{prefix}.fused_wqa_wkv")
            self.q_norm = RMSNorm(1280, self.eps, dtype=torch.bfloat16)
            self.kv_norm = RMSNorm(512, self.eps, dtype=torch.bfloat16)
            self.wq_b = ColumnParallelLinear(
                1280, 64 * 512, bias=False, return_bias=False, quant_config=quant,
                params_dtype=torch.bfloat16, prefix=f"{prefix}.wq_b")
            self.wo_a = ColumnParallelLinear(
                4096, 8192, bias=False, return_bias=False, quant_config=quant,
                params_dtype=torch.bfloat16, prefix=f"{prefix}.wo_a")
            self.wo_b = RowParallelLinear(
                8192, 5120, bias=False, return_bias=False, quant_config=quant,
                params_dtype=torch.bfloat16, prefix=f"{prefix}.wo_b")
            self.rotary_emb = build_deepseek_v4_rope(
                config, head_dim=512, rope_head_dim=64,
                max_position_embeddings=config.max_position_embeddings, compress_ratio=self.compress_ratio)
            self.compressor = (OwnedCompressorProjection(config, quant, self.compress_ratio, f"{prefix}.compressor")
                               if self.is_kv_source else None)
            self.indexer = (OwnedIndexerProjection(config, quant, self.is_kv_source, f"{prefix}.indexer")
                            if self.is_index_source else None)
            self.register_buffer("_empty_source", torch.empty((0, 512), dtype=torch.bfloat16), persistent=False)
        # Every packed projection must have an explicit physical binding. A
        # delegated dense/plugin reconstruction path is never admitted here.
        packed = [self.fused_wqa_wkv, self.wq_b, self.wo_a, self.wo_b]
        if self.compressor is not None:
            packed.append(self.compressor.fused_wkv_wgate)
        if self.indexer is not None:
            packed.append(self.indexer.wq_b)
            if self.is_kv_source:
                packed.append(self.indexer.wk)
        if any(not isinstance(layer.quant_method, FixedPackedMethod) for layer in packed):
            raise ValidationError("owned attention contains an unbound packed projection")

    @classmethod
    def get_padded_num_q_heads(cls, num_heads):
        if num_heads != 64:
            raise ValidationError("owned attention requires all 64 local heads")
        return num_heads

    @property
    def _can_fuse_query_quant(self):
        return False

    def finalize_loading(self):
        if self._ready or not bool(torch.isfinite(self.attn_sink).all()):
            raise ValidationError("attention loading is duplicate or the learned sink was not loaded")
        if self.compressor is not None:
            self.state.attach_compressor(self.layer_id, DeviceCompressionState(
                self.compress_ratio, self.compressor.norm.weight, eps=self.eps,
                context=self.state.plan.geometry.context, admitted_bytes=self._compression_admitted_bytes))
        self._ready = True

    def forward(self, positions, hidden_states, llama_4_scaling=None):
        window = self.state.current
        if (not self._ready or llama_4_scaling is not None or hidden_states.device != self.state.device or
                hidden_states.dtype != torch.bfloat16 or hidden_states.shape != (window.operation.rows, 5120) or
                positions.device != self.state.device or positions.dtype != torch.int64 or
                positions.shape != (window.operation.rows,) or not positions.is_contiguous() or
                positions.tolist() != list(range(window.operation.start, window.operation.start + window.operation.rows))):
            raise ValidationError("attention input is not the current bounded BF16 layer window")
        try:
            return super().forward(positions, hidden_states)
        except BaseException:
            self.state.fail()
            raise

    def _run_parallel_input_projections(self, hidden_states):
        qr_kv = self._fused_wqa_wkv_gemm(hidden_states)
        scores = self.compressor.fused_wkv_wgate(hidden_states) if self.compressor is not None else None
        weights = self.indexer.weights_proj(hidden_states) if self.indexer is not None else None
        return qr_kv, scores, weights

    def _publish_compressed(self, scores):
        rows = self.state.compress(self.layer_id, scores)
        if not rows.logical_positions:
            return
        positions = torch.tensor([logical * self.compress_ratio for logical in rows.logical_positions],
                                 dtype=torch.int64, device=self.state.device)
        keys = rows.values.clone().unsqueeze(1)
        self.rotary_emb.forward_cuda(positions, keys)
        index = self.indexer.k_norm(self.indexer.wk(rows.values)).unsqueeze(1)
        self.rotary_emb.forward_cuda(positions, index)
        self.state.publish_source(self.layer_id, rows, keys[:, 0].contiguous(), index[:, 0].contiguous())

    def project_encoder_source(self, hidden_states):
        """Produce layer-20 global encoder state without decoder attention."""
        window = self.state.current
        if (not self._ready or self.layer_id != 20 or window.phase != "encoder" or
                hidden_states.device != self.state.device or hidden_states.dtype != torch.bfloat16 or
                hidden_states.shape != (window.operation.rows, 5120)):
            raise ValidationError("source-only projection requires the real layer-20 encoder boundary")
        try:
            self._publish_compressed(self.compressor.fused_wkv_wgate(hidden_states))
        except BaseException:
            self.state.fail()
            raise

    def _prepare_and_attn(self, hidden_states, qr, kv, qr_scale, kv_score, indexer_weights, positions, attn_out):
        if qr_scale is not None:
            raise ValidationError("EXL3 attention does not admit fused activation-quantized Q input")
        q = self.wq_b(qr).view(-1, 64, 512)
        # qr and kv already passed the real fused Q/KV RMSNorm. As in the
        # pinned plain-BF16 path, apply RoPE without another query norm.
        self.rotary_emb.forward_cuda(positions, q, kv.unsqueeze(1))
        if self.compressor is not None:
            self._publish_compressed(kv_score)
        if self.indexer is not None:
            index_q = self.indexer.wq_b(qr).view(-1, 32, 128)
            prepared, weights = fused_indexer_q_rope_quant(
                positions, index_q, self.rotary_emb.cos_sin_cache, indexer_weights,
                128 ** -.5, 32 ** -.5, use_fp4=False, weights_out_dtype=torch.float32)
            # Keep the pinned Windows query preparation: its FP8 scalar scale
            # is folded into weights. Stored index K remains BF16 throughout.
            if not isinstance(prepared, torch.Tensor) or prepared.dtype != torch.float8_e4m3fn:
                raise ValidationError("unexpected index-query precision profile")
            self.state.select(self.layer_id, prepared.float().contiguous(), weights.contiguous())
        self.forward_mqa(q, kv, positions, attn_out)

    def forward_mqa(self, q, kv, positions, output):
        window = self.state.current
        operation = window.operation
        swa = self.state.windows[self.layer_id]
        first = max(0, operation.start - 127)
        past = range(first, operation.start)
        source = self.state.sources[self.kv_source_layer_id] if self.compress_ratio else None
        with ExitStack() as stack:
            stack.enter_context(swa.read_window(operation))
            slots = torch.tensor(swa.physical(past), dtype=torch.int64, device=self.state.device)
            # Preserve the old ring tail before writing up to 64 current rows.
            staging = torch.cat((swa.data.index_select(0, slots), kv), dim=0)
            ids = torch.full((operation.rows, 128 + (512 if source is not None else 0)),
                             -1, dtype=torch.int32, device=self.state.device)
            host_ids = []
            selected = self.state.selection(self.index_source_layer_id).topk.tolist() if source is not None else None
            if source is not None:
                stack.enter_context(source.read_window(operation))
            for row, position in enumerate(range(operation.start, operation.start + operation.rows)):
                local = list(range(max(0, position - 127) - first, position - first + 1))
                local.extend([-1] * (128 - len(local)))
                if source is not None:
                    physical = source.physical(selected[row])
                    local.extend(slot + len(staging) if slot >= 0 else -1 for slot in physical)
                host_ids.append(local)
            ids.copy_(torch.tensor(host_ids, dtype=torch.int32, device=self.state.device))
            result = sparse_attention(q.float().contiguous(), staging,
                                      source.data if source is not None else self._empty_source,
                                      ids, self.attn_sink, scale=self.scale)
            output.copy_(result.to(torch.bfloat16))
        swa.write(list(range(operation.start, operation.start + operation.rows)),
                  list(range(operation.start, operation.start + operation.rows)), kv.contiguous())

    def _o_proj(self, attn_out, positions):
        self.rotary_emb.forward_cuda(positions, attn_out, inverse=True)
        grouped = attn_out.reshape(len(attn_out), 8, 4096)
        return self.wo_b(self.wo_a(grouped))
