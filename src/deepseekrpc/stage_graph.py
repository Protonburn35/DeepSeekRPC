"""Explicit owned stages over the actual patched vLLM decoder layers.

This module requires completed memory admission, pinned CUDA components and
real local TP=PP=1 groups. The separate coordinator owns begin/finish, complete
CED passes, replica transfer, sampling and all-stage commit.
"""
from __future__ import annotations

from dataclasses import dataclass
import inspect
from pathlib import Path
import threading

import torch
from torch import nn

from .attention_state import AttentionState, AttentionWindow
from .jsonutil import ValidationError
from .native_weights import NativeParameter, NativeWeightLoader, native_source_name
from .planner import Role
from .runtime_policy import assert_exact_cuda_math
from .state import Operation


def initialize_local_groups(vllm_config, rendezvous: Path):
    """Real loopback-local Gloo groups; TCP stage RPC remains separate."""
    from vllm.config import set_current_vllm_config
    from vllm.distributed.parallel_state import (
        ensure_model_parallel_initialized, init_distributed_environment, set_custom_all_reduce,
    )
    if torch.distributed.is_initialized() or rendezvous.exists():
        raise ValidationError("local stage groups require a fresh worker and unused rendezvous file")
    if not torch.distributed.is_gloo_available():
        raise RuntimeError("the Windows Torch build must provide the actual Gloo backend")
    set_custom_all_reduce(False)
    with set_current_vllm_config(vllm_config):
        # The pinned vLLM code strips file:// and passes the native path to
        # FileStore directly. A Windows file URI's leading slash is unsuitable.
        init_distributed_environment(world_size=1, rank=0, local_rank=0, backend="gloo",
                                     distributed_init_method="file://" + str(rendezvous.resolve()))
        ensure_model_parallel_initialized(1, 1, backend="gloo")


@dataclass(frozen=True)
class StageBoundary:
    window: AttentionWindow
    hidden: object
    pre_mix: object

    def internal(self, expected: AttentionWindow, device):
        if (self.window != expected or self.hidden.device != device or self.pre_mix.device != device or
                self.hidden.dtype != torch.float32 or self.pre_mix.dtype != torch.float32 or
                self.hidden.shape != (expected.operation.rows, 4, 5120) or
                self.pre_mix.shape != (expected.operation.rows, 4) or
                not self.hidden.is_contiguous() or not self.pre_mix.is_contiguous() or
                not bool(torch.isfinite(self.hidden).all()) or not bool(torch.isfinite(self.pre_mix).all())):
            raise ValidationError("stage input is not the current FP32 activation/pre_mix boundary")
        hidden = self.hidden.to(torch.bfloat16)
        # The pinned graph's internal residual is BF16. FP32 transport promotes
        # it exactly; receiving it must recover the original BF16 bits.
        if not torch.equal(hidden.float(), self.hidden):
            raise ValidationError("stage wire hidden values do not preserve the pinned internal BF16 streams")
        return hidden, self.pre_mix


@dataclass(frozen=True)
class StageResult:
    operation: Operation
    boundary: StageBoundary | None = None
    logits: object | None = None
    target_features: object | None = None
    encoder_source_complete: bool = False


@dataclass(frozen=True)
class DraftResult:
    operation: Operation
    base_logits: object
    head_hidden: object


class _Body(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleDict()
        self.mtp = nn.ModuleDict()


class OwnedStageGraph(nn.Module):
    """Every owned target, vision and draft parameter has an explicit binding."""
    def __init__(self, vllm_config, state: AttentionState, *, row_stores=None, include_draft=True):
        super().__init__()
        assert_exact_cuda_math()
        from vllm.config import set_current_vllm_config
        from vllm.model_executor.layers.layernorm import RMSNorm
        from vllm.model_executor.layers.linear import ReplicatedLinear
        from vllm.model_executor.layers.logits_processor import LogitsProcessor
        from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead, VocabParallelEmbedding
        from vllm.models.deepseek_v41.common.engram import EngramLayout
        from vllm.models.deepseek_v41.nvidia.model import DeepseekV4DecoderLayer
        from .vllm_attention import OwnedBF16Attention, OwnedDSparkAttention
        from .vllm_engram import DiskEngram
        from .vllm_experts import FixedPackedMethod, OwnedExl3Config

        parallel, scheduler = vllm_config.parallel_config, vllm_config.scheduler_config
        if (any(getattr(parallel, key) != 1 for key in (
                "tensor_parallel_size", "pipeline_parallel_size", "data_parallel_size",
                "prefill_context_parallel_size", "decode_context_parallel_size")) or
                parallel.enable_expert_parallel or parallel.use_sequence_parallel_moe or parallel.enable_dbo or
                not vllm_config.model_config.enforce_eager or vllm_config.model_config.dtype != torch.bfloat16 or
                vllm_config.kernel_config.enable_jit_warmup or vllm_config.cache_config.swa_bounded_replay or
                scheduler.max_num_seqs != 1 or scheduler.max_num_batched_tokens != 64 or
                not isinstance(vllm_config.quant_config, OwnedExl3Config)):
            raise ValidationError("stage graph requires its explicit single-session TP=PP=1 eager BF16 configuration")
        if "rpc_attention_factory" not in inspect.signature(DeepseekV4DecoderLayer).parameters:
            raise ValidationError("the real pinned decoder stage-factory patch is missing")
        self.config, self.state, self.role = vllm_config, state, state.role
        self.hf_config = vllm_config.model_config.hf_config
        self.begin_layer, self.end_layer = state.plan.bounds(self.role)
        self._ready, self._mutex = False, threading.RLock()
        self.model = _Body()
        self.include_draft = bool(include_draft and self.role == Role.TERMINAL)
        if self.include_draft and any(state.plan.owner(layer) != Role.TERMINAL for layer in (37, 38, 39)):
            raise ValidationError("this DSpark profile requires target layers 37/38/39 on stage C")
        row_stores = dict(row_stores or {})
        expected_engram = set(state.plan.geometry.engram_layers) if self.role == Role.HOST else set()
        if set(row_stores) != expected_engram:
            raise ValidationError("stage Engram stores differ from the two owned verified original tables")
        layout = EngramLayout.from_config(self.hf_config) if self.role == Role.HOST else None

        def attention_factory(config, prefix, **kwargs):
            cls = OwnedDSparkAttention if int(prefix.split(".")[2]) >= 40 else OwnedBF16Attention
            return cls(config, prefix, state=state, **kwargs)

        def engram_factory(config, quant, engram_layout, layer_index, use_sequence_parallel, prefix, **kwargs):
            if kwargs.get("prefetch_stream") is not None:
                raise ValidationError("owned Engram uses its explicit bounded synchronous stream")
            layer = engram_layout.layer_ids[layer_index]
            return DiskEngram(config, quant, engram_layout, layer_index, use_sequence_parallel,
                              prefix, role=self.role, row_store=row_stores[layer], device=state.device)

        previous_dtype = torch.get_default_dtype()
        try:
            torch.set_default_dtype(torch.bfloat16)
            with torch.device(state.device), set_current_vllm_config(vllm_config):
                for layer in range(self.begin_layer, self.end_layer):
                    self.model.layers[str(layer)] = DeepseekV4DecoderLayer(
                        vllm_config, prefix=f"model.layers.{layer}", engram_layout=layout,
                        rpc_attention_factory=attention_factory, rpc_engram_factory=engram_factory)
                if self.role == Role.HOST or self.include_draft:
                    self.model.embed_tokens = VocabParallelEmbedding(
                        self.hf_config.vocab_size, 5120, params_dtype=torch.bfloat16,
                        prefix="model.embed_tokens")
                if self.role == Role.HOST:
                    from vllm.models.deepseek_v4.common.vision import DeepseekV4Aligner, DeepseekV4ViT
                    self.vision, self.aligner = DeepseekV4ViT(self.hf_config), DeepseekV4Aligner(self.hf_config)
                    self.vision.to(dtype=torch.bfloat16)
                    self.aligner.to(dtype=torch.bfloat16)
                    for name in ("image_start", "image_end", "image_newline"):
                        setattr(self, name, nn.Parameter(torch.empty(5120, dtype=torch.float32), requires_grad=False))
                if self.role == Role.TERMINAL:
                    self.model.norm = RMSNorm(5120, self.hf_config.rms_norm_eps, dtype=torch.bfloat16)
                    self.lm_head = ParallelLMHead(self.hf_config.vocab_size, 5120,
                                                 quant_config=vllm_config.quant_config, prefix="lm_head")
                    if not isinstance(self.lm_head.quant_method, FixedPackedMethod):
                        raise ValidationError("the real packed vocabulary head is not bound")
                    self.logits_processor = LogitsProcessor(self.hf_config.vocab_size)
                if self.include_draft:
                    from vllm.model_executor.models.qwen3_dspark import DSparkConfidenceHead, DSparkMarkovHead
                    for draft in range(3):
                        self.model.mtp[str(draft)] = DeepseekV4DecoderLayer(
                            vllm_config, prefix=f"model.layers.{40 + draft}", rpc_attention_factory=attention_factory)
                    first, last = self.model.mtp["0"], self.model.mtp["2"]
                    first.main_proj = ReplicatedLinear(15360, 5120, bias=False, return_bias=False,
                                                       quant_config=vllm_config.quant_config, prefix="model.mtp.0.main_proj")
                    first.main_norm = RMSNorm(5120, self.hf_config.rms_norm_eps, dtype=torch.bfloat16)
                    last.norm = RMSNorm(5120, self.hf_config.rms_norm_eps, dtype=torch.bfloat16)
                    last.markov_head = DSparkMarkovHead(self.hf_config.vocab_size, self.hf_config.vocab_size,
                                                       256, prefix="model.mtp.2.markov_head")
                    last.confidence_head = DSparkConfidenceHead(5376, prefix="model.mtp.2.confidence_head")
                    self.markov_logits_processor = LogitsProcessor(self.hf_config.vocab_size)
                    self.markov_logits_processor.head_dtype = torch.float32
        finally:
            torch.set_default_dtype(previous_dtype)
        self.requires_grad_(False)
        self.eval()

    def load_owned_native(self, source, *, admitted_bytes: int):
        if self._ready:
            raise ValidationError("stage graph weights already loaded")
        bindings = []
        for name, parameter in self.named_parameters():
            if name.startswith(("vision.", "aligner.")) or name in {"image_start", "image_end", "image_newline"}:
                source_name = name
            else:
                source_name = native_source_name(name)
                source_name = source_name.replace(".markov_head.markov_w1.", ".markov_head.embed.")
                source_name = source_name.replace(".markov_head.markov_w2.", ".markov_head.head.")
            bindings.append(NativeParameter(source_name, name, parameter))
        required = {name for name in source.names if not name.endswith((".trellis", ".suh", ".svh", ".mul1"))}
        loader = NativeWeightLoader(source, bindings, required_names=required, admitted_bytes=admitted_bytes)
        manifest = loader.load()
        from vllm.config import set_current_vllm_config
        with torch.device(self.state.device), set_current_vllm_config(self.config), torch.inference_mode():
            for module in self.modules():
                method = getattr(module, "quant_method", None)
                if method is not None:
                    method.process_weights_after_loading(module)
            for layer in [*self.model.layers.values(), *self.model.mtp.values()]:
                layer.attn.finalize_loading()
            if self.role == Role.HOST:
                first = self.model.layers["0"]
                first.hc_attn_fn_broadcast = first.hc_attn_fn.detach().view(24, 4, 5120).sum(1)
        self._ready = True
        return manifest

    @torch.inference_mode()
    def run_image(self, patches, types, *, vit_h, vit_w, cancelled):
        """Actual owned tower/aligner; bounded fused SDPA and per-block cancel."""
        from vllm.config import set_current_vllm_config
        from vllm.forward_context import set_forward_context
        from vllm.models.deepseek_v4.common.vision import get_vision_cos_sin
        from vllm.models.deepseek_v41.common.vl_cudagraph import DeepseekV4VLEncoderCudaGraphMixin
        from torch.nn.attention import SDPBackend, sdpa_kernel
        from ._vendor.deepseek_1017.image_processor import image_token_types
        from .jsonutil import integer
        integer(vit_h, "vision grid height", 1, 4096)
        integer(vit_w, "vision grid width", 1, 4096)
        expected_types = image_token_types((vit_h + 2) // 3, (vit_w + 2) // 3)
        with self._mutex:
            if (self.role != Role.HOST or not self._ready or not callable(cancelled) or
                    vit_h * vit_w > 9216 or expected_types.numel() > 1024 or
                    patches.device != self.state.device or patches.dtype != torch.bfloat16 or
                    patches.shape != (vit_h * vit_w, 3, 14, 14) or not patches.is_contiguous() or
                    types.device != self.state.device or types.dtype != torch.int64 or
                    types.shape != expected_types.shape or not torch.equal(types.cpu(), expected_types)):
                raise ValidationError("vision inputs differ from the admitted original BF16 patch/span geometry")
            with set_current_vllm_config(self.config), set_forward_context({}, self.config, num_tokens=patches.shape[0], skip_compiled=True), \
                    sdpa_kernel([SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION]):
                # The unchanged ViT operations, with cancellation boundaries
                # between actual blocks. An unsupported fused SDPA fails here;
                # no quadratic math-attention fallback is enabled.
                with torch.device("cpu"):
                    cos, sin = get_vision_cos_sin(vit_h, vit_w, self.vision.rope_dim, self.vision.rope_theta)
                cos, sin = cos.to(patches.device), sin.to(patches.device)
                hidden = self.vision.patch_embed(patches)
                for block in self.vision.blocks:
                    if cancelled():
                        raise InterruptedError("vision cancelled between actual tower blocks")
                    hidden = block(hidden, cos, sin)
                hidden = self.vision.norm(hidden)
                if cancelled():
                    raise InterruptedError("vision cancelled before original aligner")
                embeds = self.aligner(hidden, vit_h, vit_w)
                span = DeepseekV4VLEncoderCudaGraphMixin._build_image_span(self, embeds, types)
                if span.dtype != torch.bfloat16 or span.shape != (types.numel(), 5120) or not bool(torch.isfinite(span).all()):
                    raise ValidationError("owned vision graph did not produce its exact finite BF16 image span")
                return span.contiguous()

    @torch.inference_mode()
    def run_target(self, token_ids, *, incoming: StageBoundary | None = None,
                   engram_hashes=None, text_mask=None, image_rows=None):
        from vllm.forward_context import set_forward_context
        from vllm.model_executor.kernels.mhc.triton import hc_collapse_triton
        from .vllm_mhc import mhc_post, mhc_pre_delayed
        with self._mutex:
            window = self.state.current
            operation = window.operation
            if (not self._ready or window.phase not in {"encoder", "target", "decode"} or
                    token_ids.device != self.state.device or token_ids.dtype != torch.int64 or
                    token_ids.shape != (operation.rows,) or not token_ids.is_contiguous() or
                    bool((token_ids < 0).any()) or bool((token_ids >= self.hf_config.vocab_size).any())):
                raise ValidationError("stage target execution requires its real current bounded token window")
            positions = torch.arange(operation.start, operation.start + operation.rows, dtype=torch.int64,
                                     device=self.state.device)
            if self.role == Role.HOST:
                if incoming is not None or engram_hashes is None or engram_hashes.device != self.state.device or \
                        engram_hashes.dtype != torch.int64 or engram_hashes.shape != (operation.rows, 2, 24):
                    raise ValidationError("stage A requires original current Engram hashes and no preceding stage")
                hidden, pre_mix = self.model.embed_tokens(token_ids), None
                if text_mask is not None and (text_mask.device != self.state.device or text_mask.dtype != torch.bool or
                                              text_mask.shape != (operation.rows,)):
                    raise ValidationError("text/image positions do not match the current stage window")
                has_image = text_mask is not None and not bool(text_mask.all())
                if has_image != (image_rows is not None):
                    raise ValidationError("every current image-span row requires its original precomputed vision embedding")
                if image_rows is not None:
                    if (text_mask is None or image_rows.device != self.state.device or
                            image_rows.dtype != torch.bfloat16 or image_rows.shape != hidden.shape or
                            not bool(torch.isfinite(image_rows).all())):
                        raise ValidationError("image embedding rows differ from the admitted original image span")
                    hidden = torch.where(text_mask[:, None], hidden, image_rows)
                for layer in self.model.layers.values():
                    if layer.engram is not None:
                        layer.engram.prepare_embeddings(engram_hashes[:, layer.engram.layer_hash_index].contiguous())
            else:
                if incoming is None or engram_hashes is not None or image_rows is not None:
                    raise ValidationError("worker stage requires the exact preceding FP32 activation boundary")
                hidden, pre_mix = incoming.internal(window, self.state.device)
            residual = post_mix = res_mix = None
            features = {}
            try:
                with set_forward_context({}, self.config, num_tokens=operation.rows, skip_compiled=True):
                    for layer_id in range(self.begin_layer, min(self.end_layer, 20) if window.phase == "encoder" else self.end_layer):
                        if self.state.cancelled():
                            raise InterruptedError("target execution cancelled at a layer boundary")
                        layer = self.model.layers[str(layer_id)]
                        hidden, residual, post_mix, res_mix, pre_mix, auxiliary = layer(
                            hidden, positions, token_ids, pre_mix, post_mix, res_mix, residual,
                            engram_hashes, text_mask, capture_previous_aux=layer_id in (37, 38, 39))
                        if auxiliary is not None:
                            features[layer_id] = auxiliary
                    if residual is not None:
                        hidden = mhc_post(hidden, residual, post_mix, res_mix)
                    elif not (window.phase == "encoder" and self.begin_layer <= 20 < self.end_layer):
                        raise ValidationError("stage phase contains no owned target layer execution")
                    if window.phase == "encoder" and self.begin_layer <= 20 < self.end_layer:
                        source = self.model.layers["20"]
                        _, _, normalized, _ = mhc_pre_delayed(
                            hidden, source.hc_attn_fn, source.hc_attn_scale, source.hc_attn_base,
                            source.rms_norm_eps, source.hc_eps, source.hc_eps, source.hc_post_alpha,
                            source.hc_sinkhorn_iters, pre_mix=pre_mix, norm_weight=source.attn_norm.weight,
                            norm_eps=source.attn_norm.variance_epsilon)
                        source.attn.project_encoder_source(normalized)
                        return StageResult(operation, encoder_source_complete=True)
                    if self.role != Role.TERMINAL:
                        return StageResult(operation, boundary=StageBoundary(window, hidden.float().contiguous(), pre_mix.contiguous()))
                    if set(features) != {37, 38, 39}:
                        raise ValidationError("target verification did not capture all three DSpark attention inputs")
                    target_features = torch.cat([features[index] for index in (37, 38, 39)], dim=-1)
                    if self.include_draft:
                        first = self.model.mtp["0"]
                        main_x = first.main_norm(first.main_proj(target_features))
                        for layer in self.model.mtp.values():
                            layer.attn.seed_context(main_x, positions)
                    logits = None
                    head_rows = window.target_head_rows()
                    if head_rows is not None:
                        begin, end = head_rows
                        collapsed = hc_collapse_triton(hidden[begin:end], pre_mix[begin:end])
                        logits = self.logits_processor(self.lm_head, self.model.norm(collapsed))
                        if logits is None or logits.dtype != torch.float32 or not bool(torch.isfinite(logits).all()):
                            raise ValidationError("the real target vocabulary projection did not produce finite FP32 logits")
                    return StageResult(operation, logits=logits, target_features=target_features)
            except BaseException:
                self.state.fail()
                raise

    @torch.inference_mode()
    def run_draft(self, first_token: int):
        """Evaluate the complete bidirectional draft graph; sampling is separate."""
        from vllm.forward_context import set_forward_context
        from vllm.model_executor.kernels.mhc.triton import hc_collapse_triton
        from .jsonutil import integer
        from .vllm_mhc import mhc_post
        with self._mutex:
            window = self.state.current
            operation = window.operation
            integer(first_token, "DSpark seed token", 0, self.hf_config.vocab_size - 1)
            if (not self._ready or not self.include_draft or window.phase != "draft" or
                    operation.rows != 5 or operation.start < 1):
                raise ValidationError("draft execution requires stage C's current five-row proposal window")
            ids = torch.full((5,), self.hf_config.dspark_noise_token_id, dtype=torch.int64, device=self.state.device)
            ids[0] = first_token
            positions = torch.arange(operation.start, operation.start + 5, dtype=torch.int64, device=self.state.device)
            hidden = self.model.embed_tokens(ids).unsqueeze(1).repeat(1, 4, 1)
            pre_mix = residual = post_mix = res_mix = None
            try:
                with set_forward_context({}, self.config, num_tokens=5, skip_compiled=True):
                    for layer in self.model.mtp.values():
                        if self.state.cancelled():
                            raise InterruptedError("DSpark execution cancelled at a layer boundary")
                        hidden, residual, post_mix, res_mix, pre_mix, _ = layer(
                            hidden, positions, ids, pre_mix, post_mix, res_mix, residual)
                    hidden = mhc_post(hidden, residual, post_mix, res_mix)
                    head_hidden = hc_collapse_triton(hidden, pre_mix)
                    logits = self.logits_processor(self.lm_head, self.model.mtp["2"].norm(head_hidden))
                    if (logits is None or logits.dtype != torch.float32 or
                            logits.shape != (5, self.hf_config.vocab_size) or not bool(torch.isfinite(logits).all())):
                        raise ValidationError("DSpark vocabulary projection did not produce finite complete FP32 logits")
                    return DraftResult(operation, logits, head_hidden)
            except BaseException:
                self.state.fail()
                raise

    @torch.inference_mode()
    def draft_conditional_logits(self, result: DraftResult, row: int, previous_token: int):
        """Original sequential Markov bias plus diagnostic confidence, no pruning."""
        from .jsonutil import integer
        with self._mutex:
            window = self.state.current
            integer(row, "DSpark proposal row", 0, 4)
            integer(previous_token, "DSpark previous token", 0, self.hf_config.vocab_size - 1)
            if (not self.include_draft or window.phase != "draft" or result.operation != window.operation or
                    result.base_logits.device != self.state.device or result.head_hidden.device != self.state.device or
                    result.base_logits.dtype != torch.float32 or result.head_hidden.dtype != torch.bfloat16 or
                    result.base_logits.shape != (5, self.hf_config.vocab_size) or result.head_hidden.shape != (5, 5120)):
                raise ValidationError("conditional draft logits refer to a stale or incompatible proposal block")
            last = self.model.mtp["2"]
            token = torch.tensor([previous_token], dtype=torch.int64, device=self.state.device)
            markov = last.markov_head.embed(token)
            bias = last.markov_head.bias(markov, self.markov_logits_processor)
            logits = result.base_logits[row:row + 1] + bias
            confidence = torch.sigmoid(last.confidence_head(result.head_hidden[row:row + 1], markov))
            if logits.dtype != torch.float32 or not bool(torch.isfinite(logits).all()) or not bool(torch.isfinite(confidence).all()):
                raise ValidationError("DSpark conditional logits/confidence are nonfinite or have the wrong dtype")
            return logits, confidence
