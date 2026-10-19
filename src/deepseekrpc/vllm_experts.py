"""Real vLLM quantization-method adapter for the bounded owned expert tier.

Import this only in a configured worker with the actual pinned vLLM installed.
It does not register a competing graph or mutate global vLLM namespaces.
"""
from __future__ import annotations

from .jsonutil import ValidationError
from .routed_experts import BoundedRoutedExperts

from vllm.model_executor.layers.fused_moe.fused_moe_method_base import FusedMoEMethodBase
from vllm.model_executor.layers.fused_moe.routed_experts import RoutedExperts
from vllm_exl3.exl3 import Exl3Config


class BoundedMoEMethod(FusedMoEMethodBase):
    def __init__(self, moe, executor: BoundedRoutedExperts):
        super().__init__(moe)
        parallel = moe.moe_parallel_config
        if (parallel.tp_size != 1 or parallel.ep_size != 1 or parallel.enable_eplb or
                moe.experts_per_token != executor.topk):
            raise ValidationError("layer RPC requires complete local experts with TP=EP=1")
        self.executor, self._ready = executor, False

    def get_fused_moe_quant_config(self, layer):
        return None

    def create_weights(self, layer, num_experts, hidden_size,
                       intermediate_size_per_partition, params_dtype, **extra_weight_attrs):
        catalog = self.executor.catalog
        if ((num_experts, hidden_size, intermediate_size_per_partition) !=
                (len(catalog.keys), catalog.hidden, catalog.intermediate) or
                layer.swiglu_limit != 10.0 or layer.apply_router_weight_on_input):
            raise ValidationError("vLLM layer differs from the admitted October routed expert graph")
        # Packed host bytes and the one fixed device pool are owned by executor.
        # Creating per-layer/all-expert CUDA Parameters would defeat that bound.
        if any(True for _ in layer.parameters(recurse=False)):
            raise ValidationError("routed expert parameters were allocated before bounded admission")

    def process_weights_after_loading(self, layer):
        if any(True for _ in layer.parameters(recurse=False)):
            raise ValidationError("upstream loading allocated an unbounded routed expert tensor")
        self._ready = True

    def apply(self, layer, x, topk_weights, topk_ids, shared_experts, shared_experts_input):
        if not self._ready:
            raise ValidationError("bounded routed experts were not admitted after owned loading")
        return self.executor.forward(x, topk_weights, topk_ids)


class OwnedExl3Config(Exl3Config):
    """Delegate dense quantization to the actual plugin, bind routed layers locally."""
    @classmethod
    def bind(cls, base: Exl3Config, executors: dict[str, BoundedRoutedExperts]):
        if not isinstance(base, Exl3Config) or base.codebook != "mul1" or not executors:
            raise ValidationError("owned EXL3 configuration requires the exact mul1 plugin and expert inventory")
        instance = cls.__new__(cls)
        instance.__dict__.update(base.__dict__)
        instance._owned_executors = dict(executors)
        return instance

    def get_quant_method(self, layer, prefix):
        if isinstance(layer, RoutedExperts):
            if prefix not in self._owned_executors:
                raise ValidationError("vLLM attempted to construct unowned routed experts")
            return BoundedMoEMethod(layer.moe_config, self._owned_executors[prefix])
        return super().get_quant_method(layer, prefix)
