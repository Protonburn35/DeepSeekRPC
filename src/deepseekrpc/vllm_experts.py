"""Real vLLM quantization-method adapter for the bounded owned expert tier.

Import this only in a configured worker with the actual pinned vLLM installed.
It does not register a competing graph or mutate global vLLM namespaces.
"""
from __future__ import annotations

from .jsonutil import ValidationError
from .routed_experts import BoundedRoutedExperts
from .dense_weights import DenseBinding, FixedPackedWeights

from vllm.model_executor.layers.fused_moe.fused_moe_method_base import FusedMoEMethodBase
from vllm.model_executor.layers.fused_moe.routed_experts import RoutedExperts
from vllm_exl3.exl3 import Exl3Config
from vllm.model_executor.layers.linear import LinearBase, LinearMethodBase
from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead


class BoundedMoEMethod(FusedMoEMethodBase):
    def __init__(self, moe, executor: BoundedRoutedExperts):
        super().__init__(moe)
        parallel = moe.moe_parallel_config
        if (any(getattr(parallel, key) != 1 for key in ("tp_size", "ep_size", "dp_size", "pcp_size", "sp_size")) or parallel.enable_eplb or
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
    def bind(cls, base: Exl3Config, executors: dict[str, BoundedRoutedExperts], *,
             packed_weights: FixedPackedWeights | None = None,
             dense_bindings: dict[str, DenseBinding] | None = None):
        if not isinstance(base, Exl3Config) or base.codebook != "mul1" or not executors:
            raise ValidationError("owned EXL3 configuration requires the exact mul1 plugin and expert inventory")
        instance = cls.__new__(cls)
        instance.__dict__.update(base.__dict__)
        instance._owned_executors = dict(executors)
        instance._packed_weights, instance._dense_bindings = packed_weights, dict(dense_bindings or {})
        if bool(instance._dense_bindings) != (packed_weights is not None):
            raise ValidationError("permanent packed store and module bindings must be admitted together")
        if packed_weights is not None:
            for binding in instance._dense_bindings.values():
                packed_weights.binding_geometry(binding)
        return instance

    def get_quant_method(self, layer, prefix):
        if prefix in self._dense_bindings:
            return FixedPackedMethod(self._packed_weights, self._dense_bindings[prefix])
        if isinstance(layer, RoutedExperts):
            if prefix not in self._owned_executors:
                raise ValidationError("vLLM attempted to construct unowned routed experts")
            return BoundedMoEMethod(layer.moe_config, self._owned_executors[prefix])
        if isinstance(layer, (LinearBase, ParallelLMHead)):
            raise ValidationError("owned EXL3 graph attempted an unbound permanent packed projection")
        return super().get_quant_method(layer, prefix)


class FixedPackedMethod(LinearMethodBase):
    """Real vLLM linear API over the one permanent packed arena."""
    def __init__(self, weights: FixedPackedWeights, binding: DenseBinding):
        self.weights, self.binding = weights, binding
        self._ready = False

    def create_weights(self, layer, input_size_per_partition, output_partition_sizes,
                       input_size, output_size, params_dtype, **extra_weight_attrs):
        inputs, outputs = self.weights.binding_geometry(self.binding)
        if (input_size_per_partition != inputs or input_size != inputs or
                sum(output_partition_sizes) != outputs or output_size != outputs):
            raise ValidationError("vLLM packed linear shape differs from its owned physical matrices")
        if any(True for _ in layer.parameters(recurse=False)):
            raise ValidationError("packed linear staging parameters already allocated")

    def process_weights_after_loading(self, layer):
        if self.weights.closed or any(True for _ in layer.parameters(recurse=False)):
            raise ValidationError("permanent packed linear admission/loading is incomplete")
        self._ready = True

    def apply(self, layer, x, bias=None):
        if not self._ready or bias is not None:
            raise ValidationError("permanent packed linear is not ready or has an unexpected bias")
        return self.weights.forward(self.binding, x)
