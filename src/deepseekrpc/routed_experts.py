"""Bounded routed EXL3 execution over leased, original packed expert bundles."""
from __future__ import annotations

from dataclasses import dataclass

from .jsonutil import ValidationError, integer
from .packed import PackedBundleLayout
from .storage import ExpertKey


def partition_routes(ids, *, experts: int, topk: int):
    """Retain every selected route and reject duplicate experts within a row."""
    integer(experts, "routed expert count", 1, 384)
    integer(topk, "routed top-k", 1, min(6, experts))
    if not isinstance(ids, list) or not 1 <= len(ids) <= 64:
        raise ValidationError("routed work requires between 1 and 64 rows")
    groups = {}
    for row, choices in enumerate(ids):
        if not isinstance(choices, list) or len(choices) != topk:
            raise ValidationError("router did not provide every selected expert")
        seen = set()
        for rank, expert in enumerate(choices):
            integer(expert, "selected expert", 0, experts - 1)
            if expert in seen:
                raise ValidationError("a token selected the same expert more than once")
            seen.add(expert)
            groups.setdefault(expert, []).append((row, rank))
    return tuple((expert, tuple(groups[expert])) for expert in sorted(groups))


@dataclass(frozen=True)
class ExpertCatalog:
    keys: tuple[ExpertKey, ...]
    layouts: tuple[PackedBundleLayout, ...]
    hidden: int
    intermediate: int

    @classmethod
    def from_source(cls, source, *, model: str, quantization: str, layer: int,
                    experts: int, prefix: str):
        integer(experts, "routed expert count", 1, 384)
        layouts, keys = [], []
        for expert in range(experts):
            key = ExpertKey(model, layer, expert, "bundle", quantization)
            layout = PackedBundleLayout.from_source(source, f"{prefix}.{expert}")
            keys.append(key)
            layouts.append(layout)
        geometry = {(item.matrices[0][2].in_features, item.matrices[0][2].out_features) for item in layouts}
        if len(geometry) != 1:
            raise ValidationError("routed experts disagree on layer projection geometry")
        hidden, intermediate = geometry.pop()
        return cls(tuple(keys), tuple(layouts), hidden, intermediate)

    @property
    def slot_bytes(self):
        return max(item.nbytes for item in self.layouts)


def weighted_swiglu(gate, up, weights, *, limit=10.0):
    """Official order: FP32 asymmetric clamp, SiLU, weight, then caller casts."""
    import torch
    import torch.nn.functional as functional
    if (limit != 10.0 or gate.dtype != torch.float32 or up.dtype != torch.float32 or
            gate.shape != up.shape or gate.ndim != 2 or weights.shape != (len(gate), 1) or
            gate.device != up.device or gate.device != weights.device):
        raise ValidationError("expert activation shape, precision or October clamp mismatch")
    return functional.silu(gate.clamp(max=limit)) * up.clamp(min=-limit, max=limit) * weights.float()


class BoundedRoutedExperts:
    """All selected experts, with FP32 accumulation and no dense reconstruction.

    Router weights already include normalization and the model's routing scale.
    The vLLM runner computes/adds its shared expert separately. This operation
    consumes only the routed contribution and does not alter that shared path.
    """
    def __init__(self, catalog: ExpertCatalog, cache, *, topk: int = 6):
        integer(topk, "routed top-k", 1, min(6, len(catalog.keys)))
        if catalog.slot_bytes > cache.slot_bytes:
            raise ValidationError("largest owned expert exceeds the admitted GPU slot")
        self.catalog, self.cache, self.topk = catalog, cache, topk

    def forward(self, inputs, topk_weights, topk_ids):
        import torch
        from .runtime_policy import assert_exact_exl3
        assert_exact_exl3()
        catalog = self.catalog
        if (not inputs.is_cuda or inputs.device != self.cache.device or inputs.ndim != 2 or
                not 1 <= len(inputs) <= 64 or inputs.shape[1] != catalog.hidden or
                inputs.dtype not in (torch.bfloat16, torch.float16, torch.float32) or
                topk_weights.device != inputs.device or topk_ids.device != inputs.device or
                topk_weights.shape != (len(inputs), self.topk) or topk_ids.shape != topk_weights.shape or
                topk_weights.dtype != torch.float32 or topk_ids.dtype not in (torch.int32, torch.int64)):
            raise ValidationError("routed execution requires bounded CUDA rows and exact router metadata")
        if not bool(torch.isfinite(inputs).all()) or not bool(torch.isfinite(topk_weights).all()) or bool((topk_weights < 0).any()):
            raise ValidationError("routed activations or probabilities are invalid")
        groups = partition_routes(topk_ids.tolist(), experts=len(catalog.keys), topk=self.topk)
        output = torch.zeros(inputs.shape, dtype=torch.float32, device=inputs.device)
        for expert, positions in groups:
            rows, ranks = zip(*positions)
            # Every route appears once. No expert cap or probability threshold.
            row_ids = torch.tensor(rows, dtype=torch.long, device=inputs.device)
            rank_ids = torch.tensor(ranks, dtype=torch.long, device=inputs.device)
            selected = inputs.index_select(0, row_ids).to(torch.float16).contiguous()
            weights = topk_weights[row_ids, rank_ids].unsqueeze(-1)
            if not bool(torch.isfinite(selected).all()):
                raise ValidationError("EXL3 FP16 input conversion overflowed")
            with self.cache.lease(catalog.keys[expert], catalog.layouts[expert]) as lease:
                gate = lease.linear("w1", selected)
                up = lease.linear("w3", selected)
                intermediate = weighted_swiglu(gate, up, weights).to(torch.float16).contiguous()
                if not bool(torch.isfinite(intermediate).all()):
                    raise ValidationError("EXL3 weighted SwiGLU conversion overflowed")
                down = lease.linear("w2", intermediate)
                output.index_add_(0, row_ids, down)
        if not bool(torch.isfinite(output).all()):
            raise ValidationError("EXL3 routed output is nonfinite")
        converted = output.to(inputs.dtype)
        if not bool(torch.isfinite(converted).all()):
            raise ValidationError("routed output conversion overflowed")
        return converted
