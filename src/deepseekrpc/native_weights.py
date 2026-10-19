"""Bounded bootstrap of actual unquantized vLLM parameters from owned bytes."""
from __future__ import annotations

from dataclasses import dataclass
import re

from .dense_weights import PackedBootUploader
from .inventory import MAX_READ
from .jsonutil import ValidationError, integer


def native_source_name(target: str) -> str:
    """Translate text/draft native parameters without the stock MTP drop rule."""
    if target == "model.embed_tokens.weight":
        return "embed.weight"
    if target == "model.norm.weight":
        return "norm.weight"
    if not re.fullmatch(r"model\.(?:layers|mtp)\.[0-9]{1,2}\.[a-zA-Z0-9_.]+", target):
        raise ValidationError("native parameter needs an explicit owned text/draft or vision binding")
    result = target.removeprefix("model.")
    if result.endswith(".ffn.gate.e_score_correction_bias"):
        result = result.removesuffix("e_score_correction_bias") + "bias"
    return result


@dataclass(frozen=True)
class NativeParameter:
    source_name: str
    target_name: str
    destination: object


class NativeWeightLoader:
    """Copy into pre-admitted CUDA parameters, preserving declared dtype rules.

    The real graph determines each destination dtype. Native F16/F32 source
    values are never inferred from a default dtype. Every conversion appears
    in the returned manifest; BF16 norms and FP32 routers remain distinct.
    Fused vision projections may pass explicit disjoint contiguous views.
    B reads exclusively from its locked resident store through bounded pinned staging.
    """
    def __init__(self, source, bindings: list[NativeParameter], *, required_names,
                 admitted_bytes: int):
        import torch
        integer(admitted_bytes, "native weight bootstrap admission", 1, 1 << 40)
        required_names = frozenset(required_names)
        if (not bindings or len(bindings) > 8192 or
                any(not isinstance(binding, NativeParameter) for binding in bindings) or
                {binding.source_name for binding in bindings} != required_names or
                len(bindings) != len(required_names) or not required_names <= source.names):
            raise ValidationError("native parameter plan does not cover each required owned tensor exactly once")
        types = {"BF16": torch.bfloat16, "F16": torch.float16, "F32": torch.float32}
        self._items, spans, devices = [], [], set()
        self.parameter_bytes = 0
        for binding in bindings:
            info, target = source.info(binding.source_name), binding.destination
            if (info.dtype not in types or target.dtype not in types.values() or
                    not target.is_cuda or not target.is_contiguous() or tuple(target.shape) != info.shape):
                raise ValidationError("native weight dtype, shape or CUDA target differs from the physical plan")
            devices.add(target.device)
            size = target.numel() * target.element_size()
            begin = target.data_ptr()
            spans.append((begin, begin + size))
            self.parameter_bytes += size
            self._items.append((binding, info, types[info.dtype]))
        if not torch.cuda.is_available() or len(devices) != 1:
            raise RuntimeError("native weight loading requires one actual CUDA device")
        spans.sort()
        if any(left[1] > right[0] for left, right in zip(spans, spans[1:])):
            raise ValidationError("native parameter destinations overlap")
        from .resident_weights import LockedOwnedWeights
        host_staging = MAX_READ if isinstance(source, LockedOwnedWeights) else 3 * MAX_READ
        # One reusable raw CUDA tile; dtype conversion writes straight into
        # the final parameter without an additional converted tensor.
        self.admission_bytes = self.parameter_bytes + MAX_READ + host_staging
        if self.admission_bytes > admitted_bytes:
            raise ValidationError("native parameters and bounded boot staging exceed admission")
        self.source, self.device = source, devices.pop()
        self._loaded = False

    def load(self):
        import torch
        if self._loaded:
            raise ValidationError("native bootstrap has already completed")
        manifest = []
        with torch.no_grad(), torch.cuda.device(self.device):
            uploader = PackedBootUploader(self.source)
            scratch = torch.empty(MAX_READ, dtype=torch.uint8, device=self.device)
            for binding, info, dtype in self._items:
                target = binding.destination.flatten()
                source_size = torch.empty((), dtype=dtype, device="cpu").element_size()
                # Bound both the read tile and the destination write tile.
                elements = MAX_READ // max(source_size, target.element_size())
                for start in range(0, target.numel(), elements):
                    count = min(elements, target.numel() - start)
                    raw = scratch[:count * source_size]
                    uploader.copy_range(binding.source_name, start * source_size, raw)
                    target[start:start + count].copy_(raw.view(dtype))
                    if not bool(torch.isfinite(target[start:start + count]).all()):
                        raise ValidationError("native parameter conversion overflowed or contains nonfinite values")
                manifest.append({"source": binding.source_name, "target": binding.target_name,
                                 "source_dtype": info.dtype, "target_dtype": str(target.dtype),
                                 "source_bytes": info.nbytes,
                                 "target_bytes": target.numel() * target.element_size()})
            torch.cuda.current_stream(self.device).synchronize()
        self._loaded = True
        return manifest
