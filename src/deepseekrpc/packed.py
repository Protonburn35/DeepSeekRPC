"""Pinned mul1 tensor layouts and real ExLlama packed CUDA bindings."""
from __future__ import annotations

from dataclasses import dataclass
import struct

from .inventory import TensorInfo
from .jsonutil import ValidationError
from .runtime_policy import assert_exact_exl3, configure_exact_exl3

MUL1_MARKER = -2082680531
PACK_K = frozenset({2, 3, 4, 5, 6, 8})


@dataclass(frozen=True)
class PackedPart:
    info: TensorInfo
    offset: int


@dataclass(frozen=True)
class PackedLinearLayout:
    prefix: str
    in_features: int
    out_features: int
    k: int
    parts: tuple[PackedPart, ...]
    nbytes: int

    @classmethod
    def from_source(cls, source, prefix: str):
        suffixes = ("trellis", "suh", "svh", "mul1")
        if any(f"{prefix}.{suffix}" not in source.names for suffix in suffixes):
            raise ValidationError("packed linear is incomplete or not owned")
        if f"{prefix}.mcg" in source.names:
            raise ValidationError("selected October pack requires the mul1 codebook exclusively")
        infos = [source.info(f"{prefix}.{suffix}") for suffix in suffixes]
        trellis, suh, svh, marker = infos
        if (trellis.dtype != "I16" or len(trellis.shape) != 3 or
                suh.dtype != "F16" or len(suh.shape) != 1 or svh.dtype != "F16" or len(svh.shape) != 1 or
                marker.dtype != "I32" or marker.shape != () or marker.nbytes != 4):
            raise ValidationError("packed linear physical types/shapes differ from pinned format")
        inputs, outputs = suh.shape[0], svh.shape[0]
        k, remainder = divmod(trellis.shape[-1], 16)
        if (inputs <= 0 or outputs <= 0 or inputs % 128 or outputs % 128 or remainder or k not in PACK_K or
                trellis.shape[:2] != (inputs // 16, outputs // 16)):
            raise ValidationError("packed linear dimensions, Hadamard padding or physical K mismatch")
        if struct.unpack("<i", source.read_range(marker.name, 0, 4))[0] != MUL1_MARKER:
            raise ValidationError("packed linear mul1 multiplier marker mismatch")
        parts, offset = [], 0
        for info in infos:
            offset = (offset + 15) // 16 * 16
            parts.append(PackedPart(info, offset))
            offset += info.nbytes
        return cls(prefix, inputs, outputs, k, tuple(parts), (offset + 255) // 256 * 256)


class DevicePackedLinear:
    """A real BC_LinearEXL3 binding over an admitted live CUDA allocation.

    The owner must retain that allocation/lease until every queued operation
    completes. Construction and forward both fail if the native stack is absent.
    """
    def __init__(self, layout: PackedLinearLayout, storage):
        configure_exact_exl3()
        import torch
        if (not storage.is_cuda or storage.dtype != torch.uint8 or storage.ndim != 1 or
                not storage.is_contiguous() or storage.numel() < layout.nbytes):
            raise ValidationError("packed linear needs admitted contiguous CUDA byte storage")
        from vllm_exl3.exl3 import make_linear_exl3
        views = {}
        for part in layout.parts:
            dtype = {"F16": torch.float16, "I16": torch.int16, "I32": torch.int32}[part.info.dtype]
            views[part.info.name.rsplit(".", 1)[1]] = storage.narrow(0, part.offset, part.info.nbytes).view(dtype).view(part.info.shape)
        with torch.cuda.device(storage.device):
            self.inner = make_linear_exl3(**views, out_dtype=torch.float32)
            # Avoid persistent dense reconstruction at every row count.
            self.inner.config.infer_params.no_reconstruct = True
        self.layout, self.storage = layout, storage
        self.device, self.closed = storage.device, False

    def forward(self, inputs):
        import torch
        assert_exact_exl3()
        if self.closed:
            raise ValidationError("packed linear lease has ended")
        if (inputs.device != self.device or inputs.dtype != torch.float16 or not inputs.is_contiguous() or
                inputs.ndim != 2 or not 1 <= inputs.shape[0] <= 64 or inputs.shape[1] != self.layout.in_features):
            raise ValidationError("packed linear input must be a bounded contiguous FP16 CUDA tile")
        with torch.cuda.device(self.device):
            return self.inner.forward(inputs, {"reconstruct": False}, out_dtype=torch.float32)

    def close(self):
        # CUDA cache owner retains storage until the recorded last-use event.
        self.closed = True
