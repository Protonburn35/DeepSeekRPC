"""Bounded FP32 pair state around vLLM's actual V4.1 compression kernel."""
from __future__ import annotations

from dataclasses import dataclass
import math
import threading

from .jsonutil import ValidationError, integer
from .protocol import hex_id
from .state import Operation


def group_boundaries(start: int, rows: int, ratio: int, *, context: int):
    """Return input offsets, compressed logical IDs and producing token IDs."""
    integer(context, "compression context", 1, 1048576)
    integer(start, "compression position", 0, context - 1)
    integer(rows, "compression rows", 1, min(64, context - start))
    integer(ratio, "compression ratio", 1, 2)
    tokens = tuple(p for p in range(start, start + rows) if (p + 1) % ratio == 0)
    return tuple(p - start for p in tokens), tuple(p // ratio for p in tokens), tokens


@dataclass(frozen=True)
class CompressedRows:
    operation: Operation
    logical_positions: tuple[int, ...]
    token_positions: tuple[int, ...]
    values: object


class DeviceCompressionState:
    """One session, one window, one actual CUDA stream; no CPU inference path.

    The ratio-2 kernel keeps its original FP32 [value | gate] rows. A fixed
    before-image plus at most 64 input rows restores any accepted prefix,
    including a rejected suffix that crossed a pair boundary. Output latents
    are normalized BF16 and must be consumed before this window finishes.
    """
    def __init__(self, ratio: int, norm_weight, *, eps: float, context: int,
                 admitted_bytes: int):
        import torch
        self.ratio = integer(ratio, "compression ratio", 1, 2)
        self.context = integer(context, "compression context", 1, 1048576)
        integer(admitted_bytes, "compression state admission", 1, 1 << 30)
        if (isinstance(eps, bool) or not isinstance(eps, (float, int)) or
                not math.isfinite(eps) or eps <= 0):
            raise ValidationError("compression RMSNorm epsilon must be positive and finite")
        if (norm_weight.device.type != "cuda" or not torch.cuda.is_available() or
                norm_weight.shape != (512,) or norm_weight.dtype != torch.bfloat16 or
                not norm_weight.is_contiguous() or not bool(torch.isfinite(norm_weight).all())):
            raise ValidationError("compression requires the actual learned BF16 CUDA norm")
        self.norm_weight, self.eps, self.device = norm_weight, float(eps), norm_weight.device
        # Fixed latent scratch plus the bounded selected-row result; norm
        # parameters and input scores belong to the parent graph's ledger.
        self.allocation_bytes = 2 * 64 * 512 * 2 + 4096
        if ratio == 2:
            self.allocation_bytes += (2 * 2 + 64) * 1024 * 4
        if self.allocation_bytes > admitted_bytes:
            raise ValidationError("FP32 compression state and bounded undo exceed admission")
        self._mutex = threading.RLock()
        self.active = self.session = None
        self.epoch, self.position = -1, 0
        self.closed = self._failed = self._processed = False
        with torch.cuda.device(self.device):
            self.stream = torch.cuda.current_stream(self.device)
            self._latent = torch.zeros((64, 512), dtype=torch.bfloat16, device=self.device)
            self._ring = self._before = self._scores = None
            if ratio == 2:
                self._ring = torch.zeros((1, 2, 1024), dtype=torch.float32, device=self.device)
                self._before = torch.zeros_like(self._ring)
                self._scores = torch.zeros((64, 1024), dtype=torch.float32, device=self.device)
            self.stream.synchronize()

    def _open(self):
        import torch
        if self.closed or torch.cuda.current_stream(self.device) != self.stream:
            raise ValidationError("compression state is closed or used from another compute stream")

    def reset(self, session: str, epoch: int):
        import torch
        with self._mutex, torch.cuda.device(self.device):
            self._open()
            hex_id(session, 32, "compression session")
            integer(epoch, "compression epoch", 0, (1 << 63) - 1)
            if self.active is not None or epoch <= self.epoch:
                raise ValidationError("compression reset requires quiescence and a newer epoch")
            if self._ring is not None:
                self._ring.zero_()
            self.stream.synchronize()
            self.session, self.epoch, self.position = session, epoch, 0
            self._failed = self._processed = False

    def begin(self, operation: Operation):
        import torch
        with self._mutex, torch.cuda.device(self.device):
            self._open()
            if (self.active is not None or operation.session != self.session or
                    operation.epoch != self.epoch or operation.start != self.position or
                    operation.rows > 64 or operation.start + operation.rows > self.context):
                raise ValidationError("compression window is stale, out of order or exceeds capacity")
            if self._ring is not None:
                # Until the backup finishes, no live state has been modified.
                self._before.copy_(self._ring)
                self.stream.synchronize()
            self.active = operation
            self._failed = self._processed = False

    def rewind_for_replay(self, session: str, epoch: int):
        """Restart compression after the coordinator validates full history."""
        import torch
        with self._mutex, torch.cuda.device(self.device):
            self._open()
            if self.active is not None or session != self.session or epoch != self.epoch or self._failed:
                raise ValidationError("compression replay rewind requires the current quiescent timeline")
            if self._ring is not None:
                self._ring.zero_()
            self.stream.synchronize()
            self.position = 0

    def forward(self, operation: Operation, scores) -> CompressedRows:
        import torch
        from vllm.models.deepseek_v41.common.ops.fused_compress_quant_cache import fused_save_compress_norm
        with self._mutex, torch.cuda.device(self.device):
            self._open()
            if (self.active != operation or self._processed or self._failed or
                    scores.device != self.device or scores.dtype != torch.float32 or
                    scores.shape != (operation.rows, 512 * self.ratio) or
                    not scores.is_contiguous() or not bool(torch.isfinite(scores).all())):
                raise ValidationError("compression input is stale, repeated or not finite FP32")
            offsets, logical, tokens = group_boundaries(
                operation.start, operation.rows, self.ratio, context=self.context)
            try:
                positions = torch.arange(operation.start, operation.start + operation.rows,
                                         dtype=torch.int64, device=self.device)
                slots = positions.remainder(2) if self.ratio == 2 else positions
                query_start = request_ids = None
                if self.ratio == 2:
                    self._scores[:operation.rows].copy_(scores)
                    query_start = torch.tensor([0, operation.rows], dtype=torch.int32, device=self.device)
                    request_ids = torch.zeros(operation.rows, dtype=torch.int32, device=self.device)
                output = self._latent[:operation.rows]
                output.zero_()
                fused_save_compress_norm(scores, positions, self._ring, slots, query_start,
                                         request_ids, self.norm_weight, self.eps, self.ratio, output)
                selected = output[torch.tensor(offsets, dtype=torch.int64, device=self.device)].contiguous()
                if not bool(torch.isfinite(selected).all()):
                    raise ValidationError("compressed BF16 latent overflowed or is nonfinite")
                self.stream.synchronize()
            except BaseException:
                self._failed = True
                raise
            self._processed = True
            return CompressedRows(operation, logical, tokens, selected)

    def finish(self, operation: Operation, accepted: int):
        import torch
        with self._mutex, torch.cuda.device(self.device):
            self._open()
            integer(accepted, "compression accepted prefix", 0, operation.rows)
            if self.active != operation or (accepted and (self._failed or not self._processed)):
                raise ValidationError("compression finish targets an inactive or incomplete window")
            try:
                if self._ring is not None:
                    self._ring.copy_(self._before)
                    # Reconstruct the exact original kernel's last two ring
                    # stores for the accepted prefix. No FP16/BF16 conversion.
                    for row in range(max(0, accepted - 2), accepted):
                        self._ring[0, (operation.start + row) % 2].copy_(self._scores[row])
                self.stream.synchronize()
            except BaseException:
                self._failed = True
                raise
            self.position = operation.start + accepted
            self.active = None
            self._failed = self._processed = False

    def close(self):
        import torch
        with self._mutex, torch.cuda.device(self.device):
            if not self.closed:
                self.stream.synchronize()
                self._ring = self._before = self._scores = self._latent = self.norm_weight = None
                self.closed = True
