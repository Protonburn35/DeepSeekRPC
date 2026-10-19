"""Lossless native TCP tensor transfer through fixed, bounded pinned tiles."""
from dataclasses import replace
import threading

from .fragments import FragmentCursor
from .jsonutil import ValidationError
from .protocol import DATA_KINDS, MAX_PAYLOAD


class CudaTensorWire:
    """One serialized data-plane owner; control/cancel use another connection.

    Destinations must already exist under the stage's memory admission. A peer
    never chooses an allocation. A failed or cancelled partial transfer closes
    the channel and must fault/abort its surrounding model window.
    """
    # Pinned tile + writable CPU tile + at most one immutable frame payload.
    staging_bytes = 3 * MAX_PAYLOAD

    def __init__(self, connection, *, device, cancelled=None):
        import torch
        if connection.plane != "data" or cancelled is not None and not callable(cancelled):
            raise ValidationError("CUDA tensor wire needs an actual data connection and cancellation observer")
        self.device = torch.device(device)
        if self.device.type != "cuda" or not torch.cuda.is_available():
            raise RuntimeError("CUDA tensor wire requires the actual model GPU")
        if self.device.index is None:
            self.device = torch.device("cuda", torch.cuda.current_device())
        self.connection, self.cancelled = connection, cancelled
        self._lock = threading.Lock()
        self._scratch = bytearray(MAX_PAYLOAD)
        self._scratch_tensor = torch.frombuffer(self._scratch, dtype=torch.uint8)
        self._pinned = torch.empty(MAX_PAYLOAD, dtype=torch.uint8, device="cpu", pin_memory=True)

    def _check(self, template, descriptor, tensor):
        import torch
        types = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32,
                 "i32": torch.int32, "i64": torch.int64, "u8": torch.uint8}
        if (template.kind not in DATA_KINDS or descriptor.fragment_offset or descriptor.fragment_bytes or
                tensor.device != self.device or tensor.dtype != types[descriptor.dtype] or
                tuple(tensor.shape) != descriptor.shape or not tensor.is_contiguous()):
            raise ValidationError("tensor wire storage differs from its locally admitted typed descriptor")
        return tensor.reshape(-1).view(torch.uint8)

    def _cancel(self):
        if self.cancelled is not None and self.cancelled():
            raise InterruptedError("tensor transfer cancelled at a bounded fragment boundary")

    def send(self, template, descriptor, tensor):
        import torch
        with self._lock, torch.cuda.device(self.device):
            try:
                raw = self._check(template, descriptor, tensor)
                if (template.run, template.sender, template.receiver) != (
                        self.connection.run, self.connection.local, self.connection.peer):
                    raise ValidationError("outgoing tensor template differs from its authenticated peer")
                offsets = range(0, descriptor.nbytes, MAX_PAYLOAD) if descriptor.nbytes else (0,)
                for sequence, start in enumerate(offsets):
                    self._cancel()
                    count = min(MAX_PAYLOAD, descriptor.nbytes - start)
                    if count:
                        self._pinned[:count].copy_(raw[start:start + count], non_blocking=False)
                        self._scratch_tensor[:count].copy_(self._pinned[:count])
                    payload = bytes(memoryview(self._scratch)[:count])
                    part = replace(descriptor, fragment_offset=start, fragment_bytes=count)
                    self.connection.send(replace(template, sequence=template.sequence + sequence,
                                                 tensor=part, payload=payload))
            except BaseException:
                self.connection.close()
                raise

    def receive(self, template, descriptor, destination):
        import torch
        with self._lock, torch.cuda.device(self.device):
            try:
                raw = self._check(template, descriptor, destination)
                if (template.run, template.sender, template.receiver) != (
                        self.connection.run, self.connection.peer, self.connection.local):
                    raise ValidationError("incoming tensor template differs from its authenticated peer")
                cursor = FragmentCursor(template, descriptor)
                while not cursor.complete:
                    self._cancel()
                    frame = self.connection.receive()
                    start, end = cursor.accept(frame)
                    count = end - start
                    if count:
                        self._scratch[:count] = frame.payload
                        self._pinned[:count].copy_(self._scratch_tensor[:count])
                        raw[start:end].copy_(self._pinned[:count], non_blocking=False)
                return destination
            except BaseException:
                self.connection.close()
                raise
