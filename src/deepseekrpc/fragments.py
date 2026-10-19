"""Streaming tensor fragments into caller-admitted storage; no peer-driven allocation."""
from dataclasses import replace

from .jsonutil import ValidationError, integer
from .protocol import DTYPES, MAX_PAYLOAD, Frame, TensorDescriptor


def fragments(template: Frame, descriptor: TensorDescriptor, storage,
              chunk_bytes: int = MAX_PAYLOAD):
    integer(chunk_bytes, "chunk size", 1, MAX_PAYLOAD)
    if chunk_bytes % DTYPES[descriptor.dtype]:
        raise ValidationError("chunk size must preserve element boundaries")
    view = memoryview(storage).cast("B")
    if len(view) != descriptor.nbytes:
        raise ValidationError("storage differs from declared tensor length")
    offsets = range(0, len(view), chunk_bytes) if view else (0,)
    for sequence, offset in enumerate(offsets):
        length = min(chunk_bytes, len(view) - offset)
        part = replace(descriptor, fragment_offset=offset, fragment_bytes=length)
        yield replace(template, tensor=part, sequence=template.sequence + sequence,
                      payload=bytes(view[offset:offset + length]))


class FragmentCursor:
    """Validate a stream before a CPU/GPU consumer writes an admitted range."""

    def __init__(self, template: Frame, descriptor: TensorDescriptor):
        self.template, self.descriptor = template, descriptor
        self.offset, self.sequence = 0, template.sequence
        self.complete = False

    def accept(self, frame: Frame) -> tuple[int, int]:
        if self.complete or frame.tensor is None:
            raise ValidationError("unexpected fragment after completion or without descriptor")
        # Bind every fragment to the operation/timeline, not just the tensor ID.
        for name in ("kind", "run", "sender", "receiver", "session", "request", "epoch",
                     "operation", "transaction", "position", "rows", "slot_generation", "flags"):
            if getattr(frame, name) != getattr(self.template, name):
                raise ValidationError("fragment operation identity mismatch")
        d = frame.tensor
        for name in ("semantic", "source_layer", "dtype", "shape", "logical_row", "tensor_id"):
            if getattr(d, name) != getattr(self.descriptor, name):
                raise ValidationError("fragment tensor identity mismatch")
        if frame.sequence != self.sequence or d.fragment_offset != self.offset:
            raise ValidationError("out-of-order, duplicate or overlapping fragment")
        if not d.fragment_bytes and d.nbytes:
            raise ValidationError("zero-progress fragment")
        end = self.offset + d.fragment_bytes
        if end > self.descriptor.nbytes:
            raise ValidationError("fragment overflows admitted storage")
        start = self.offset
        self.offset, self.sequence = end, self.sequence + 1
        self.complete = end == self.descriptor.nbytes
        return start, end


class FragmentReceiver(FragmentCursor):
    """Storage is preallocated by local policy, never sized from an incoming frame."""

    def __init__(self, template: Frame, descriptor: TensorDescriptor, storage):
        self.storage = memoryview(storage).cast("B")
        if self.storage.readonly or len(self.storage) != descriptor.nbytes:
            raise ValidationError("destination must be admitted, writable, and exactly sized")
        super().__init__(template, descriptor)

    def accept(self, frame: Frame) -> bool:
        start, end = super().accept(frame)
        self.storage[start:end] = frame.payload
        return self.complete
