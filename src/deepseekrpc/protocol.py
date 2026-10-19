"""Bounded, typed, lossless wire framing. Authentication is provided by TLS."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import IntEnum
import hashlib
import re
import struct

from .jsonutil import ValidationError, canonical, integer, loads
from .planner import Role

PREFIX = struct.Struct("!4sBBHII")
MAX_ENVELOPE = 16 * 1024
MAX_PAYLOAD = 1024 * 1024
MAX_TENSOR = 512 * 1024 * 1024
MAX_CONTEXT = 1048576
U64 = (1 << 64) - 1
DTYPES = {"fp16": 2, "bf16": 2, "fp32": 4, "i32": 4, "i64": 8, "u8": 1}
SEMANTICS = {"hidden_states", "pre_mix", "token_ids", "positions", "image_mask",
             "validity", "kv", "index_k", "topk", "candidates", "snapshot"}


class Kind(IntEnum):
    HELLO = 1
    CAPABILITIES = 2
    PLAN_ACK = 3
    READY = 4
    BEGIN = 10
    PREFIX_PROBE = 11
    PREFIX_ACCEPT = 12
    RESET = 13
    STEP_BEGIN = 20
    SOURCE_DELTA = 21
    STAGE_INPUT = 22
    STAGE_DONE = 23
    DRAFT = 30
    VERIFY_RESULT = 31
    COMMIT = 32
    COMMIT_ACK = 33
    ABORT = 34
    CANCEL = 40
    CANCEL_ACK = 41
    HEARTBEAT = 42
    ERROR = 43
    SNAPSHOT_PREPARE = 50
    SNAPSHOT_CHUNK = 51
    SNAPSHOT_SEAL = 52
    RESTORE_PREPARE = 53
    RESTORE_APPLY = 54
    RESTORE_ACK = 55


DATA_KINDS = {Kind.SOURCE_DELTA, Kind.STAGE_INPUT, Kind.SNAPSHOT_CHUNK}


def hex_id(value, size: int, name: str) -> str:
    if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{" + str(size) + r"}", value):
        raise ValidationError(f"invalid {name} identity")
    return value


@dataclass(frozen=True)
class TensorDescriptor:
    semantic: str
    source_layer: int
    dtype: str
    shape: tuple[int, ...]
    logical_row: int
    tensor_id: str
    fragment_offset: int
    fragment_bytes: int

    def __post_init__(self):
        if self.semantic not in SEMANTICS or self.dtype not in DTYPES:
            raise ValidationError("unknown tensor semantic or dtype")
        integer(self.source_layer, "source layer", -1, 42)
        integer(self.logical_row, "logical row", 0, MAX_CONTEXT)
        hex_id(self.tensor_id, 32, "tensor")
        if not isinstance(self.shape, tuple) or len(self.shape) > 4:
            raise ValidationError("wire tensor rank must be at most four")
        count = 1
        for dim in self.shape:
            count *= integer(dim, "tensor dimension", 0, MAX_CONTEXT)
            if count * DTYPES[self.dtype] > MAX_TENSOR:
                raise ValidationError("tensor exceeds negotiated hard bound")
        size = count * DTYPES[self.dtype]
        integer(self.fragment_offset, "fragment offset", 0, size)
        integer(self.fragment_bytes, "fragment bytes", 0, min(MAX_PAYLOAD, size - self.fragment_offset))
        if self.fragment_offset % DTYPES[self.dtype] or self.fragment_bytes % DTYPES[self.dtype]:
            raise ValidationError("fragment splits an element")
        if self.semantic in {"hidden_states", "pre_mix"} and self.dtype != "fp32":
            raise ValidationError("this protocol implementation requires FP32 activation wire")
        if self.semantic in {"token_ids", "positions", "topk", "candidates"} and self.dtype not in {"i32", "i64"}:
            raise ValidationError("integer semantic requires lossless integer storage")
        if self.shape and self.logical_row + self.shape[0] > MAX_CONTEXT:
            raise ValidationError("tensor logical row interval exceeds context")

    @property
    def nbytes(self) -> int:
        count = DTYPES[self.dtype]
        for dim in self.shape:
            count *= dim
        return count


@dataclass(frozen=True)
class Frame:
    kind: Kind
    run: str
    sender: Role
    receiver: Role
    session: str
    request: str
    epoch: int
    operation: int
    sequence: int
    transaction: str
    position: int = 0
    rows: int = 0
    slot_generation: int = 0
    flags: int = 0
    tensor: TensorDescriptor | None = None
    payload: bytes = b""

    def __post_init__(self):
        if not isinstance(self.kind, Kind) or not isinstance(self.sender, Role) or not isinstance(self.receiver, Role):
            raise ValidationError("untyped frame kind or role")
        if self.sender == self.receiver:
            raise ValidationError("sender and receiver must differ")
        hex_id(self.run, 64, "run")
        for name in ("session", "request", "transaction"):
            hex_id(getattr(self, name), 32, name)
        for name in ("epoch", "operation", "sequence", "slot_generation"):
            integer(getattr(self, name), name, 0, U64)
        integer(self.flags, "flags", 0, 0)
        integer(self.position, "position", 0, MAX_CONTEXT)
        integer(self.rows, "rows", 0, min(4096, MAX_CONTEXT - self.position))
        if not isinstance(self.payload, bytes) or len(self.payload) > MAX_PAYLOAD:
            raise ValidationError("frame payload exceeds hard bound")
        if self.tensor is not None:
            if not isinstance(self.tensor, TensorDescriptor) or self.tensor.fragment_bytes != len(self.payload):
                raise ValidationError("tensor fragment does not cover payload exactly")
            if self.kind not in DATA_KINDS:
                raise ValidationError("tensor payload on control message")
        elif self.kind in DATA_KINDS:
            raise ValidationError("data message requires a tensor descriptor")

    def encode(self) -> tuple[bytes, bytes, bytes]:
        envelope = asdict(self)
        del envelope["payload"]
        del envelope["kind"]
        envelope["payload_sha256"] = hashlib.sha256(self.payload).hexdigest()
        metadata = canonical(envelope)
        if len(metadata) > MAX_ENVELOPE:
            raise ValidationError("envelope exceeds hard bound")
        return PREFIX.pack(b"DSRP", 1, 0, self.kind, len(metadata), len(self.payload)), metadata, self.payload


def parse_prefix(data: bytes) -> tuple[Kind, int, int]:
    if len(data) != PREFIX.size:
        raise ValidationError("truncated frame prefix")
    magic, major, minor, kind, header_bytes, payload_bytes = PREFIX.unpack(data)
    if magic != b"DSRP" or (major, minor) != (1, 0):
        raise ValidationError("unsupported protocol magic/version")
    integer(header_bytes, "envelope length", 1, MAX_ENVELOPE)
    integer(payload_bytes, "payload length", 0, MAX_PAYLOAD)
    try:
        return Kind(kind), header_bytes, payload_bytes
    except ValueError as exc:
        raise ValidationError("unknown message kind") from exc


def decode(kind: Kind, metadata: bytes, payload: bytes) -> Frame:
    envelope = loads(metadata, limit=MAX_ENVELOPE)
    expected = set(Frame.__dataclass_fields__) - {"kind", "payload"} | {"payload_sha256"}
    if not isinstance(envelope, dict) or set(envelope) != expected:
        raise ValidationError("incomplete or unknown envelope fields")
    digest = hex_id(envelope.pop("payload_sha256"), 64, "payload digest")
    if digest != hashlib.sha256(payload).hexdigest():
        raise ValidationError("payload digest mismatch")
    try:
        # bool is an int subclass, so reject it before enum conversion.
        envelope["sender"] = Role(integer(envelope["sender"], "sender", 0, 2))
        envelope["receiver"] = Role(integer(envelope["receiver"], "receiver", 0, 2))
        descriptor = envelope["tensor"]
        if descriptor is not None:
            if not isinstance(descriptor, dict) or set(descriptor) != set(TensorDescriptor.__dataclass_fields__):
                raise ValidationError("unknown tensor descriptor fields")
            if not isinstance(descriptor["shape"], list):
                raise ValidationError("wire shape must be an array")
            descriptor["shape"] = tuple(descriptor["shape"])
            envelope["tensor"] = TensorDescriptor(**descriptor)
        return Frame(kind=kind, payload=payload, **envelope)
    except (TypeError, ValueError) as exc:
        raise ValidationError("invalid frame field types or values") from exc
