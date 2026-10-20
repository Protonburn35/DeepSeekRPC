"""Bounded safetensors metadata parsing and selective local payload access."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import os
import hashlib
import re
import struct
import threading

from .jsonutil import ValidationError, integer, loads
from .planner import Role, StagePlan

MAX_HEADER = 8 * 1024 * 1024
MAX_READ = 1024 * 1024
DTYPE_BYTES = {"F64": 8, "F32": 4, "F16": 2, "BF16": 2, "I64": 8, "U64": 8,
               "I32": 4, "U32": 4, "I16": 2, "U16": 2, "I8": 1, "U8": 1,
               "BOOL": 1, "F8_E4M3": 1, "F8_E5M2": 1, "F8_E8M0": 1}


@dataclass(frozen=True)
class TensorInfo:
    name: str
    dtype: str
    shape: tuple[int, ...]
    offset: int
    nbytes: int

    def as_dict(self):
        return asdict(self)


def parse_header(data: bytes, file_bytes: int) -> dict[str, TensorInfo]:
    integer(file_bytes, "file bytes", 10, (1 << 63) - 1)
    header = loads(data, limit=MAX_HEADER)
    if not isinstance(header, dict):
        raise ValidationError("safetensors header must be an object")
    base = 8 + len(data)
    if base > file_bytes:
        raise ValidationError("header extends beyond file")
    result, intervals = {}, []
    for name, entry in header.items():
        if name == "__metadata__":
            if not isinstance(entry, dict) or any(not isinstance(v, str) for v in entry.values()):
                raise ValidationError("safetensors metadata must contain strings")
            continue
        if not name or len(name) > 1024 or not isinstance(entry, dict):
            raise ValidationError("invalid tensor descriptor")
        if set(entry) != {"dtype", "shape", "data_offsets"}:
            raise ValidationError("unknown tensor descriptor fields")
        dtype = entry["dtype"]
        if dtype not in DTYPE_BYTES:
            raise ValidationError(f"unsupported tensor dtype: {dtype}")
        if not isinstance(entry["shape"], list) or len(entry["shape"]) > 8:
            raise ValidationError("invalid tensor rank")
        count = 1
        for dim in entry["shape"]:
            count *= integer(dim, "dimension", 0, (1 << 40))
            if count > (1 << 63) - 1:
                raise ValidationError("tensor element count overflow")
        offsets = entry["data_offsets"]
        if not isinstance(offsets, list) or len(offsets) != 2:
            raise ValidationError("invalid tensor offsets")
        begin, end = [integer(v, "offset", 0, file_bytes - base) for v in offsets]
        expected = count * DTYPE_BYTES[dtype]
        if end < begin or end - begin != expected:
            raise ValidationError("tensor dtype/shape/byte interval mismatch")
        result[name] = TensorInfo(name, dtype, tuple(entry["shape"]), base + begin, expected)
        intervals.append((begin, end))
    cursor = 0
    # Zero-byte tensors sort before positive-length entries at the same offset.
    for begin, end in sorted(intervals):
        if begin != cursor:
            raise ValidationError("overlapping tensor bytes or unindexed payload gap")
        cursor = end
    if cursor != file_bytes - base:
        raise ValidationError("unindexed trailing safetensors bytes")
    return result


class TensorFile:
    """Unbuffered bounded reads; callers own separate allocation/admission budgets.

    No demand mapping is used. Windows handles share reads but deny writers and
    deletion before header/payload verification and until close. This synchronous
    baseline does not claim qualified OVERLAPPED I/O or physical residency.
    """

    def __init__(self, path: Path):
        from .file_io import open_read
        self._file = open_read(path)
        self._lock = threading.Lock()
        try:
            size = os.fstat(self._file.fileno()).st_size
            prefix = self._file.read(8)
            if len(prefix) != 8:
                raise ValidationError("truncated safetensors length")
            length = struct.unpack("<Q", prefix)[0]
            if not 2 <= length <= min(MAX_HEADER, size - 8):
                raise ValidationError("safetensors header outside bounded file")
            data = self._file.read(length)
            if len(data) != length:
                raise ValidationError("short header read")
            self.tensors = parse_header(data, size)
            self.file_bytes = size
            self.header_sha256 = hashlib.sha256(data).hexdigest()
            self._identity = self._stat_identity()
        except BaseException:
            self._file.close()
            raise

    def _stat_identity(self):
        stat = os.fstat(self._file.fileno())
        return stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns

    def verify_file(self, expected_sha256: str):
        """Verify the local publisher payload in bounded buffers before loading.

        Model files must remain immutable after verification. This check reads
        the entire selected shard at boot, including any unowned bytes; it does
        not materialize unowned tensors or download anything.
        """
        from .protocol import hex_id
        hex_id(expected_sha256, 64, "publisher shard")
        with self._lock:
            if self._file.closed or self._stat_identity() != self._identity:
                raise ValidationError("model shard closed or changed before integrity verification")
            self._file.seek(0)
            digest = hashlib.sha256()
            while chunk := self._file.read(MAX_READ):
                digest.update(chunk)
            if digest.hexdigest() != expected_sha256 or self._stat_identity() != self._identity:
                raise ValidationError("local model shard publisher SHA-256 mismatch or concurrent mutation")

    def read_range(self, name: str, offset: int, length: int) -> bytes:
        info = self.tensors[name]
        integer(offset, "tensor-relative offset", 0, info.nbytes)
        integer(length, "bounded read length", 0, min(MAX_READ, info.nbytes - offset))
        with self._lock:
            if self._file.closed:
                raise ValidationError("payload file is closed")
            if self._stat_identity() != self._identity:
                raise ValidationError("model shard changed after admission")
            self._file.seek(info.offset + offset)
            data = self._file.read(length)
            if len(data) != length:
                raise ValidationError("short tensor payload read")
            return data

    def chunks(self, name: str, chunk_bytes: int = MAX_READ):
        integer(chunk_bytes, "chunk size", 1, MAX_READ)
        for offset in range(0, self.tensors[name].nbytes, chunk_bytes):
            yield self.read_range(name, offset, min(chunk_bytes, self.tensors[name].nbytes - offset))

    def close(self):
        with self._lock:
            self._file.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


def ownership(name: str, plan: StagePlan) -> tuple[Role, str]:
    match = re.match(r"layers\.(\d+)\.", name)
    if match:
        layer = int(match.group(1))
        role = plan.owner(layer)
        if ".ffn.experts." in name:
            component = "routed_experts"
        elif ".ffn.shared_experts." in name:
            component = "shared_expert"
        elif ".engram.embed." in name:
            component = "engram_table"
        elif ".engram." in name:
            component = "engram_projection"
        elif ".attn." in name:
            component = "attention_index"
        else:
            component = "layer_other"
        return role, component
    root = name.split(".")[0]
    if root in {"embed", "vision", "aligner", "image_start", "image_end", "image_newline"}:
        return Role.HOST, "embedding" if root == "embed" else "vision"
    if root in {"mtp", "head", "norm"}:
        return Role.TERMINAL, "draft" if root == "mtp" else "head_norm"
    raise ValidationError(f"tensor ownership is not defined: {name}")
