"""Selective local shard loading under pinned metadata and layer ownership.

No checkpoint acquisition, demand mapping or inference fallback is implemented
here. B's boot source can be permanently closed after its resident preload.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
import re
import threading
from . import metrics

from .inventory import MAX_READ, TensorFile, ownership
from .jsonutil import ValidationError, integer, loads
from .planner import Role, StagePlan
from .protocol import hex_id

MAX_INDEX = 32 * 1024 * 1024
MAX_TENSORS = 200000
SHARD_NAME = re.compile(r"model-[0-9]{5}-of-[0-9]{5}\.safetensors\Z")


@dataclass(frozen=True)
class ShardPin:
    name: str
    revision: str
    file_bytes: int
    file_sha256: str
    header_bytes: int
    header_sha256: str

    def __post_init__(self):
        if not isinstance(self.name, str) or not SHARD_NAME.fullmatch(self.name):
            raise ValidationError("shard pin requires a plain checkpoint shard filename")
        hex_id(self.revision, 40, "model revision")
        hex_id(self.file_sha256, 64, "publisher shard")
        hex_id(self.header_sha256, 64, "shard header")
        integer(self.file_bytes, "shard bytes", 10, (1 << 63) - 1)
        integer(self.header_bytes, "shard header bytes", 2, min(8 * 1024 * 1024, self.file_bytes - 8))

    @classmethod
    def from_lock(cls, lock: dict, *, model: str = "pack") -> dict[str, ShardPin]:
        pins = {}
        for entry in lock["files"]:
            if entry["model"] == model:
                pin = cls(**{field: entry[field] for field in cls.__dataclass_fields__})
                if pin.name in pins:
                    raise ValidationError("duplicate shard pin")
                pins[pin.name] = pin
        if not pins or len({p.revision for p in pins.values()}) != 1:
            raise ValidationError("missing pins or mixed model revisions")
        return pins


class SelectiveWeights:
    """Open and verify only shards containing this rank's indexed weights.

    Metadata and publisher hashes are trusted local deployment inputs. Files
    must be immutable; Windows handles enforce deny-write/delete sharing before
    verification. Integrity verification reads the whole local shard at boot,
    then tensor reads stay bounded to the owned byte ranges. This is an I/O
    component, not proof of physical residency or a model executor.
    """
    def __init__(self, root: Path, index: Path, *, index_sha256: str,
                 pins: dict[str, ShardPin], plan: StagePlan, role: Role,
                 replicate_draft_embedding: bool = False):
        if not isinstance(role, Role):
            raise ValidationError("weight owner must be an explicit stage role")
        if type(replicate_draft_embedding) is not bool or replicate_draft_embedding and role != Role.TERMINAL:
            raise ValidationError("only stage C may admit the explicit DSpark text-embedding replica")
        hex_id(index_sha256, 64, "checkpoint index")
        with index.open("rb") as stream:
            data = stream.read(MAX_INDEX + 1)
        if len(data) > MAX_INDEX or hashlib.sha256(data).hexdigest() != index_sha256:
            raise ValidationError("checkpoint index bound or fingerprint mismatch")
        parsed = loads(data, limit=MAX_INDEX)
        mapping = parsed.get("weight_map") if isinstance(parsed, dict) else None
        if not isinstance(mapping, dict) or not 1 <= len(mapping) <= MAX_TENSORS:
            raise ValidationError("checkpoint index tensor count outside bound")
        if not pins or any(name != pin.name for name, pin in pins.items()):
            raise ValidationError("shard pin dictionary has conflicting identities")
        if len({p.revision for p in pins.values()}) != 1:
            raise ValidationError("weight source mixes revisions")
        self.role, self.plan, self.root = role, plan, root.resolve(strict=True)
        self._mapping, self._pins = mapping, dict(pins)
        selected = set()
        for name, filename in mapping.items():
            if not isinstance(name, str) or len(name) > 1024 or not isinstance(filename, str) or filename not in pins:
                raise ValidationError("invalid indexed tensor identity or missing shard pin")
            if ownership(name, plan)[0] == role:
                selected.add(name)
        if not selected:
            raise ValidationError("checkpoint index has no owned tensors")
        self.replicated_names = frozenset({"embed.weight"}) if replicate_draft_embedding else frozenset()
        if not self.replicated_names <= mapping.keys():
            raise ValidationError("DSpark embedding replica is absent from the pinned checkpoint index")
        selected.update(self.replicated_names)
        self.names = frozenset(selected)
        self._files: dict[str, TensorFile] = {}
        self._mutex = threading.RLock()
        self.closed = False
        self.verified_shards = self.integrity_bytes = self.payload_bytes = 0

    def _source(self, name: str) -> TensorFile:
        if self.closed:
            raise ValidationError("weight source is permanently closed; disk fallback is forbidden")
        if name not in self.names:
            raise ValidationError("tensor is not owned by this stage")
        filename = self._mapping[name]
        source = self._files.get(filename)
        if source is None:
            pin = self._pins[filename]
            path = (self.root / filename).resolve(strict=True)
            if path.parent != self.root:
                raise ValidationError("model shard escapes its admitted directory")
            source = TensorFile(path)
            try:
                if (source.file_bytes != pin.file_bytes or source.header_sha256 != pin.header_sha256 or
                        min(t.offset for t in source.tensors.values()) != 8 + pin.header_bytes):
                    raise ValidationError("local shard differs from the pinned physical header")
                indexed = {key for key, value in self._mapping.items() if value == filename}
                if set(source.tensors) != indexed:
                    raise ValidationError("checkpoint index and pinned shard tensor names disagree")
                with metrics.current.measure("integrity_read"):
                    source.verify_file(pin.file_sha256)
                metrics.current.add("integrity_read_bytes", pin.file_bytes)
            except BaseException:
                source.close()
                raise
            self._files[filename] = source
            self.verified_shards += 1
            self.integrity_bytes += pin.file_bytes
        return source

    def info(self, name: str):
        with self._mutex:
            return self._source(name).tensors[name]

    def read_range(self, name: str, offset: int, length: int) -> bytes:
        with self._mutex:
            with metrics.current.measure("model_file_read"):
                result = self._source(name).read_range(name, offset, length)
            self.payload_bytes += len(result)
            metrics.current.add("model_file_reads")
            metrics.current.add("model_file_read_bytes", len(result))
            metrics.current.event("file_read", bytes=len(result))
            return result

    def chunks(self, name: str, chunk_bytes: int = MAX_READ):
        integer(chunk_bytes, "owned tensor chunk size", 1, MAX_READ)
        size = self.info(name).nbytes
        for offset in range(0, size, chunk_bytes):
            yield self.read_range(name, offset, min(chunk_bytes, size - offset))

    def close(self):
        with self._mutex:
            for source in self._files.values():
                source.close()
            self._files.clear()
            self.closed = True

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
