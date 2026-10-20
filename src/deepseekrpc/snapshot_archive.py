"""Bounded state archives and atomic, confined host-owned snapshot storage."""
from contextlib import contextmanager
from dataclasses import dataclass
import hashlib
import math
import os
from pathlib import Path
import re
import shutil
import struct
import uuid

from .jsonutil import ValidationError, canonical, fingerprint, integer, loads
from .protocol import MAX_PAYLOAD, hex_id

MAGIC = b"DSRSNAP\0"
PREFIX = struct.Struct("<8sII")
MAX_HEADER = 8 * 1024 ** 2
MAX_RECORDS = 16384
MAX_ARCHIVE = 80 * 1024 ** 3
DTYPES = {"u8": 1, "i32": 4, "i64": 8, "bf16": 2, "fp32": 4}
OWNER = b"DeepSeekRPC three-stage snapshot store v1\n"


@dataclass(frozen=True)
class Record:
    name: str
    dtype: str
    shape: tuple[int, ...]

    def __post_init__(self):
        if (not isinstance(self.name, str) or not re.fullmatch(r"[a-zA-Z0-9_.:-]{1,160}", self.name) or
                not isinstance(self.dtype, str) or self.dtype not in DTYPES or
                not isinstance(self.shape, tuple) or not 1 <= len(self.shape) <= 4):
            raise ValidationError("invalid bounded snapshot record descriptor")
        for dim in self.shape:
            integer(dim, "snapshot record dimension", 0, 32 * 1024 ** 2)
        if self.nbytes > 2 * 1024 ** 3:
            raise ValidationError("snapshot record exceeds its state/image bound")

    @property
    def nbytes(self):
        return math.prod(self.shape) * DTYPES[self.dtype]

    def as_dict(self):
        return {"name": self.name, "dtype": self.dtype, "shape": list(self.shape)}


def encode_header(metadata, records):
    if not isinstance(metadata, dict) or not isinstance(records, (list, tuple)) or len(records) > MAX_RECORDS:
        raise ValidationError("snapshot metadata/record inventory exceeds bounds")
    if any(not isinstance(r, Record) for r in records) or len({r.name for r in records}) != len(records):
        raise ValidationError("snapshot contains an invalid or duplicated record")
    encoded = canonical({"metadata": metadata, "records": [r.as_dict() for r in records]})
    if len(encoded) > MAX_HEADER or PREFIX.size + len(encoded) + sum(r.nbytes for r in records) > MAX_ARCHIVE:
        raise ValidationError("snapshot exceeds its bounded header/total state size")
    return PREFIX.pack(MAGIC, 1, len(encoded)) + encoded


def read_header(stream, *, expected_bytes=None):
    initial = stream.read(PREFIX.size)
    if len(initial) != PREFIX.size:
        raise ValidationError("truncated snapshot prefix")
    magic, version, size = PREFIX.unpack(initial)
    if magic != MAGIC or version != 1 or not 1 <= size <= MAX_HEADER:
        raise ValidationError("unsupported or oversized snapshot header")
    data = stream.read(size)
    if len(data) != size:
        raise ValidationError("truncated snapshot header")
    value = loads(data, limit=MAX_HEADER)
    if (not isinstance(value, dict) or set(value) != {"metadata", "records"} or not isinstance(value["metadata"], dict) or
            not isinstance(value["records"], list) or len(value["records"]) > MAX_RECORDS):
        raise ValidationError("unknown snapshot header fields")
    records = []
    for record in value["records"]:
        if not isinstance(record, dict) or set(record) != {"name", "dtype", "shape"} or not isinstance(record["shape"], list):
            raise ValidationError("unknown snapshot record fields")
        records.append(Record(record["name"], record["dtype"], tuple(record["shape"])))
    header = encode_header(value["metadata"], records)
    if header != initial + data:
        raise ValidationError("snapshot header is not canonical")
    total = len(header) + sum(r.nbytes for r in records)
    if expected_bytes is not None and total != expected_bytes:
        raise ValidationError("snapshot inventory differs from its complete file length")
    return value["metadata"], tuple(records), header, total


def archive_chunks(header, records, readers):
    """Pack arbitrary bounded record readers into <=1 MiB archive tiles."""
    import io
    metadata, validated, actual, _ = read_header(io.BytesIO(header))
    if actual != header or validated != tuple(records) or set(readers) != {r.name for r in records}:
        raise ValidationError("snapshot readers differ from the admitted archive inventory")
    buffer = bytearray()

    def pack(data):
        view = memoryview(data)
        for start in range(0, len(view), MAX_PAYLOAD):
            part = view[start:start + MAX_PAYLOAD]
            while part:
                count = min(MAX_PAYLOAD - len(buffer), len(part))
                buffer.extend(part[:count])
                part = part[count:]
                if len(buffer) == MAX_PAYLOAD:
                    yield bytes(buffer)
                    buffer.clear()

    yield from pack(header)
    for record in records:
        count = 0
        for part in readers[record.name]():
            if not isinstance(part, (bytes, bytearray, memoryview)) or len(part) > MAX_PAYLOAD:
                raise ValidationError("snapshot record reader exceeds its bounded transfer tile")
            count += len(part)
            if count > record.nbytes:
                raise ValidationError("snapshot reader exceeded its declared record length")
            yield from pack(part)
        if count != record.nbytes:
            raise ValidationError("snapshot reader ended before its declared record length")
    if buffer:
        yield bytes(buffer)


def file_chunks(stream, count):
    while count:
        data = stream.read(min(MAX_PAYLOAD, count))
        if not data:
            raise ValidationError("snapshot record is truncated")
        count -= len(data)
        yield data


class SnapshotStore:
    def __init__(self, root: Path, *, identity, maximum=16):
        self.identity = hex_id(identity, 64, "snapshot model/runtime/session compatibility")
        self.maximum = integer(maximum, "snapshot count bound", 1, 128)
        root.mkdir(parents=True, exist_ok=True)
        self.root = root.resolve(strict=True)
        marker = self.root / ".owner"
        if not marker.exists():
            if next(self.root.iterdir(), None) is not None:
                raise ValidationError("snapshots require a dedicated empty host directory")
            marker.write_bytes(OWNER)
        if marker.is_symlink() or marker.stat().st_size != len(OWNER) or marker.read_bytes() != OWNER:
            raise ValidationError("snapshot storage ownership marker differs")

    def _directory(self, identifier):
        hex_id(identifier, 32, "opaque snapshot")
        path = self.root / identifier
        if path.is_symlink() or path.resolve().parent != self.root:
            raise ValidationError("snapshot directory escapes its host-owned root")
        return path

    def list(self):
        result = []
        for index, path in enumerate(self.root.iterdir()):
            if index > self.maximum + 5:
                raise ValidationError("snapshot directory inventory exceeds its bounded quota")
            if re.fullmatch(r"[a-f0-9]{32}", path.name) and not path.is_symlink() and path.is_dir():
                manifest = self._manifest(path.name)
                result.append({"id": path.name, "bytes": sum(f["bytes"] for f in manifest["files"].values()),
                               "identity": manifest["identity"]})
        return sorted(result, key=lambda record: record["id"])

    def _manifest(self, identifier):
        directory = self._directory(identifier)
        path = directory / "manifest.json"
        if path.is_symlink() or not path.is_file() or path.stat().st_size > 64 * 1024:
            raise ValidationError("snapshot has no bounded committed manifest")
        with path.open("rb") as stream:
            value = loads(stream.read(64 * 1024 + 1), limit=64 * 1024)
        if (not isinstance(value, dict) or set(value) != {"format", "id", "identity", "timeline", "files"} or
                type(value["format"]) is not int or value["format"] != 1 or value["id"] != identifier or
                not isinstance(value["files"], dict) or set(value["files"]) != set("ABC") or not isinstance(value["timeline"], dict)):
            raise ValidationError("snapshot manifest is incomplete or has unknown fields")
        hex_id(value["identity"], 64, "snapshot identity")
        for stage, record in value["files"].items():
            if not isinstance(record, dict) or set(record) != {"bytes", "sha256", "header_sha256"}:
                raise ValidationError("snapshot file metadata is incomplete")
            integer(record["bytes"], "snapshot file length", PREFIX.size, MAX_ARCHIVE)
            for key in ("sha256", "header_sha256"):
                hex_id(record[key], 64, "snapshot file checksum")
        return value

    @contextmanager
    def prepare(self, *, timeline, expected_bytes, identifier=None):
        integer(expected_bytes, "snapshot expected total disk bytes", 1, 3 * MAX_ARCHIVE)
        if len(self.list()) >= self.maximum:
            raise ValidationError("snapshot quota is full; explicitly remove a snapshot before saving another")
        if shutil.disk_usage(self.root).free < expected_bytes + 2 * 1024 ** 3:
            raise ValidationError("host disk lacks snapshot bytes plus the required 2 GiB free-space reserve")
        if not isinstance(timeline, dict) or len(canonical(timeline)) > 16 * 1024:
            raise ValidationError("snapshot timeline metadata exceeds its bounded inventory")
        identifier = uuid.uuid4().hex if identifier is None else hex_id(identifier, 32, "new opaque snapshot")
        if self._directory(identifier).exists():
            raise ValidationError("snapshot identifier already exists")
        pending = self.root / f".pending-{identifier}"
        pending.mkdir()
        transaction = SnapshotWrite(self, identifier, pending, timeline, expected_bytes)
        try:
            yield transaction
        finally:
            if not transaction.committed:
                # Only known generated files may be removed, never recursively
                # follow an unexpected path that appeared during a failed save.
                for name in ("stage-A.bin", "stage-B.bin", "stage-C.bin", "manifest.json"):
                    path = pending / name
                    if path.is_symlink():
                        raise ValidationError("snapshot temporary file was redirected")
                    if path.exists():
                        path.unlink()
                pending.rmdir()

    def validate(self, identifier):
        with self.validated(identifier) as (manifest, headers, streams):
            return manifest, headers

    @contextmanager
    def validated(self, identifier):
        """Hold all three files open through validation and restore.

        On Windows these handles deny write/delete sharing, so an archive
        cannot change between the validation and apply passes.
        """
        from contextlib import ExitStack
        manifest = self._manifest(identifier)
        if manifest["identity"] != self.identity:
            raise ValidationError("snapshot model/runtime/precision/plan compatibility differs")
        directory, headers = self._directory(identifier), {}
        if {path.name for path in directory.iterdir()} != {"manifest.json", "stage-A.bin", "stage-B.bin", "stage-C.bin"}:
            raise ValidationError("snapshot directory has missing or unknown files")
        with ExitStack() as ownership:
            streams = {}
            for stage, record in manifest["files"].items():
                path = directory / f"stage-{stage}.bin"
                if path.is_symlink() or not path.is_file() or path.stat().st_size != record["bytes"]:
                    raise ValidationError("snapshot stage file is missing, redirected or truncated")
                stream = ownership.enter_context(locked_read(path))
                streams[stage] = stream
                metadata, records, header, _ = read_header(stream, expected_bytes=record["bytes"])
                digest = hashlib.sha256(header)
                for data in file_chunks(stream, record["bytes"] - len(header)):
                    digest.update(data)
                if (digest.hexdigest() != record["sha256"] or hashlib.sha256(header).hexdigest() != record["header_sha256"]):
                    raise ValidationError("snapshot stage file failed its complete pre-restore checksum")
                headers[stage] = metadata, records
                stream.seek(0)
            yield manifest, headers, streams

    def open_stage(self, identifier, stage):
        if stage not in set("ABC"):
            raise ValidationError("unknown snapshot stage")
        path = self._directory(identifier) / f"stage-{stage}.bin"
        if path.is_symlink():
            raise ValidationError("snapshot stage file is redirected")
        return path.open("rb")

    def remove(self, identifier):
        self._manifest(identifier)
        directory = self._directory(identifier)
        names = {"manifest.json", "stage-A.bin", "stage-B.bin", "stage-C.bin"}
        if ({path.name for path in directory.iterdir()} != names or
                any((directory / name).is_symlink() or not (directory / name).is_file() for name in names)):
            raise ValidationError("snapshot contains an unexpected or redirected entry")
        destination = self.root / f".deleting-{identifier}"
        if destination.exists() or destination.is_symlink():
            raise ValidationError("snapshot deletion transaction already exists")
        os.replace(directory, destination)
        for name in names:
            (destination / name).unlink()
        destination.rmdir()
        return {"id": identifier, "deleted": True}


@contextmanager
def locked_read(path):
    from .file_io import open_read
    with open_read(path, buffering=-1, sequential=True) as stream:
        yield stream


class SnapshotWrite:
    def __init__(self, store, identifier, directory, timeline, expected):
        if not isinstance(timeline, dict) or len(canonical(timeline)) > 16 * 1024:
            raise ValidationError("snapshot timeline metadata exceeds its bounded inventory")
        self.store, self.id, self.directory = store, identifier, directory
        self.timeline, self.expected, self.files, self.committed = timeline, expected, {}, False

    def write_stage(self, stage, chunks):
        if stage not in set("ABC") or stage in self.files or self.committed:
            raise ValidationError("snapshot stage is duplicated or already committed")
        digest, count = hashlib.sha256(), 0
        path = self.directory / f"stage-{stage}.bin"
        try:
            with path.open("xb") as stream:
                for data in chunks:
                    if not isinstance(data, bytes) or not 0 < len(data) <= MAX_PAYLOAD:
                        raise ValidationError("snapshot writer needs nonempty bounded tiles")
                    count += len(data)
                    if count > MAX_ARCHIVE or count > self.expected:
                        raise ValidationError("snapshot stream exceeded its admitted host disk length")
                    stream.write(data)
                    digest.update(data)
                stream.flush()
                os.fsync(stream.fileno())
        finally:
            if callable(getattr(chunks, "close", None)):
                chunks.close()
        with path.open("rb") as stream:
            _, _, header, _ = read_header(stream, expected_bytes=count)
        self.files[stage] = {"bytes": count, "sha256": digest.hexdigest(), "header_sha256": hashlib.sha256(header).hexdigest()}
        return dict(self.files[stage])

    def commit(self):
        if self.committed or set(self.files) != set("ABC") or sum(f["bytes"] for f in self.files.values()) != self.expected:
            raise ValidationError("all three complete stage streams must match the admitted total before snapshot publication")
        value = {"format": 1, "id": self.id, "identity": self.store.identity, "timeline": self.timeline, "files": self.files}
        with (self.directory / "manifest.json").open("xb") as stream:
            stream.write(canonical(value))
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(self.directory, self.store._directory(self.id))
        self.committed = True
        return value
