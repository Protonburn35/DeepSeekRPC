import hashlib
import json
import os
from pathlib import Path
import struct

import numpy as np
import pytest
from safetensors.numpy import save_file

from deepseekrpc.inventory import TensorFile
from deepseekrpc.jsonutil import ValidationError
from deepseekrpc.planner import Geometry, Role, StagePlan
from deepseekrpc.resident_weights import LockedOwnedWeights
from deepseekrpc.weights import SelectiveWeights, ShardPin


@pytest.fixture
def checkpoint(tmp_path):
    config = json.loads((Path(__file__).parents[1] / "provenance/model-config.json").read_text())
    plan = StagePlan(Geometry.from_config(config), (18, 5, 17))
    mapping, pins, values = {}, {}, {}
    for part, layer in enumerate([0, 18, 23], 1):
        filename = f"model-{part:05}-of-00003.safetensors"
        name = f"layers.{layer}.ffn.experts.0.w1.suh"
        tensors = {name: np.arange(128, dtype=np.float16)}
        if layer == 0:
            tensors["embed.weight"] = np.arange(256, dtype=np.float16).reshape(16, 16)
        if layer == 18:
            # An unowned tensor in the same physical shard must remain unreadable.
            tensors["layers.23.ffn.experts.1.w1.suh"] = np.full(16, 7, dtype=np.float16)
        save_file(tensors, tmp_path / filename)
        payload = (tmp_path / filename).read_bytes()
        header_size, = struct.unpack("<Q", payload[:8])
        pins[filename] = ShardPin(filename, "a" * 40, len(payload), hashlib.sha256(payload).hexdigest(),
                                  header_size, hashlib.sha256(payload[8:8 + header_size]).hexdigest())
        mapping.update({key: filename for key in tensors})
        values.update(tensors)
    index = tmp_path / "model.safetensors.index.json"
    index.write_text(json.dumps({"weight_map": mapping}))
    digest = hashlib.sha256(index.read_bytes()).hexdigest()
    return tmp_path, index, digest, pins, plan, values


def source(checkpoint, role=Role.MIDDLE):
    root, index, digest, pins, plan, _ = checkpoint
    return SelectiveWeights(root, index, index_sha256=digest, pins=pins, plan=plan, role=role)


def test_only_owned_shard_is_opened_and_payload_range_is_bounded(checkpoint):
    root, _, _, pins, _, values = checkpoint
    for filename in ("model-00001-of-00003.safetensors", "model-00003-of-00003.safetensors"):
        (root / filename).unlink()
    name = "layers.18.ffn.experts.0.w1.suh"
    with source(checkpoint) as weights:
        assert weights.names == {name}
        assert weights.verified_shards == weights.payload_bytes == 0
        assert b"".join(weights.chunks(name, 31)) == values[name].tobytes()
        assert weights.verified_shards == 1 and weights.payload_bytes == 256
        assert weights.integrity_bytes == pins["model-00002-of-00003.safetensors"].file_bytes
        with pytest.raises(ValidationError, match="not owned"):
            weights.read_range("layers.23.ffn.experts.1.w1.suh", 0, 2)
        with pytest.raises(ValidationError):
            weights.read_range(name, 255, 2)
    with pytest.raises(ValidationError, match="permanently closed"):
        weights.read_range(name, 0, 2)


def test_publisher_hash_detects_corrupt_payload_with_unchanged_header(checkpoint):
    root, _, _, _, _, _ = checkpoint
    path = root / "model-00002-of-00003.safetensors"
    data = bytearray(path.read_bytes())
    data[-1] ^= 1
    path.write_bytes(data)
    with source(checkpoint) as weights:
        with pytest.raises(ValidationError, match="publisher SHA-256"):
            weights.info(next(iter(weights.names)))
        assert weights.verified_shards == 0


def test_only_explicit_stage_c_dspark_replica_can_read_the_pinned_text_embedding(checkpoint):
    root, index, digest, pins, plan, values = checkpoint
    with source(checkpoint, Role.TERMINAL) as ordinary:
        with pytest.raises(ValidationError, match="not owned"):
            ordinary.read_range("embed.weight", 0, 2)
    with SelectiveWeights(root, index, index_sha256=digest, pins=pins, plan=plan,
                          role=Role.TERMINAL, replicate_draft_embedding=True) as replica:
        assert replica.replicated_names == {"embed.weight"}
        assert replica.read_range("embed.weight", 0, 512) == values["embed.weight"].tobytes()
        with pytest.raises(ValidationError, match="not owned"):
            replica.read_range("layers.0.ffn.experts.0.w1.suh", 0, 2)
    with pytest.raises(ValidationError, match="only stage C"):
        SelectiveWeights(root, index, index_sha256=digest, pins=pins, plan=plan,
                         role=Role.MIDDLE, replicate_draft_embedding=True)


def test_index_fingerprint_and_unsafe_shard_path_are_rejected(checkpoint):
    root, index, digest, pins, plan, _ = checkpoint
    with pytest.raises(ValidationError, match="index bound or fingerprint"):
        SelectiveWeights(root, index, index_sha256="0" * 64, pins=pins, plan=plan, role=Role.MIDDLE)
    with pytest.raises(ValidationError, match="plain checkpoint"):
        ShardPin("../model-00001-of-00003.safetensors", "a" * 40, 20, "0" * 64, 2, "0" * 64)


def test_model_file_mutation_after_verification_is_rejected(checkpoint):
    root = checkpoint[0]
    path = root / "model-00002-of-00003.safetensors"
    name = "layers.18.ffn.experts.0.w1.suh"
    with source(checkpoint) as weights:
        weights.info(name)
        if os.name == "nt":
            # Windows rejects the edit before any admitted payload can change.
            with pytest.raises(PermissionError):
                path.open("ab")
            assert weights.read_range(name, 0, 2) == checkpoint[-1][name].tobytes()[:2]
        else:
            # Portable development readers retain the existing stat guard.
            with path.open("ab") as stream:
                stream.write(b"x")
            with pytest.raises(ValidationError, match="changed after admission"):
                weights.read_range(name, 0, 2)


@pytest.mark.skipif(os.name != "nt", reason="requires actual Windows file-sharing enforcement")
def test_active_verified_shard_allows_readers_but_denies_writes_and_replacement(checkpoint):
    path = checkpoint[0] / "model-00002-of-00003.safetensors"
    name = "layers.18.ffn.experts.0.w1.suh"
    original = path.read_bytes()
    replacement = path.with_suffix(".replacement")
    replacement.write_bytes(original)
    with source(checkpoint) as weights:
        weights.info(name)
        assert path.read_bytes() == original
        with pytest.raises(PermissionError):
            path.open("r+b")
        with pytest.raises(PermissionError):
            path.unlink()
        with pytest.raises(PermissionError):
            os.replace(replacement, path)
        assert weights.read_range(name, 0, 256) == checkpoint[-1][name].tobytes()
    # Ownership ends at close; a later installation can replace the file.
    os.replace(replacement, path)
    with path.open("r+b") as writer:
        writer.write(original)


@pytest.mark.skipif(os.name != "nt", reason="requires actual Windows file-sharing enforcement")
def test_tensor_admission_refuses_an_existing_writer_and_releases_failed_handles(tmp_path):
    directory = tmp_path / "model \u0394"
    directory.mkdir()
    path = directory / "source.safetensors"
    save_file({"value": np.arange(8, dtype=np.int64)}, path)
    with path.open("r+b"):
        with pytest.raises(PermissionError):
            TensorFile(path)
    with TensorFile(path) as tensor:
        assert tensor.read_range("value", 0, 8) == np.arange(8, dtype=np.int64).tobytes()[:8]
    path.write_bytes(b"bad")
    with pytest.raises(ValidationError, match="truncated"):
        TensorFile(path)
    # A constructor failure must release its deny-write/delete read handle.
    path.unlink()


@pytest.mark.skipif(os.name != "nt", reason="requires real Win32 locked pages")
def test_middle_locked_preload_closes_disk_and_runs_without_files(checkpoint):
    root, _, _, _, _, values = checkpoint
    boot = source(checkpoint)
    name = "layers.18.ffn.experts.0.w1.suh"
    with LockedOwnedWeights(boot, admitted_bytes=4096) as resident:
        assert boot.closed and not boot._files
        assert resident.payload_bytes == 256 and resident.allocation_bytes == 4096
        for path in root.glob("*.safetensors"):
            path.unlink()
        assert resident.read_range(name, 0, 256) == values[name].tobytes()
        with pytest.raises(ValidationError, match="permanently closed"):
            boot.read_range(name, 0, 2)
    with pytest.raises(ValidationError):
        resident.read_range(name, 0, 1)


def test_failed_middle_admission_closes_boot_source(checkpoint):
    boot = source(checkpoint)
    with pytest.raises(ValidationError, match="exceed admitted"):
        LockedOwnedWeights(boot, admitted_bytes=64)
    assert boot.closed
