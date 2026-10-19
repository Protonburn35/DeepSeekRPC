import json
import hashlib
import struct

import numpy as np
import pytest

from deepseekrpc.engram_storage import EngramRows, decode_fp8_rows
from deepseekrpc.jsonutil import ValidationError


def table(tmp_path):
    weights = np.arange(16 * 256, dtype=np.uint8).reshape(16, 256) % 127
    scales = np.full((16, 8), 127, dtype=np.uint8)
    raw = weights.tobytes() + scales.tobytes()
    metadata = {"layers.1.engram.embed.weight": {"dtype": "F8_E4M3", "shape": [16, 256],
                                                "data_offsets": [0, weights.nbytes]},
                "layers.1.engram.embed.scale": {"dtype": "F8_E8M0", "shape": [16, 8],
                                               "data_offsets": [weights.nbytes, len(raw)]}}
    header = json.dumps(metadata).encode()
    path = tmp_path / "synthetic_engram.safetensors"
    path.write_bytes(struct.pack("<Q", len(header)) + header + raw)
    return path, weights, scales


def test_exact_encoded_rows_and_bounded_cache(tmp_path):
    path, weights, scales = table(tmp_path)
    with EngramRows(path, layer=1, cache_bytes=1040) as store:
        ids = np.array([3, 8, 3], dtype=np.int64)
        values = store.gather_encoded(ids)
        np.testing.assert_array_equal(values[:, :256], weights[ids])
        np.testing.assert_array_equal(values[:, 256:], scales[ids])
        assert store.disk_row_reads == 2 and store.cache_hits == 1
        store.gather_encoded(np.array([7, 8, 9, 10], dtype=np.int32))
        assert len(store._cache) <= 2
        with pytest.raises(ValidationError):
            store.gather_encoded(np.array([16], dtype=np.int64))
    with pytest.raises(ValidationError):
        store.gather_encoded(np.array([3], dtype=np.int64))


def test_no_cache_preserves_every_exact_row(tmp_path):
    path, weights, scales = table(tmp_path)
    with EngramRows(path, layer=1, cache_bytes=0) as store:
        ids = np.array([3, 8, 3], dtype=np.int64)
        actual = store.gather_encoded(ids)
        np.testing.assert_array_equal(actual, np.concatenate((weights[ids], scales[ids]), axis=1))
        assert store.disk_row_reads == 3 and not store._cache


@pytest.mark.reference
def test_all_finite_fp8_codes_against_pytorch_cpu():
    torch = pytest.importorskip("torch")
    codes = np.arange(256, dtype=np.uint8)
    finite = codes[(codes & 127) != 127]
    for exponent in (0, 64, 126, 127, 130):
        encoded = np.zeros((1, 264), dtype=np.uint8)
        encoded[0, :len(finite)] = finite
        encoded[0, 256:] = exponent
        actual = decode_fp8_rows(encoded)
        weights = torch.from_numpy(encoded[:, :256].copy()).view(torch.float8_e4m3fn).float()
        scales = torch.from_numpy(encoded[:, 256:].copy()).view(torch.float8_e8m0fnu).float()
        expected = (weights.reshape(1, 8, 32) * scales[..., None]).reshape(1, 256).numpy()
        np.testing.assert_array_equal(actual.view(np.uint32), expected.view(np.uint32))


@pytest.mark.parametrize("bad", [127, 255])
def test_nonfinite_fp8_rows_rejected(bad):
    encoded = np.zeros((1, 264), dtype=np.uint8)
    encoded[0, 0] = bad
    encoded[0, 256:] = 127
    with pytest.raises(ValidationError):
        decode_fp8_rows(encoded)


def test_zero_ue8m0_byte_keeps_official_nonzero_scale():
    # Official ParallelEngramEmbedding uses standard UE8M0 .float(), whereas
    # the upstream vLLM integer-exponent shortcut maps this encoding to zero.
    encoded = np.zeros((1, 264), dtype=np.uint8)
    encoded[:, :256] = 0x38  # exact FP8 +1
    expected = np.full((1, 256), np.float32(2.0 ** -127), dtype=np.float32)
    actual = decode_fp8_rows(encoded)
    np.testing.assert_array_equal(actual.view(np.uint32), expected.view(np.uint32))
    assert np.all(actual > 0)


def test_production_engram_admission_requires_the_pinned_complete_publisher_file(tmp_path):
    from deepseekrpc.weights import ShardPin
    from dataclasses import replace
    path, _, _ = table(tmp_path)
    admitted = tmp_path / "model-00047-of-00048.safetensors"
    path.rename(admitted)
    raw = admitted.read_bytes()
    header_bytes, = struct.unpack("<Q", raw[:8])
    pin = ShardPin(admitted.name, "a" * 40, len(raw), hashlib.sha256(raw).hexdigest(),
                   header_bytes, hashlib.sha256(raw[8:8 + header_bytes]).hexdigest())
    with EngramRows(admitted, layer=1, cache_bytes=0, pin=pin) as rows:
        assert rows.verified and rows.layer == 1
    with pytest.raises(ValidationError, match="publisher SHA"):
        EngramRows(admitted, layer=1, cache_bytes=0, pin=replace(pin, file_sha256="b" * 64))
