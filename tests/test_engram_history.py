import gzip
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from deepseekrpc.engram_history import EngramHistory
from deepseekrpc.jsonutil import ValidationError
from deepseekrpc.tokenizer import TokenizerAdapter

ROOT = Path(__file__).parents[1]
pytestmark = pytest.mark.reference


@pytest.fixture(scope="module")
def tokenizer(tmp_path_factory):
    pytest.importorskip("torch")
    pytest.importorskip("tokenizers")
    path = tmp_path_factory.mktemp("tokenizer") / "tokenizer.json"
    path.write_bytes(gzip.decompress((ROOT / "tests/fixtures/tokenizer.json.gz").read_bytes()))
    lock = json.loads((ROOT / "provenance/token-map.lock.json").read_text())
    return TokenizerAdapter(path, expected_sha256=lock["tokenizer_sha256"])


@pytest.fixture(scope="module")
def fixture():
    return json.loads((ROOT / "tests/fixtures/engram-hashes.json").read_text())


def history(tokenizer, *, context=64):
    config = json.loads((ROOT / "provenance/model-config.json").read_text())
    lock = json.loads((ROOT / "provenance/token-map.lock.json").read_text())
    return EngramHistory(config, tokenizer, expected_map_sha256=lock["canonical_lookup_sha256"], context=context)


def test_real_tokenizer_keeps_special_ids_and_no_added_bos(tokenizer):
    assert tokenizer.encode("<｜begin▁of▁sentence｜>") == [0]
    assert tokenizer.encode("<｜end▁of▁sentence｜>") == [1]
    assert tokenizer.encode("<｜deepseek_image｜>") == [129264]
    text = "The cat and the DOG."
    assert tokenizer.decode(tokenizer.encode(text)) == text


def test_frozen_official_hashes_across_odd_chunks_and_images(tokenizer, fixture):
    state = history(tokenizer)
    expected = np.asarray(fixture["hashes"], dtype=np.int64)
    assert hashlib.sha256(expected.astype("<i8").tobytes()).hexdigest() == fixture["hashes_int64_le_sha256"]
    output, start = [], 0
    for rows in (1, 2, 4, 3, 5):
        hashes = state.begin(fixture["input_ids"][start:start + rows], fixture["text_mask"][start:start + rows])
        assert not hashes.flags.writeable
        output.append(hashes)
        state.commit(rows)
        start += rows
    np.testing.assert_array_equal(np.concatenate(output), expected)


def test_hash_rejection_suffix_and_cancel_do_not_contaminate_next_request(tokenizer, fixture):
    state = history(tokenizer)
    cache_identity = id(state._state.cache)
    state.begin(fixture["input_ids"][:7], fixture["text_mask"][:7])
    state.commit(4)
    rest = state.begin(fixture["input_ids"][4:], fixture["text_mask"][4:])
    np.testing.assert_array_equal(rest, np.asarray(fixture["hashes"], dtype=np.int64)[4:])
    state.abort()
    assert state.position == 4
    state.reset()
    actual = state.begin(fixture["input_ids"], fixture["text_mask"])
    np.testing.assert_array_equal(actual, fixture["hashes"])
    assert id(state._state.cache) == cache_identity
    state.commit(len(fixture["input_ids"]))


def test_hash_capacity_and_invalid_input_refuse_without_truncation(tokenizer, fixture):
    state = history(tokenizer)
    for bad in ([-1], [len(tokenizer)], [True], [1.5]):
        with pytest.raises(ValidationError):
            state.begin(bad)
    assert state.position == 0 and state._pending is None
    for _ in range(4):
        state.begin(fixture["input_ids"], fixture["text_mask"])
        state.commit(15)
    assert state.position == 60
    with pytest.raises(ValidationError, match="no history was truncated"):
        state.begin(fixture["input_ids"][:5])
    assert state.position == 60


def test_full_capacity_hash_metadata_is_eager_not_full_model_validation(tokenizer):
    state = history(tokenizer, context=1048576)
    assert state.persistent_history_bytes == 1048576 * 8
    assert state.position == 0
