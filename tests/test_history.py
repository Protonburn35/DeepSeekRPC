from dataclasses import replace

import numpy as np
import pytest

from deepseekrpc.history import ImageSpan, TokenHistory
from deepseekrpc.jsonutil import ValidationError


def test_history_hash_is_chunk_independent_and_counts_only_published_outputs():
    first, second = TokenHistory("a" * 64, capacity=32), TokenHistory("a" * 64, capacity=32)
    initial = first.set_prompt([0, 3, 4, 3])
    assert second.set_prompt([0, 3, 4, 3, 7, 8, 7]) != initial
    # A whole prompt and prompt + separately published text have identical
    # token/image identity, while output penalties retain request boundaries.
    first.append([7, 8], expected_digest=first.preview_append([7, 8]))
    first.append([7], expected_digest=first.preview_append([7]))
    assert first.digest == second.digest
    assert first.prefix_digest(4) == initial
    assert first.counts()[3] == 2 and first.counts()[7] == 2
    assert first.generated_counts()[3] == 0 and first.generated_counts()[7] == 2
    assert second.generated_counts().sum() == 0
    assert first.rows(4, 3) == ((7, 8, 7), (True, True, True))


def test_wrong_append_digest_and_bad_prompt_are_atomic():
    history = TokenHistory("a" * 64, capacity=8)
    before = history.set_prompt([1, 3, 3])
    counts = history.counts()
    with pytest.raises(ValidationError, match="all-stage history"):
        history.append([4], expected_digest="0" * 64)
    with pytest.raises(ValidationError, match="image spans"):
        history.set_prompt([1, 3], [True, False])
    assert history.length == 3 and history.digest == before
    np.testing.assert_array_equal(history.counts(), counts)
    assert history.generated_counts().sum() == 0


def test_image_identity_binds_bytes_processor_and_span_without_paths():
    tokens = [0, 129264, 129264, 129264, 7, 8]
    mask = [True, False, False, False, True, True]
    image = ImageSpan(1, 3, "b" * 64)
    history = TokenHistory("a" * 64, capacity=32)
    digest = history.set_prompt(tokens, mask, [image])
    with pytest.raises(ValidationError, match="inside"):
        history.prefix_digest(2)
    prefix = history.prefix_digest(4)
    exact = TokenHistory("a" * 64, capacity=32)
    assert exact.set_prompt(tokens[:4], mask[:4], [image]) == prefix
    changed = TokenHistory("a" * 64, capacity=32)
    assert changed.set_prompt(tokens, mask, [replace(image, key="c" * 64)]) != digest
    assert b"path" not in history.image_manifest()


@pytest.mark.parametrize("tokens", ([True], [-1], [129280], [1.0]))
def test_invalid_token_types_and_ranges_cannot_enter_history(tokens):
    history = TokenHistory("a" * 64, capacity=8)
    with pytest.raises(ValidationError, match="integer token"):
        history.set_prompt(tokens)


def test_prefix_identity_changes_with_model_tokenizer_or_precision_contract():
    first, second = TokenHistory("a" * 64, capacity=8), TokenHistory("b" * 64, capacity=8)
    assert first.set_prompt([1, 2]) != second.set_prompt([1, 2])
