from dataclasses import replace

import numpy as np
import pytest

from deepseekrpc.cache_pool import SlotMap
from deepseekrpc.history import ImageSpan, TokenHistory
from deepseekrpc.jsonutil import ValidationError
from deepseekrpc.prefix import checkpoint_anchor, prefix_offer, select_prefix
from deepseekrpc.state import Operation


def history(length, *, images=(), token=5):
    result = TokenHistory("a" * 64)
    tokens = np.full(length, token, dtype=np.int32)
    mask = np.ones(length, dtype=np.bool_)
    for image in images:
        mask[image.start:image.start + image.length] = False
        tokens[image.start:image.start + image.length] = 129264
    result.set_prompt(tokens, mask, images)
    return result


@pytest.mark.parametrize("length,anchor", [(1, 0), (63, 0), (64, 0), (65, 64), (128, 64), (129, 128), (1048576, 1048512)])
def test_checkpoint_leaves_original_final_chunk_for_new_encoder_barrier(length, anchor):
    assert checkpoint_anchor(history(length)) == anchor


def test_checkpoint_does_not_split_an_original_image_even_after_rounding():
    prompt = history(129, images=(ImageSpan(12, 53, "b" * 64), ImageSpan(67, 62, "c" * 64)))
    assert checkpoint_anchor(prompt) == 0
    assert checkpoint_anchor(history(140, images=prompt.images)) == 0
    assert checkpoint_anchor(history(193, images=prompt.images)) == 192


def test_unanimous_prefix_compares_original_image_bytes_and_requires_exact_features():
    image = ImageSpan(8, 48, "b" * 64)
    old = history(129, images=(image,))
    offer = prefix_offer(old.identity, old, 128)
    new = history(170, images=(image,))
    selected, reason = select_prefix([offer] * 3, new, features_available=lambda _: True)
    assert selected == offer and "full target replay" in reason
    assert select_prefix([offer] * 3, new, features_available=lambda _: False)[0] is None
    changed = history(170, images=(replace(image, key="c" * 64),))
    assert select_prefix([offer] * 3, changed, features_available=lambda _: True)[0] is None
    assert select_prefix([offer, None, offer], new, features_available=lambda _: True)[0] is None
    assert select_prefix([offer] * 3, history(170, images=(image,), token=6), features_available=lambda _: True)[0] is None
    assert select_prefix([offer] * 3, history(127, images=(image,)), features_available=lambda _: True)[0] is None


def test_prefix_retention_changes_epoch_and_invalidates_every_tail_slot_with_distinct_maps():
    for offset in (0, 3, 7):
        mapping = SlotMap(8, context=16, offset=offset)
        mapping.reset("a" * 32, 0)
        op = Operation("a" * 32, "b" * 32, 0, 1, "c" * 32, "d" * 64, 0, 8)
        mapping.begin(op)
        mapping.reserve(list(range(8)), list(range(8)))
        mapping.finish(op, 8)
        slots = mapping.physical(range(4))
        mapping.retain_prefix("e" * 32, 1, 4)
        assert mapping.physical(range(4)) == slots
        assert mapping.session == "e" * 32 and mapping.epoch == 1
        assert np.count_nonzero(mapping.generations) == 4
        with pytest.raises(ValidationError, match="absent"):
            mapping.physical([4])
        with pytest.raises(ValidationError, match="stale"):
            mapping.begin(op)


def test_missing_prefix_cannot_partially_change_metadata():
    mapping = SlotMap(8, context=16, offset=3)
    mapping.reset("a" * 32, 0)
    before = mapping.logical.copy()
    with pytest.raises(ValidationError, match="missing"):
        mapping.retain_prefix("e" * 32, 1, 4)
    np.testing.assert_array_equal(mapping.logical, before)
    assert mapping.session == "a" * 32 and mapping.epoch == 0
