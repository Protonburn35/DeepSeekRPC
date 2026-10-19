import pytest

from deepseekrpc.cache_pool import SlotMap
from deepseekrpc.jsonutil import ValidationError
from deepseekrpc.state import Operation


def operation(start=0, rows=4, *, epoch=0, number=1):
    return Operation("a" * 32, "b" * 32, epoch, number, "c" * 32, "d" * 64, start, rows)


def test_shadow_local_slot_mapping_preserves_logical_identity_and_prefix_rollback():
    maps = [SlotMap(3, context=16, offset=offset) for offset in (0, 2)]
    for mapping in maps:
        mapping.reset("a" * 32, 0)
        op = operation(rows=6)
        mapping.begin(op)
        mapping.reserve([0, 1, 2], [0, 1, 2])
        mapping.reserve([3, 4, 5], [3, 4, 5])
        mapping.finish(op, 4)
        assert mapping.physical([1, 2, 3, -1]) == [(value + mapping.offset) % 3 for value in (1, 2, 3)] + [-1]
        with pytest.raises(ValidationError, match="overwritten"):
            mapping.physical([0])
        with pytest.raises(ValidationError):
            mapping.physical([4])
    assert maps[0].physical([1, 2, 3]) != maps[1].physical([1, 2, 3])


def test_zero_acceptance_restores_both_old_ring_rows_and_generations():
    mapping = SlotMap(2, context=16, offset=1)
    mapping.reset("a" * 32, 0)
    first = operation(rows=2)
    mapping.begin(first)
    mapping.reserve([0, 1], [0, 1])
    mapping.finish(first, 2)
    old_generations = mapping.generations.copy()
    second = operation(2, rows=4, number=2)
    mapping.begin(second)
    mapping.reserve([2, 3], [2, 3])
    mapping.reserve([4, 5], [4, 5])
    mapping.finish(second, 0)
    assert mapping.physical([0, 1]) == [1, 0]
    assert (mapping.generations == old_generations).all()


def test_invalid_scatter_batch_is_atomic_and_bounded_before_mutation():
    mapping = SlotMap(2, context=16, offset=0, max_mutations=2)
    mapping.reset("a" * 32, 0)
    op = operation()
    mapping.begin(op)
    with pytest.raises(ValidationError, match="repeats"):
        mapping.reserve([0, 2], [0, 1])
    assert not mapping.journal and (mapping.generations == 0).all()
    mapping.reserve([0, 1], [0, 1])
    with pytest.raises(ValidationError, match="capacity"):
        mapping.reserve([2], [2])
    assert len(mapping.journal) == 2
    mapping.finish(op, 0)


def test_new_epoch_invalidates_old_rows_without_reallocating_metadata():
    mapping = SlotMap(4, context=16, offset=2)
    arrays = id(mapping.logical), id(mapping.generations)
    mapping.reset("a" * 32, 0)
    op = operation(rows=1)
    mapping.begin(op)
    mapping.reserve([0], [0])
    mapping.finish(op, 1)
    mapping.reset("a" * 32, 1)
    assert arrays == (id(mapping.logical), id(mapping.generations))
    with pytest.raises(ValidationError):
        mapping.physical([0])
    with pytest.raises(ValidationError, match="stale"):
        mapping.begin(op)


def test_uncertain_active_window_can_only_be_discarded_into_a_newer_epoch():
    mapping = SlotMap(4, context=16, offset=2)
    mapping.reset("a" * 32, 0)
    op = operation(rows=1)
    mapping.begin(op)
    mapping.reserve([0], [0])
    arrays = id(mapping.logical), id(mapping.generations)
    with pytest.raises(ValidationError, match="quiescence"):
        mapping.reset("a" * 32, 1)
    with pytest.raises(ValidationError, match="newer epoch"):
        mapping.discard_for_reset("a" * 32, 0)
    assert mapping.active == op and len(mapping.journal) == 1
    mapping.discard_for_reset("a" * 32, 1)
    assert mapping.active is None and not mapping.journal
    assert arrays == (id(mapping.logical), id(mapping.generations))
    assert (mapping.logical == -1).all() and (mapping.generations == 0).all()
    with pytest.raises(ValidationError, match="stale"):
        mapping.begin(op)
