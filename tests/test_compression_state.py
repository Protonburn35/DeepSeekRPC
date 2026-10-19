import pytest

from deepseekrpc.compression_state import group_boundaries
from deepseekrpc.jsonutil import ValidationError


@pytest.mark.parametrize("start,rows,ratio,expected", [
    (0, 1, 2, ((), (), ())),
    (1, 1, 2, ((0,), (0,), (1,))),
    (2, 3, 2, ((1,), (1,), (3,))),
    (3, 4, 2, ((0, 2), (1, 2), (3, 5))),
    (1048573, 3, 1, ((0, 1, 2), (1048573, 1048574, 1048575), (1048573, 1048574, 1048575))),
])
def test_pair_closure_retains_absolute_producing_token_across_chunk_boundaries(start, rows, ratio, expected):
    assert group_boundaries(start, rows, ratio, context=1048576) == expected


def test_compressor_refuses_context_overflow_and_unbounded_chunks():
    for start, rows in ((1048575, 2), (0, 65)):
        with pytest.raises(ValidationError):
            group_boundaries(start, rows, 2, context=1048576)
