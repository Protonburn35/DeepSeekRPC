import numpy as np
import pytest

from deepseekrpc.index_selection import merge_ranked


@pytest.mark.reference
@pytest.mark.parametrize("pieces", [(17, 14, 26), (1, 8, 7, 29, 12), (57,)])
def test_streaming_global_topk_matches_independent_full_ranking_across_odd_tiles(pieces):
    torch = pytest.importorskip("torch")
    rng = np.random.default_rng(1017)
    values = rng.normal(size=57).astype(np.float32)
    values[[0, 11, 56]] = 1.75  # explicit tie, lowest logical ID first
    expected = np.lexsort((np.arange(57), -values))[:13]
    scores, ids = torch.empty(0), torch.empty(0, dtype=torch.int64)
    start = 0
    for count in pieces:
        scores, ids = merge_ranked(scores, ids, torch.from_numpy(values[start:start + count]),
                                   torch.arange(start, start + count), keep=13)
        start += count
    assert start == 57
    np.testing.assert_array_equal(ids.numpy(), expected)
    np.testing.assert_array_equal(scores.numpy(), values[expected])


@pytest.mark.reference
def test_newest_partial_candidate_block_is_retained_when_older_blocks_outscore_it():
    torch = pytest.importorskip("torch")
    # Source blocks 0,1,2 are very strong; the newest block 3 is pinned.
    scores = torch.tensor([9., 8., 7., float("inf")])
    empty_scores, empty_ids = torch.empty(0), torch.empty(0, dtype=torch.int64)
    scores, ids = merge_ranked(empty_scores, empty_ids, scores, torch.arange(4), keep=2)
    assert ids.tolist() == [3, 0]
