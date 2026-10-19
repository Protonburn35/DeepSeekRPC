import numpy as np
import pytest

from deepseekrpc.jsonutil import ValidationError
from deepseekrpc.routed_experts import partition_routes, weighted_swiglu


def test_route_partition_retains_every_top_six_assignment_including_zero_weight_routes():
    ids = [[383, 9, 0, 14, 22, 8], [14, 0, 1, 3, 8, 15]]
    groups = partition_routes(ids, experts=384, topk=6)
    actual = [(row, rank, expert) for expert, positions in groups for row, rank in positions]
    assert sorted(actual) == [(row, rank, expert) for row, choices in enumerate(ids) for rank, expert in enumerate(choices)]
    assert dict(groups)[14] == ((0, 3), (1, 0))


@pytest.mark.parametrize("ids", [[[0, 1, 2, 3, 4, 4]], [[0, 1, 2, 3, 4]], [[0, 1, 2, 3, 4, 384]], [[True, 1, 2, 3, 4, 5]]])
def test_invalid_routes_cannot_silently_drop_or_repeat_a_selected_expert(ids):
    with pytest.raises(ValidationError):
        partition_routes(ids, experts=384, topk=6)


@pytest.mark.reference
def test_weighted_clamp_matches_official_expert_order_and_pre_down_half_rounding():
    torch = pytest.importorskip("torch")
    gate = np.array([[-15.0, 15.0, 0.5, -0.2], [12.0, -12.0, 1.1, 1.5]], np.float32)
    up = np.array([[15.0, -15.0, 0.2, -0.2], [-12.0, 12.0, 0.9, 3.1]], np.float32)
    weights = np.array([[0.12345], [0.23456]], np.float32)
    from oracles.official_methods import Expert
    source = Expert()
    source.swiglu_limit = 10.0
    source.w1 = lambda x: torch.from_numpy(gate)
    source.w3 = lambda x: torch.from_numpy(up)
    source.w2 = lambda x: x
    expected = source.forward(torch.ones((2, 4), dtype=torch.float16), torch.from_numpy(weights))
    actual = weighted_swiglu(torch.from_numpy(gate), torch.from_numpy(up), torch.from_numpy(weights)).half()
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    unweighted = weighted_swiglu(torch.from_numpy(gate), torch.from_numpy(up), torch.ones((2, 1))).half()
    assert not torch.equal(actual.float(), unweighted.float() * torch.from_numpy(weights))
