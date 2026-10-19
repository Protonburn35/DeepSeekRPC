import math

import numpy as np
import pytest

from deepseekrpc.jsonutil import ValidationError
from deepseekrpc.reference_attention import index_scores_reference, sparse_attention_reference


def test_sparse_sink_duplicate_and_padding_against_independent_torch_dense():
    torch = pytest.importorskip("torch")
    rng = np.random.default_rng(41)
    q = rng.normal(size=(3, 2, 512)).astype(np.float32) / 4
    window = rng.normal(size=(5, 512)).astype(np.float16)
    source = rng.normal(size=(7, 512)).astype(np.float16)
    ids = np.array([[0, 7, 7, -1], [-1, -1, -1, -1], [1, 4, 11, -1]], dtype=np.int32)
    sink = np.array([1.7, -0.9], dtype=np.float32)
    actual = sparse_attention_reference(q, window, source, ids, sink, scale=512**-0.5)
    kv = torch.tensor(np.concatenate([window, source]).astype(np.float32))
    expected = []
    for row in range(3):
        selected = kv[ids[row][ids[row] >= 0].astype(np.int64)]
        logits = torch.tensor(q[row]) @ selected.T / math.sqrt(512)
        probabilities = torch.softmax(torch.cat([logits, torch.tensor(sink)[:, None]], dim=1), dim=1)
        expected.append(probabilities[:, :-1] @ selected)
    np.testing.assert_allclose(actual, torch.stack(expected).numpy(), rtol=2e-5, atol=2e-6)
    assert np.array_equal(actual[1], np.zeros_like(actual[1]))


def test_index_rectifies_before_signed_head_weights_and_respects_visibility():
    torch = pytest.importorskip("torch")
    rng = np.random.default_rng(1017)
    q = rng.normal(size=(3, 32, 128)).astype(np.float32) / 8
    keys = rng.normal(size=(97, 128)).astype(np.float16)
    weights = rng.normal(size=(3, 32)).astype(np.float32) / 16
    ids = np.tile(np.arange(-1, 97, dtype=np.int32), (3, 1))
    visible = np.array([0, 42, 97], dtype=np.int32)
    actual = index_scores_reference(q, keys, weights, ids, visible)
    scores = torch.einsum("rhd,kd->rhk", torch.tensor(q), torch.tensor(keys.astype(np.float32)))
    scores = (scores.relu() * torch.tensor(weights)[:, :, None]).sum(1).numpy()
    for row in range(3):
        np.testing.assert_allclose(actual[row, 1:visible[row] + 1], scores[row, :visible[row]], rtol=2e-5, atol=2e-6)
        assert np.isneginf(actual[row, 0])
        assert np.all(np.isneginf(actual[row, visible[row] + 1:]))


def test_reference_refuses_invalid_slot_instead_of_substituting_a_row():
    with pytest.raises(ValidationError, match="slot"):
        sparse_attention_reference(np.ones((1, 1, 8), np.float32), np.ones((1, 8), np.float16),
                                   np.ones((1, 8), np.float16), np.array([[2]], np.int32),
                                   np.zeros(1, np.float32), scale=1.0)
