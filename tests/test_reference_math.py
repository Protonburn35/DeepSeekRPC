import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from deepseekrpc.reference_math import expert, hc_post, hc_pre, mul1_codebook, route


def oracle_module(name):
    path = Path(__file__).parent / "oracles" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def oracle():
    torch = pytest.importorskip("torch", reason="install requirements-reference-linux.lock with the CPU index")
    torch.set_num_threads(2)
    return oracle_module("official_methods")


@pytest.mark.reference
def test_reference_router_unbiased_weights_and_image_bias(oracle):
    torch = oracle.torch
    rng = np.random.default_rng(41017)
    x = rng.normal(size=(7, 5120)).astype(np.float32) / 10
    weight = rng.normal(size=(384, 5120)).astype(np.float32) / 10
    bias = rng.normal(size=384).astype(np.float32) / 4
    image_bias = rng.normal(size=384).astype(np.float32) / 4
    mask = np.array([False, True, False, True, True, False, False])
    model = SimpleNamespace(weight=torch.from_numpy(weight), bias=torch.from_numpy(bias),
                            bias_vl=torch.from_numpy(image_bias), gate_temp=1.0,
                            score_func="sqrtsoftplus", topk=6, norm_topk_prob=True, route_scale=1.5)
    expected_weights, expected_ids = oracle.Gate.forward(model, torch.from_numpy(x), torch.from_numpy(mask))
    weights, ids = route(x, weight, bias, image_mask=mask, image_bias=image_bias)
    np.testing.assert_array_equal(ids, expected_ids.numpy())
    np.testing.assert_allclose(weights, expected_weights.numpy(), rtol=2e-6, atol=1e-6)
    np.testing.assert_allclose(weights.sum(axis=1), 1.5, rtol=1e-6)


@pytest.mark.reference
@pytest.mark.parametrize("rows", [1, 7])
def test_reference_expert_actual_5120_2304_shape(oracle, rows):
    torch = oracle.torch
    rng = np.random.default_rng(1017)
    x = rng.normal(size=(rows, 5120)).astype(np.float32) / 10
    w1 = rng.normal(size=(2304, 5120)).astype(np.float32) / 20
    w3 = rng.normal(size=(2304, 5120)).astype(np.float32) / 20
    w2 = rng.normal(size=(5120, 2304)).astype(np.float32) / 20
    weights = rng.uniform(size=(rows, 1)).astype(np.float32)
    model = SimpleNamespace(swiglu_limit=10, w1=lambda v: torch.nn.functional.linear(v, torch.from_numpy(w1)),
                            w2=lambda v: torch.nn.functional.linear(v, torch.from_numpy(w2)),
                            w3=lambda v: torch.nn.functional.linear(v, torch.from_numpy(w3)))
    expected = oracle.Expert.forward(model, torch.from_numpy(x), torch.from_numpy(weights)).numpy()
    actual = expert(x, w1, w2, w3, routing_weight=weights)
    # Unquantized FP32 operation baseline only; no EXL3 numerical claim.
    np.testing.assert_allclose(actual, expected, rtol=1e-4, atol=2e-6)


@pytest.mark.reference
def test_reference_swiglu_asymmetric_clamp(oracle):
    torch = oracle.torch
    x = np.array([[20, -20, 2, -2]], dtype=np.float32)
    identity = np.eye(4, dtype=np.float32)
    model = SimpleNamespace(swiglu_limit=10, w1=lambda v: v, w2=lambda v: v, w3=lambda v: v)
    expected = oracle.Expert.forward(model, torch.from_numpy(x)).numpy()
    np.testing.assert_allclose(expert(x, identity, identity, identity), expected, rtol=2e-6, atol=1e-6)


@pytest.mark.reference
def test_reference_mhc_four_stream_axis_and_premix(oracle):
    torch = oracle.torch
    rng = np.random.default_rng(41)
    residual = rng.normal(size=(1, 7, 4, 5120)).astype(np.float32)
    pre = rng.normal(size=(1, 7, 4)).astype(np.float32)
    post = rng.normal(size=(1, 7, 4)).astype(np.float32)
    comb = rng.normal(size=(1, 7, 4, 4)).astype(np.float32)
    collapsed = oracle.Block.hc_pre(None, torch.from_numpy(residual), torch.from_numpy(pre)).numpy()
    np.testing.assert_allclose(hc_pre(residual, pre), collapsed, rtol=2e-6, atol=1e-6)
    expanded = oracle.Block.hc_post(None, torch.from_numpy(collapsed), torch.from_numpy(residual),
                                   torch.from_numpy(post), torch.from_numpy(comb)).numpy()
    np.testing.assert_allclose(hc_post(collapsed, residual, post, comb), expanded, rtol=2e-6, atol=2e-6)


def test_every_mul1_procedural_codebook_window():
    oracle = oracle_module("exl3_dequant")
    windows = np.arange(65536, dtype=np.uint16)
    expected = np.array([oracle.decode_mul1_py(int(v)) for v in windows], dtype=np.float16)
    actual = mul1_codebook(windows)
    np.testing.assert_array_equal(actual.view(np.uint16), expected.view(np.uint16))
