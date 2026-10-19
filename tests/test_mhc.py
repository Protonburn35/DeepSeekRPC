import numpy as np
import pytest


@pytest.mark.reference
@pytest.mark.parametrize("broadcast", [False, True])
def test_delayed_mhc_coefficients_match_independent_float64_equations(broadcast):
    torch = pytest.importorskip("torch")
    from deepseekrpc.vllm_mhc import _coefficients
    rng = np.random.default_rng(1017)
    residual = rng.normal(size=(7, 4, 128)).astype(np.float32)
    source = residual[:, 0].copy() if broadcast else residual.reshape(7, -1)
    fn = rng.normal(0, .02, (24, source.shape[1])).astype(np.float32)
    scale = np.array([.7, .4, .2], np.float32)
    base = rng.normal(0, .2, 24).astype(np.float32)
    value = source.astype(np.float64)
    mix = (value @ fn.astype(np.float64).T) / np.sqrt(np.mean(value ** 2, axis=1, keepdims=True) + 1e-20)
    expected_pre = 1 / (1 + np.exp(-(mix[:, :4] * scale[0] + base[:4]))) + 1e-6
    expected_post = 2 / (1 + np.exp(-(mix[:, 4:8] * scale[1] + base[4:8])))
    logits = mix[:, 8:].reshape(-1, 4, 4) * scale[2] + base[8:].reshape(1, 4, 4)
    exp = np.exp(logits - logits.max(axis=-1, keepdims=True))
    expected_comb = exp / exp.sum(axis=-1, keepdims=True) + 1e-6
    expected_comb /= expected_comb.sum(axis=-2, keepdims=True) + 1e-6
    for _ in range(19):
        expected_comb /= expected_comb.sum(axis=-1, keepdims=True) + 1e-6
        expected_comb /= expected_comb.sum(axis=-2, keepdims=True) + 1e-6
    post, comb, pre = _coefficients(torch.from_numpy(residual), torch.from_numpy(fn), torch.from_numpy(scale),
                                    torch.from_numpy(base), 1e-20, 1e-6, 1e-6, 2., 20,
                                    torch.from_numpy(source) if broadcast else None)
    for actual, expected in ((post.squeeze(-1), expected_post), (pre, expected_pre), (comb, expected_comb)):
        np.testing.assert_allclose(actual.numpy(), expected, rtol=2e-6, atol=2e-7)
