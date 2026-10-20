"""Small actual CUDA mHC/router diagnostics with independent CPU equations."""
import numpy as np


def run_graph_checks(device):
    import torch
    from vllm.model_executor.layers.fused_moe.router.fused_topk_bias_router import fused_topk_bias
    from .vllm_mhc import mhc_pre_delayed, mhc_post
    rng, results = np.random.default_rng(41017), []

    def tensor(array, dtype=torch.float32):
        return torch.tensor(array, dtype=dtype, device=device)

    residual = rng.integers(-64, 65, (7, 4, 5120)).astype(np.float32) / 128
    source = tensor(residual, torch.bfloat16)
    scale = np.array([.7, .4, .2], np.float32)
    base = rng.normal(0, .2, 24).astype(np.float32)
    for broadcast in (False, True):
        value = residual[:, 0] if broadcast else residual.reshape(7, -1)
        fn = rng.integers(-16, 17, (24, value.shape[1])).astype(np.float32) / 8192
        pre_mix = rng.integers(1, 5, (7, 4)).astype(np.float32) / 8
        observed = mhc_pre_delayed(source, tensor(fn), tensor(scale), tensor(base),
            1e-20, 1e-6, 1e-6, 2., 20, pre_mix=tensor(pre_mix),
            x=source[:, 0].contiguous() if broadcast else None)
        # FP64 normalized projections, sigmoid and alternating Sinkhorn sums.
        value = value.astype(np.float64)
        mix = (value @ fn.astype(np.float64).T) / np.sqrt(np.mean(value ** 2, axis=1, keepdims=True) + 1e-20)
        pre = 1 / (1 + np.exp(-(mix[:, :4] * scale[0] + base[:4]))) + 1e-6
        post = 2 / (1 + np.exp(-(mix[:, 4:8] * scale[1] + base[4:8])))
        logits = mix[:, 8:].reshape(-1, 4, 4) * scale[2] + base[8:].reshape(1, 4, 4)
        comb = np.exp(logits - logits.max(axis=-1, keepdims=True))
        comb = comb / comb.sum(axis=-1, keepdims=True) + 1e-6
        comb /= comb.sum(axis=-2, keepdims=True) + 1e-6
        for _ in range(19):
            comb /= comb.sum(axis=-1, keepdims=True) + 1e-6
            comb /= comb.sum(axis=-2, keepdims=True) + 1e-6
        for actual, expected in ((observed[0].squeeze(-1), post), (observed[1], comb), (observed[3], pre)):
            np.testing.assert_allclose(actual.cpu().numpy(), expected, rtol=6e-6, atol=4e-7)
        # Dyadic pre_mix makes this four-stream collapse exact before the one
        # prescribed BF16 rounding, independent of coefficient tolerances.
        collapsed = (residual.astype(np.float64) * pre_mix[..., None]).sum(axis=1)
        rounded = torch.tensor(collapsed, dtype=torch.bfloat16)
        torch.testing.assert_close(observed[2].cpu().view(torch.int16), rounded.view(torch.int16), rtol=0, atol=0)
        x = tensor(rng.normal(0, .2, (7, 5120)).astype(np.float32))
        actual = mhc_post(x, source, observed[0], observed[1]).float().cpu().numpy()
        # Isolate actual BF16 FFN rounding from the independently checked
        # coefficients. Compare expansion in FP64 and allow one BF16 ULP.
        rounded_x = x.to(torch.bfloat16).float().cpu().numpy().astype(np.float64)
        expansion = (np.einsum("rij,rid->rjd", observed[1].cpu().numpy().astype(np.float64), residual.astype(np.float64))
                     + observed[0].cpu().numpy().astype(np.float64) * rounded_x[:, None, :])
        expected_bf16 = torch.tensor(expansion, dtype=torch.bfloat16).float().numpy()
        # A cancellation close to zero has an FP32 absolute-error allowance;
        # BF16 output spacing accounts for its explicit final rounding.
        exponent = np.frexp(abs(expected_bf16).astype(np.float64))[1]
        tolerance = np.maximum(np.ldexp(np.ones_like(expansion), exponent - 8), 2e-7)
        if np.any(abs(actual.astype(np.float64) - expansion) > tolerance + abs(expected_bf16 - expansion)):
            raise AssertionError("actual four-stream mHC expansion exceeds FP32/BF16 rounding envelope")
        results.append({"name": "actual mHC width 5120, " + ("broadcast" if broadcast else "four-stream") + " coefficients and expansion",
                        "status": "passed", "oracle": "independent FP64 equations; exact dyadic BF16 collapse"})

    # Both target and draft router geometries. The synthetic dyadic F32 dot
    # product is independently computed in FP64 before testing actual dispatch
    # to the vLLM Triton/native image-aware sqrt(softplus) top-k kernels.
    ids = np.array([37, 129263, 129264, 129265, 129268, 129269, 128799], np.int64)
    image_mask = (ids >= 129264) & (ids < 129269)
    inputs = rng.integers(-16, 17, (7, 5120)).astype(np.float32) / 64
    hidden = tensor(inputs, torch.bfloat16)
    for experts, topk in ((384, 6), (128, 3)):
        weight = rng.integers(-16, 17, (experts, 5120)).astype(np.float32) / 32768
        logits = torch.nn.functional.linear(hidden.float(), tensor(weight))
        expected_logits = inputs.astype(np.float64) @ weight.astype(np.float64).T
        np.testing.assert_allclose(logits.cpu().numpy(), expected_logits, rtol=0, atol=1e-7)
        text_bias = rng.normal(0, .2, experts).astype(np.float32)
        image_bias = rng.normal(0, .2, experts).astype(np.float32)
        weights, selected = fused_topk_bias(hidden, logits, "sqrtsoftplus", tensor(text_bias), topk, True,
            indices_type=torch.int64, input_tokens=tensor(ids, torch.int64),
            routed_scaling_factor=1.5, bias_vl=tensor(image_bias), image_sentinel_lo=129264)
        scores = np.sqrt(np.logaddexp(0., expected_logits))
        ranked = scores + np.where(image_mask[:, None], image_bias, text_bias)
        expected_ids = np.argsort(-ranked, axis=-1, kind="stable")[:, :topk]
        # Kernel output order is not a semantic expert-order guarantee.
        actual_ids, actual_weights = selected.cpu().numpy(), weights.cpu().numpy()
        np.testing.assert_array_equal(np.sort(actual_ids, axis=-1), np.sort(expected_ids, axis=-1))
        expected_weights = np.take_along_axis(scores, actual_ids, axis=-1)
        expected_weights *= 1.5 / expected_weights.sum(axis=-1, keepdims=True)
        np.testing.assert_allclose(actual_weights, expected_weights, rtol=2e-6, atol=1e-6)
        results.append({"name": f"actual {experts}-expert top-{topk} router: FP32 gate, text/image biases and sentinel boundaries",
                        "status": "passed", "oracle": "independent FP64 gate/softplus/sqrt/top-k; random nontied scores"})
    return results
