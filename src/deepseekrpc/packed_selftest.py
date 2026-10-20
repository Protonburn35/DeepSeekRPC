"""Real EXL3 CUDA self-tests against independent CPU synthetic-format equations."""
import numpy as np

from .inventory import TensorInfo
from .packed import DevicePackedLinear, MUL1_MARKER, PackedLinearLayout, PackedPart
from .reference_packed import hadamard_reference, projection_reference, reconstruct_reference
from .runtime_policy import assert_exact_exl3


def run_packed_checks(device):
    import torch
    import exllamav3_ext
    from vllm_exl3.dequant import decode_codebook_bits_py
    assert_exact_exl3()
    # Independently implemented Python IEEE-754/codebook reference from the
    # pinned plugin. The actual CUDA decoder/GEMM comes from ExLlama.
    table = np.array([decode_codebook_bits_py(index, 2) for index in range(65536)], dtype=np.uint16).view(np.float16)
    rng, checks = np.random.default_rng(41017), []
    for k in (2, 3, 4, 5, 6, 8):
        # Both 5120/2304 directions occur in the actual expert graph. The K5
        # 4096/1024 shape separately covers an independent wo_a rotation slice.
        shapes = [(5120, 2304), (2304, 5120)] + ([(4096, 1024)] if k == 5 else [])
        for inputs, outputs in shapes:
            prefix = f"selftest.k{k}.{inputs}.{outputs}"
            packed = rng.integers(0, 65536, (inputs // 16, outputs // 16, 16 * k), dtype=np.uint16)
            suh = (rng.choice([-1., 1.], inputs) / 8).astype(np.float16)
            svh = (rng.choice([-1., 1.], outputs) / 16).astype(np.float16)
            arrays = {"trellis": packed.view(np.int16), "suh": suh, "svh": svh,
                      "mul1": np.array(MUL1_MARKER, dtype=np.int32)}
            parts, offset = [], 0
            for suffix, dtype in (("trellis", "I16"), ("suh", "F16"), ("svh", "F16"), ("mul1", "I32")):
                offset = (offset + 15) // 16 * 16
                array = arrays[suffix]
                info = TensorInfo(f"{prefix}.{suffix}", dtype, array.shape, 0, array.nbytes)
                parts.append(PackedPart(info, offset))
                offset += array.nbytes
            size = (offset + 255) // 256 * 256
            layout = PackedLinearLayout(prefix, inputs, outputs, k, tuple(parts), size)
            storage = torch.zeros(size, dtype=torch.uint8, device=device)
            for part in parts:
                data = arrays[part.info.name.rsplit(".", 1)[1]].tobytes()
                storage[part.offset:part.offset + len(data)].copy_(torch.frombuffer(bytearray(data), dtype=torch.uint8))
            linear = DevicePackedLinear(layout, storage)
            try:
                expected_weights = reconstruct_reference(packed, k, table)
                actual_weights = torch.empty((inputs, outputs), dtype=torch.float16, device=device)
                exllamav3_ext.reconstruct(actual_weights, linear.inner.trellis, k, False, True)
                np.testing.assert_array_equal(actual_weights.view(torch.uint16).cpu().numpy(), expected_weights.view(np.uint16))
                del actual_weights
                for rows in (1, 7, 64):
                    x = (rng.integers(-128, 129, (rows, inputs)).astype(np.float32) / 32768).astype(np.float16)
                    expected, bound = projection_reference(x, expected_weights, suh, svh)
                    actual = linear.forward(torch.tensor(x, dtype=torch.float16, device=device)).cpu().numpy()
                    error = abs(actual.astype(np.float64) - expected)
                    if not np.isfinite(actual).all() or np.any(error > bound):
                        raise AssertionError("actual EXL3 projection exceeds its a-priori CPU rotation/dot error envelope")
                    checks.append({"k": k, "inputs": inputs, "outputs": outputs, "rows": rows,
                        "rotated_weight_bits": "exact", "maximum_error": float(error.max()),
                        "maximum_bound": float(bound.max())})
            finally:
                torch.cuda.synchronize(device)
                linear.close()
            del storage, expected_weights, packed
    # Exercise native Hadamard separately with a stricter transform-only bound.
    x = rng.normal(0, .02, (7, 5120)).astype(np.float32)
    source = torch.tensor(x, dtype=torch.float32, device=device)
    target = torch.empty_like(source)
    exllamav3_ext.had_r_128(source, target, None, None, 1.)
    expected = hadamard_reference(x)
    gamma = (9 * 2. ** -24) / (1 - 9 * 2. ** -24)
    bound = gamma * np.repeat(abs(x).reshape(7, -1, 128).sum(-1) / np.sqrt(128.), 128, axis=-1) + 1e-7
    if np.any(abs(target.cpu().numpy() - expected) > bound):
        raise AssertionError("actual EXL3 FP32 rotation differs from independent normalized Sylvester-128 equations")
    return checks
