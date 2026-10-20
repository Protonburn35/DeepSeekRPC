"""Independent CPU format equations for bounded synthetic EXL3 self-tests.

This is diagnostic math, never an inference fallback. The bitstream is circular,
MSB-first within each little-endian uint32 word. Each state is the 16 bits
ending after the next K-bit symbol. The 16x16 MMA fragment permutation follows
the published pinned EXL3 format; the GPU executor does not call this module.
"""
import numpy as np

from .jsonutil import ValidationError


def trellis_windows(packed, k):
    values = np.asarray(packed)
    if (type(k) is not int or k not in {2, 3, 4, 5, 6, 8} or values.dtype != np.uint16 or
            values.ndim != 2 or values.shape[1] != 16 * k or len(values) > 1024):
        raise ValidationError("diagnostic trellis reader requires a bounded complete mul1 tile batch")
    words = np.ascontiguousarray(values).view("<u4")
    bits = np.unpackbits(words.astype(">u4").view(np.uint8).reshape(len(values), -1), axis=1, bitorder="big")
    ends = (np.arange(256) + 1) * k
    positions = (ends[:, None] - 16 + np.arange(16)) % (256 * k)
    selected = bits[:, positions]
    return np.sum(selected.astype(np.uint32) * (1 << np.arange(15, -1, -1, dtype=np.uint32)), axis=-1).astype(np.uint16)


def fragment_permutation():
    rows, columns = np.indices((16, 16))
    lane = (columns % 8 // 2) * 8 + (rows % 8 // 2) + (columns % 2) * 4
    element = (rows // 8) * 2 + rows % 2 + (columns // 8) * 4
    return (lane * 8 + element).astype(np.int32)


def reconstruct_reference(packed, k, codebook):
    values, table = np.asarray(packed), np.asarray(codebook)
    if (values.ndim != 3 or values.dtype != np.uint16 or values.shape[-1] != 16 * k or
            table.dtype != np.float16 or table.shape != (65536,) or
            values.shape[0] * values.shape[1] * 256 > 5120 * 5120):
        raise ValidationError("diagnostic reconstruction exceeds the supported bounded synthetic matrix")
    inputs, outputs = values.shape[0] * 16, values.shape[1] * 16
    dense = np.empty((inputs, outputs), dtype=np.float16)
    permutation = fragment_permutation()
    for block in range(values.shape[0]):
        codes = trellis_windows(values[block], k)[:, permutation]
        decoded = table[codes]
        dense[block * 16:(block + 1) * 16] = decoded.transpose(1, 0, 2).reshape(16, outputs)
    return dense


def hadamard_reference(values):
    """Normalized Sylvester-128 in FP64, independent of CUDA shuffle order."""
    values = np.asarray(values)
    if values.ndim != 2 or values.shape[1] % 128 or values.size > 64 * 5120:
        raise ValidationError("diagnostic Hadamard requires a bounded 128-column grouping")
    bits = np.arange(128, dtype=np.uint8)
    parity = np.unpackbits((bits[:, None] & bits).reshape(-1, 1), axis=1).sum(axis=1) % 2
    matrix = (1 - 2 * parity.astype(np.float64)).reshape(128, 128) / np.sqrt(128.)
    return (values.astype(np.float64).reshape(-1, 128) @ matrix).reshape(values.shape)


def projection_reference(inputs, dense, suh, svh):
    """FP16 activation rotation, FP64 dot/output equations and error envelope.

    Native input Hadamard sums use FP32 and round to FP16. Include their
    possible adjacent-half boundary, FP32 dot gamma_K, output rotation gamma_7
    and final scale rounding. Report the computed bound with measured errors;
    never change it after observing a GPU result.
    """
    x = np.asarray(inputs, dtype=np.float16)
    weights = np.asarray(dense, dtype=np.float16)
    su, sv = np.asarray(suh, dtype=np.float16), np.asarray(svh, dtype=np.float16)
    if x.ndim != 2 or weights.shape != (x.shape[1], len(sv)) or su.shape != (x.shape[1],):
        raise ValidationError("diagnostic packed projection dimensions differ")
    exact = hadamard_reference(x.astype(np.float64) * su.astype(np.float64))
    rotated = exact.astype(np.float16)
    accumulation = rotated.astype(np.float64) @ weights.astype(np.float64)
    expected = hadamard_reference(accumulation) * sv.astype(np.float64)
    u = 2. ** -24
    gamma_k = x.shape[1] * u / (1 - x.shape[1] * u)
    gamma_7 = 7 * u / (1 - 7 * u)
    # A full half ULP covers an input sum's FP32 perturbation at a half
    # rounding boundary; account for its propagation through every weight.
    half_ulp = np.maximum(abs(np.spacing(rotated).astype(np.float64)), 2. ** -24)
    exact_rotation_rounding = gamma_7 * np.sqrt(128.) * np.max(abs(x.astype(np.float64) * su), axis=-1, keepdims=True)
    activation_error = half_ulp + exact_rotation_rounding
    dot_bound = activation_error @ abs(weights.astype(np.float64))
    dot_bound += gamma_k * (abs(rotated.astype(np.float64)) @ abs(weights.astype(np.float64)))
    # Absolute normalized H has every entry 1/sqrt(128).
    propagated = np.repeat((dot_bound.reshape(x.shape[0], -1, 128).sum(-1) / np.sqrt(128.)), 128, axis=-1)
    output_round = gamma_7 * np.repeat((abs(accumulation).reshape(x.shape[0], -1, 128).sum(-1) / np.sqrt(128.)), 128, axis=-1)
    bound = (propagated + output_round) * abs(sv.astype(np.float64)) + 2 * u * abs(expected) + 1e-7
    return expected, bound
