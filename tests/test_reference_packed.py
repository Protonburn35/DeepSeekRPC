import numpy as np
import pytest

from deepseekrpc.reference_packed import fragment_permutation, hadamard_reference, trellis_windows


@pytest.mark.parametrize("k", [2, 3, 4, 5, 6, 8])
def test_circular_msb_word_reader_matches_independent_scalar_bits_at_every_state(k):
    rng = np.random.default_rng(41017 + k)
    words = rng.integers(0, 1 << 32, (2, 8 * k), dtype=np.uint32)
    actual = trellis_windows(words.astype("<u4").view("<u2"), k)
    for tile in range(2):
        bit_string = "".join(f"{int(value):032b}" for value in words[tile])
        for symbol in range(256):
            end = (symbol + 1) * k
            expected = 0
            for position in range(end - 16, end):
                expected = expected * 2 + int(bit_string[position % len(bit_string)])
            assert actual[tile, symbol] == expected


def test_fragment_layout_matches_frozen_reconstruct_shuffle_coordinates():
    expected = np.empty((16, 16), dtype=np.int32)
    # Independent forward stores from the pinned reconstruct_tile CUDA body.
    for lane in range(32):
        if lane & 4:
            continue
        for fragment, row in enumerate((2 * (lane % 4), 2 * (lane % 4) + 1,
                                        2 * (lane % 4) + 8, 2 * (lane % 4) + 9)):
            for group in range(2):
                column = 2 * (lane // 8) + group * 8
                expected[row, column] = lane * 8 + fragment + group * 4
                expected[row, column + 1] = (lane + 4) * 8 + fragment + group * 4
    np.testing.assert_array_equal(fragment_permutation(), expected)
    assert sorted(expected.reshape(-1)) == list(range(256))


def test_normalized_hadamard_inverse_and_independent_parity_equations():
    values = np.arange(256, dtype=np.float64).reshape(2, 128) / 64
    result = hadamard_reference(values)
    for row in range(2):
        for column in (0, 1, 17, 127):
            expected = sum(values[row, j] * (-1 if (j & column).bit_count() % 2 else 1) for j in range(128)) / np.sqrt(128.)
            assert result[row, column] == pytest.approx(expected, abs=1e-13)
    np.testing.assert_allclose(hadamard_reference(result), values, rtol=1e-13, atol=1e-13)
