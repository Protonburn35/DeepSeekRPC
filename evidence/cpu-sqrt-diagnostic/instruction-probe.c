#include <immintrin.h>
#include <stddef.h>
void sqrt_probe(const float *input, float *output, size_t count, int variant) {
    for (size_t i = 0; i < count; i += 8) {
        __m256 x = _mm256_loadu_ps(input + i), y;
        if (variant == 0) y = _mm256_mul_ps(x, _mm256_rsqrt_ps(x));
        else if (variant == 1) y = _mm256_div_ps(_mm256_set1_ps(1.0f), _mm256_rsqrt_ps(x));
        else if (variant == 2) y = _mm256_rcp_ps(_mm256_rsqrt_ps(x));
        else y = _mm256_sqrt_ps(x);
        _mm256_storeu_ps(output + i, y);
    }
}
