#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cmath>
#include <cstdint>
#include <vector>

namespace {
void check(const torch::Tensor& t, torch::ScalarType dtype, const torch::Tensor& anchor) {
    TORCH_CHECK(t.is_cuda() && t.device() == anchor.device(), "all inputs require the same CUDA device");
    TORCH_CHECK(t.scalar_type() == dtype && t.is_contiguous(), "input dtype/contiguity mismatch");
}

__device__ float warp_sum(float value) {
    for (int distance = 16; distance; distance >>= 1)
        value += __shfl_down_sync(0xffffffff, value, distance);
    return value;
}

__global__ void sparse_kernel(
    const float* q, const half* window, const half* source, const int32_t* ids,
    const float* sink, float* out, int32_t* error, int heads, int width,
    int64_t window_rows, int64_t source_rows, int selected, float scale) {
    const int row = blockIdx.x, head = blockIdx.y;
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    __shared__ float partial[8];
    __shared__ float score;
    float query[2], acc[2] = {0.0f, 0.0f};
    for (int item = 0; item < 2; ++item) {
        const int dim = threadIdx.x + item * 256;
        query[item] = dim < width ? q[(static_cast<int64_t>(row) * heads + head) * width + dim] : 0.0f;
    }
    float maximum = sink[head], denominator = 1.0f;
    if (!isfinite(maximum)) { atomicCAS(error, 0, 2); maximum = 0.0f; }
    for (int column = 0; column < selected; ++column) {
        const int64_t id = ids[static_cast<int64_t>(row) * selected + column];
        if (id == -1) continue;
        if (id < 0 || id >= window_rows + source_rows) {
            if (threadIdx.x == 0) atomicCAS(error, 0, 1);
            continue;
        }
        const half* key = id < window_rows ? window + id * width : source + (id - window_rows) * width;
        float dot = 0.0f, value[2] = {0.0f, 0.0f};
        for (int item = 0; item < 2; ++item) {
            const int dim = threadIdx.x + item * 256;
            if (dim < width) { value[item] = __half2float(key[dim]); dot += query[item] * value[item]; }
        }
        dot = warp_sum(dot);
        if (lane == 0) partial[warp] = dot;
        __syncthreads();
        if (warp == 0) {
            float sum = warp_sum(lane < 8 ? partial[lane] : 0.0f);
            if (lane == 0) score = sum * scale;
        }
        __syncthreads();
        if (!isfinite(score)) { if (threadIdx.x == 0) atomicCAS(error, 0, 2); }
        const float next_max = fmaxf(maximum, score);
        const float previous_weight = expf(maximum - next_max), weight = expf(score - next_max);
        denominator = denominator * previous_weight + weight;
        for (int item = 0; item < 2; ++item) acc[item] = acc[item] * previous_weight + weight * value[item];
        maximum = next_max;
        __syncthreads();
    }
    for (int item = 0; item < 2; ++item) {
        const int dim = threadIdx.x + item * 256;
        if (dim < width) out[(static_cast<int64_t>(row) * heads + head) * width + dim] = acc[item] / denominator;
    }
}

__global__ void index_kernel(
    const float* q, const half* keys, const float* weights, const int32_t* ids,
    const int32_t* visible, float* out, int32_t* error, int heads, int width,
    int columns, int64_t capacity) {
    const int row = blockIdx.y, column = blockIdx.x;
    const int id = ids[static_cast<int64_t>(row) * columns + column];
    if (visible[row] < 0 || visible[row] > capacity) {
        if (threadIdx.x == 0) { atomicCAS(error, 0, 1); out[static_cast<int64_t>(row) * columns + column] = -INFINITY; }
        return;
    }
    if (id == -1 || (id >= 0 && id < capacity && id >= visible[row])) {
        if (threadIdx.x == 0) out[static_cast<int64_t>(row) * columns + column] = -INFINITY;
        return;
    }
    if (id < 0 || id >= capacity) {
        if (threadIdx.x == 0) { atomicCAS(error, 0, 1); out[static_cast<int64_t>(row) * columns + column] = -INFINITY; }
        return;
    }
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    __shared__ float head_scores[32];
    for (int head = warp; head < heads; head += 4) {
        float dot = 0.0f;
        for (int dim = lane; dim < width; dim += 32)
            dot += q[(static_cast<int64_t>(row) * heads + head) * width + dim]
                 * __half2float(keys[static_cast<int64_t>(id) * width + dim]);
        dot = warp_sum(dot);
        if (lane == 0) head_scores[head] = fmaxf(dot, 0.0f) * weights[row * heads + head];
    }
    __syncthreads();
    if (warp == 0) {
        float value = warp_sum(lane < heads ? head_scores[lane] : 0.0f);
        if (lane == 0) {
            if (!isfinite(value)) atomicCAS(error, 0, 2);
            out[static_cast<int64_t>(row) * columns + column] = value;
        }
    }
}

__global__ void engram_kernel(const uint8_t* encoded, __nv_bfloat16* out, int32_t* error, int64_t elements) {
    const int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= elements) return;
    const int64_t row = index / 256;
    const int dim = index % 256;
    const uint8_t bits = encoded[row * 264 + dim], scale = encoded[row * 264 + 256 + dim / 32];
    const int exponent = (bits >> 3) & 15, mantissa = bits & 7;
    if ((exponent == 15 && mantissa == 7) || scale == 255) { atomicCAS(error, 0, 2); out[index] = __float2bfloat16(0.0f); return; }
    float value = exponent == 0 ? ldexpf(static_cast<float>(mantissa), -9)
                               : ldexpf(1.0f + mantissa / 8.0f, exponent - 7);
    if (bits & 128) value = -value;
    // Standard UE8M0 byte zero is 2^-127, including its FP32 subnormal value.
    value *= ldexpf(1.0f, static_cast<int>(scale) - 127);
    if (!isfinite(value)) atomicCAS(error, 0, 2);
    const __nv_bfloat16 result = __float2bfloat16_rn(value);
    if (!isfinite(__bfloat162float(result))) atomicCAS(error, 0, 2);
    out[index] = result;
}

__global__ void scatter_kernel(
    half* pool, const int64_t* generations, const int32_t* slots,
    const int64_t* expected, const half* values, int32_t* error,
    int rows, int width, int64_t capacity) {
    const int row = blockIdx.x;
    const int64_t slot = slots[row];
    bool valid = slot >= 0 && slot < capacity;
    for (int previous = 0; previous < row; ++previous) if (slots[previous] == slot) valid = false;
    if (!valid || generations[slot] != expected[row]) {
        if (threadIdx.x == 0) atomicCAS(error, 0, 3);
        return;
    }
    for (int dim = threadIdx.x; dim < width; dim += blockDim.x)
        pool[slot * width + dim] = values[row * width + dim];
}
}

std::vector<torch::Tensor> ds_sparse_attention(torch::Tensor q, torch::Tensor window, torch::Tensor source,
    torch::Tensor indices, torch::Tensor sink, double scale) {
    check(q, torch::kFloat32, q); check(window, torch::kFloat16, q); check(source, torch::kFloat16, q);
    check(indices, torch::kInt32, q); check(sink, torch::kFloat32, q);
    TORCH_CHECK(q.dim() == 3 && q.size(0) > 0 && q.size(0) <= 4096 && q.size(1) > 0 && q.size(1) <= 128,
                "invalid bounded query shape");
    const auto width = q.size(2);
    TORCH_CHECK(width > 0 && width <= 512 && window.dim() == 2 && source.dim() == 2 &&
                window.size(1) == width && source.size(1) == width && window.size(0) <= 8192 &&
                source.size(0) <= 1048576, "KV dimensions/capacity differ from bounded query");
    TORCH_CHECK(indices.dim() == 2 && indices.size(0) == q.size(0) && indices.size(1) <= 4096 &&
                sink.dim() == 1 && sink.size(0) == q.size(1), "invalid sparse indices/sink shape");
    TORCH_CHECK(std::isfinite(scale) && scale > 0.0, "invalid attention scale");
    const c10::cuda::CUDAGuard guard(q.device());
    auto out = torch::empty_like(q), error = torch::zeros({1}, q.options().dtype(torch::kInt32));
    sparse_kernel<<<dim3(q.size(0), q.size(1)), 256, 0, c10::cuda::getCurrentCUDAStream()>>>(
        q.data_ptr<float>(), reinterpret_cast<half*>(window.data_ptr<at::Half>()), reinterpret_cast<half*>(source.data_ptr<at::Half>()),
        indices.data_ptr<int32_t>(), sink.data_ptr<float>(), out.data_ptr<float>(), error.data_ptr<int32_t>(),
        q.size(1), width, window.size(0), source.size(0), indices.size(1), static_cast<float>(scale));
    C10_CUDA_KERNEL_LAUNCH_CHECK(); return {out, error};
}

std::vector<torch::Tensor> ds_index_scores(torch::Tensor q, torch::Tensor keys, torch::Tensor weights,
    torch::Tensor indices, torch::Tensor visible) {
    check(q, torch::kFloat32, q); check(keys, torch::kFloat16, q); check(weights, torch::kFloat32, q);
    check(indices, torch::kInt32, q); check(visible, torch::kInt32, q);
    TORCH_CHECK(q.dim() == 3 && q.size(0) > 0 && q.size(0) <= 4096 && q.size(1) > 0 && q.size(1) <= 32 &&
                q.size(2) == 128 && keys.dim() == 2 && keys.size(1) == 128, "invalid index query/key shape");
    TORCH_CHECK(weights.dim() == 2 && weights.size(0) == q.size(0) && weights.size(1) == q.size(1) &&
                visible.dim() == 1 && visible.size(0) == q.size(0) && indices.dim() == 2 &&
                indices.size(0) == q.size(0) && indices.size(1) > 0 && indices.size(1) <= 8192, "invalid bounded index tile");
    const c10::cuda::CUDAGuard guard(q.device());
    auto out = torch::empty(indices.sizes(), q.options()), error = torch::zeros({1}, q.options().dtype(torch::kInt32));
    index_kernel<<<dim3(indices.size(1), q.size(0)), 128, 0, c10::cuda::getCurrentCUDAStream()>>>(
        q.data_ptr<float>(), reinterpret_cast<half*>(keys.data_ptr<at::Half>()), weights.data_ptr<float>(),
        indices.data_ptr<int32_t>(), visible.data_ptr<int32_t>(), out.data_ptr<float>(), error.data_ptr<int32_t>(),
        q.size(1), 128, indices.size(1), keys.size(0));
    C10_CUDA_KERNEL_LAUNCH_CHECK(); return {out, error};
}

std::vector<torch::Tensor> ds_engram_decode(torch::Tensor encoded) {
    check(encoded, torch::kUInt8, encoded);
    TORCH_CHECK(encoded.dim() == 2 && encoded.size(0) > 0 && encoded.size(0) <= 1024 && encoded.size(1) == 264,
                "encoded Engram rows must be bounded [rows,264] uint8");
    const c10::cuda::CUDAGuard guard(encoded.device());
    auto out = torch::empty({encoded.size(0), 256}, encoded.options().dtype(torch::kBFloat16));
    auto error = torch::zeros({1}, encoded.options().dtype(torch::kInt32));
    engram_kernel<<<(out.numel() + 255) / 256, 256, 0, c10::cuda::getCurrentCUDAStream()>>>(
        encoded.data_ptr<uint8_t>(), reinterpret_cast<__nv_bfloat16*>(out.data_ptr<at::BFloat16>()),
        error.data_ptr<int32_t>(), out.numel());
    C10_CUDA_KERNEL_LAUNCH_CHECK(); return {out, error};
}

torch::Tensor ds_scatter_slots(torch::Tensor pool, torch::Tensor generations, torch::Tensor slots,
    torch::Tensor expected_generations, torch::Tensor values) {
    check(pool, torch::kFloat16, pool); check(generations, torch::kInt64, pool); check(slots, torch::kInt32, pool);
    check(expected_generations, torch::kInt64, pool); check(values, torch::kFloat16, pool);
    TORCH_CHECK(pool.dim() == 2 && pool.size(1) > 0 && pool.size(1) <= 512 && generations.dim() == 1 &&
                generations.size(0) == pool.size(0), "invalid cache/generation pool");
    TORCH_CHECK(slots.dim() == 1 && slots.size(0) > 0 && slots.size(0) <= 4096 && expected_generations.dim() == 1 &&
                expected_generations.size(0) == slots.size(0) && values.dim() == 2 && values.size(0) == slots.size(0) &&
                values.size(1) == pool.size(1), "invalid cache update rows");
    const c10::cuda::CUDAGuard guard(pool.device());
    auto error = torch::zeros({1}, pool.options().dtype(torch::kInt32));
    scatter_kernel<<<slots.size(0), 256, 0, c10::cuda::getCurrentCUDAStream()>>>(
        reinterpret_cast<half*>(pool.data_ptr<at::Half>()), generations.data_ptr<int64_t>(), slots.data_ptr<int32_t>(),
        expected_generations.data_ptr<int64_t>(), reinterpret_cast<half*>(values.data_ptr<at::Half>()),
        error.data_ptr<int32_t>(), slots.size(0), pool.size(1), pool.size(0));
    C10_CUDA_KERNEL_LAUNCH_CHECK(); return error;
}
