#ifdef _WIN32
#include <torch/extension.h>
#include <cuda_runtime_api.h>
#include <dxgi1_4.h>
#include <wrl/client.h>
#include <cstdint>
#include <cstring>
#include <vector>

std::vector<uint64_t> ds_windows_memory_budget(int64_t ordinal) {
    int count = 0;
    TORCH_CHECK(cudaGetDeviceCount(&count) == cudaSuccess,
                "cannot query the actual NVIDIA devices");
    TORCH_CHECK(ordinal >= 0 && ordinal < count, "invalid CUDA device ordinal");
    cudaDeviceProp properties{};
    TORCH_CHECK(cudaGetDeviceProperties(&properties, static_cast<int>(ordinal)) == cudaSuccess,
                "cannot query the actual CUDA device properties");
    unsigned int mask = properties.luidDeviceNodeMask;
    TORCH_CHECK(mask && !(mask & (mask - 1)),
                "CUDA did not supply one Windows WDDM adapter node");
    UINT node = 0;
    while (mask > 1) { mask >>= 1; ++node; }
    LUID luid{};
    static_assert(sizeof(luid) == sizeof(properties.luid));
    std::memcpy(&luid, properties.luid, sizeof(luid));
    Microsoft::WRL::ComPtr<IDXGIFactory4> factory;
    HRESULT result = CreateDXGIFactory1(IID_PPV_ARGS(factory.GetAddressOf()));
    TORCH_CHECK(SUCCEEDED(result), "cannot create the Windows DXGI factory: ", result);
    Microsoft::WRL::ComPtr<IDXGIAdapter3> adapter;
    result = factory->EnumAdapterByLuid(luid, IID_PPV_ARGS(adapter.GetAddressOf()));
    TORCH_CHECK(SUCCEEDED(result), "cannot resolve CUDA's actual WDDM adapter: ", result);
    DXGI_QUERY_VIDEO_MEMORY_INFO local{}, shared{};
    result = adapter->QueryVideoMemoryInfo(node, DXGI_MEMORY_SEGMENT_GROUP_LOCAL, &local);
    TORCH_CHECK(SUCCEEDED(result), "cannot query the local WDDM memory budget: ", result);
    result = adapter->QueryVideoMemoryInfo(node, DXGI_MEMORY_SEGMENT_GROUP_NON_LOCAL, &shared);
    TORCH_CHECK(SUCCEEDED(result), "cannot query the nonlocal WDDM memory budget: ", result);
    DXGI_ADAPTER_DESC2 descriptor{};
    result = adapter->GetDesc2(&descriptor);
    TORCH_CHECK(SUCCEEDED(result), "cannot query the actual adapter memory capacity: ", result);
    return {static_cast<uint64_t>(properties.totalGlobalMem), local.Budget, local.CurrentUsage,
            local.AvailableForReservation, local.CurrentReservation, shared.Budget, shared.CurrentUsage,
            static_cast<uint64_t>(descriptor.DedicatedVideoMemory),
            static_cast<uint64_t>(descriptor.SharedSystemMemory)};
}
#endif
