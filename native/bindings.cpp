#include <torch/extension.h>
#include <vector>

std::vector<torch::Tensor> ds_sparse_attention(
    torch::Tensor q, torch::Tensor window, torch::Tensor source,
    torch::Tensor indices, torch::Tensor sink, double scale);
std::vector<torch::Tensor> ds_index_scores(
    torch::Tensor q, torch::Tensor keys, torch::Tensor weights,
    torch::Tensor indices, torch::Tensor visible);
std::vector<torch::Tensor> ds_engram_decode(torch::Tensor encoded);
torch::Tensor ds_scatter_slots(
    torch::Tensor pool, torch::Tensor generations, torch::Tensor slots,
    torch::Tensor expected_generations, torch::Tensor values);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("sparse_attention", &ds_sparse_attention);
    m.def("index_scores", &ds_index_scores);
    m.def("engram_decode", &ds_engram_decode);
    m.def("scatter_slots", &ds_scatter_slots);
    m.def("abi_version", [] { return 1; });
}
