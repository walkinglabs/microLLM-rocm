#include <microllm/inference/kv_cache.h>

#include <stdexcept>

#include <microllm/ops/ops.h>

namespace microllm::inference {
namespace {

void clear_tensor_row(Tensor& tensor, std::int64_t row) {
    if (!tensor.defined()) return;
    if (tensor.ndim() != 4 || tensor.shape()[0] <= row ||
        tensor.shape()[1] <= 0 || tensor.shape()[3] <= 0 ||
        tensor.stride(3) != 1 || tensor.stride(2) != tensor.shape()[3]) {
        throw std::logic_error("KV cache row layout is invalid");
    }
    const auto heads = tensor.shape()[1];
    const auto width = tensor.shape()[3];
    const auto capacity = tensor.stride(1) / width;
    if (capacity < tensor.shape()[2] || tensor.stride(0) != heads * capacity * width) {
        throw std::logic_error("KV cache row capacity layout is invalid");
    }
    auto full_row = Tensor::from_storage(
        tensor.storage(), {heads, capacity, width},
        contiguous_strides({heads, capacity, width}),
        tensor.storage_offset() + row * tensor.stride(0), tensor.dtype());
    ops::fill_(full_row, 0.0F);
}

}  // namespace

void KVCache::clear_row(std::int64_t row) {
    if (row < 0 || row >= batch_size_) {
        throw std::out_of_range("KV cache row is outside the batch");
    }
    for (auto& layer_state : layers_) {
        clear_tensor_row(layer_state.key, row);
        clear_tensor_row(layer_state.value, row);
    }
}

}  // namespace microllm::inference
