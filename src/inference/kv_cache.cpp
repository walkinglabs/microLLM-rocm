#include <microllm/inference/kv_cache.h>

#include <algorithm>
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

void resize_tensor_prefix(Tensor& tensor, std::int64_t prefix) {
    if (!tensor.defined()) return;
    const auto heads = tensor.shape()[1];
    const auto width = tensor.shape()[3];
    const auto capacity = tensor.stride(1) / width;
    if (prefix < 0 || prefix > capacity) {
        throw std::logic_error("KV cache prefix is outside backing capacity");
    }
    tensor = Tensor::from_storage(
        tensor.storage(), {tensor.shape()[0], heads, prefix, width},
        tensor.strides(), tensor.storage_offset(), tensor.dtype());
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

void KVCache::reset_row(std::int64_t row) {
    clear_row(row);
    row_positions_[static_cast<std::size_t>(row)] = 0;
    const auto prefix = *std::max_element(row_positions_.begin(), row_positions_.end());
    for (auto& layer_state : layers_) {
        resize_tensor_prefix(layer_state.key, prefix);
        resize_tensor_prefix(layer_state.value, prefix);
    }
}

}  // namespace microllm::inference
