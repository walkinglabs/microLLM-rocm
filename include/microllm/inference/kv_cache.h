#pragma once

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <utility>
#include <vector>

#include <microllm/core/tensor.h>

namespace microllm::inference {

class KVCache {
public:
    struct LayerState {
        Tensor key;
        Tensor value;
    };

    KVCache(std::int64_t layers, std::int64_t max_sequence_length,
            std::int64_t batch_size = 1,
            DType dtype = DType::Float32)
        : max_sequence_length_(max_sequence_length),
          batch_size_(batch_size),
          layer_dtypes_(layers > 0 ? static_cast<std::size_t>(layers) : 0U,
                        dtype),
          layers_(layers > 0 ? static_cast<std::size_t>(layers) : 0U) {
        if (layers <= 0 || max_sequence_length <= 0 || batch_size <= 0) {
            throw std::invalid_argument("KV cache dimensions must be positive");
        }
        if (dtype != DType::Float32 && dtype != DType::BFloat16) {
            throw std::invalid_argument("KV cache dtype must be float32 or bfloat16");
        }
    }

    KVCache(std::vector<DType> layer_dtypes,
            std::int64_t max_sequence_length,
            std::int64_t batch_size = 1)
        : max_sequence_length_(max_sequence_length),
          batch_size_(batch_size),
          layer_dtypes_(std::move(layer_dtypes)),
          layers_(layer_dtypes_.size()) {
        if (layer_dtypes_.empty() || max_sequence_length <= 0 || batch_size <= 0) {
            throw std::invalid_argument("KV cache dimensions must be positive");
        }
        for (const auto dtype : layer_dtypes_) {
            if (dtype != DType::Float32 && dtype != DType::BFloat16) {
                throw std::invalid_argument(
                    "KV cache layer dtypes must be float32 or bfloat16");
            }
        }
    }

    [[nodiscard]] std::int64_t position() const noexcept { return position_; }
    [[nodiscard]] std::int64_t max_sequence_length() const noexcept {
        return max_sequence_length_;
    }
    [[nodiscard]] std::int64_t batch_size() const noexcept { return batch_size_; }
    [[nodiscard]] DType dtype() const {
        const auto first = layer_dtypes_.front();
        for (const auto dtype : layer_dtypes_) {
            if (dtype != first) {
                throw std::logic_error("mixed KV cache has no single dtype");
            }
        }
        return first;
    }
    [[nodiscard]] DType layer_dtype(std::size_t index) const {
        return layer_dtypes_.at(index);
    }
    [[nodiscard]] bool has_mixed_dtypes() const noexcept {
        const auto first = layer_dtypes_.front();
        for (const auto dtype : layer_dtypes_) {
            if (dtype != first) return true;
        }
        return false;
    }
    [[nodiscard]] std::size_t layer_count() const noexcept { return layers_.size(); }
    [[nodiscard]] const LayerState& layer(std::size_t index) const { return layers_.at(index); }
    [[nodiscard]] LayerState& mutable_layer(std::size_t index) { return layers_.at(index); }

    void advance(std::int64_t count = 1) {
        if (count <= 0 || count > max_sequence_length_ - position_) {
            throw std::out_of_range("KV cache advance is outside capacity");
        }
        position_ += count;
    }

    void reset() {
        position_ = 0;
        for (auto& layer_state : layers_) layer_state = {};
    }

private:
    std::int64_t max_sequence_length_;
    std::int64_t batch_size_ = 1;
    std::vector<DType> layer_dtypes_;
    std::int64_t position_ = 0;
    std::vector<LayerState> layers_;
};

}  // namespace microllm::inference
