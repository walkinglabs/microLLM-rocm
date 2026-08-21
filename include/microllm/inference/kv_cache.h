#pragma once

#include <algorithm>
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
          row_positions_(batch_size > 0 ? static_cast<std::size_t>(batch_size) : 0U, 0),
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
          row_positions_(batch_size > 0 ? static_cast<std::size_t>(batch_size) : 0U, 0),
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

    [[nodiscard]] std::int64_t position() const {
        if (!positions_uniform()) {
            throw std::logic_error("KV cache rows do not share one position");
        }
        return row_positions_.front();
    }
    [[nodiscard]] std::int64_t row_position(std::int64_t row) const {
        if (row < 0 || row >= batch_size_) {
            throw std::out_of_range("KV cache row is outside the batch");
        }
        return row_positions_[static_cast<std::size_t>(row)];
    }
    [[nodiscard]] const std::vector<std::int64_t>& row_positions() const noexcept {
        return row_positions_;
    }
    [[nodiscard]] bool positions_uniform() const noexcept {
        return std::all_of(row_positions_.begin() + 1, row_positions_.end(),
                           [this](std::int64_t value) {
                               return value == row_positions_.front();
                           });
    }
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
        if (count <= 0 || std::any_of(
                              row_positions_.begin(), row_positions_.end(),
                              [this, count](std::int64_t position) {
                                  return count > max_sequence_length_ - position;
                              })) {
            throw std::out_of_range("KV cache advance is outside capacity");
        }
        for (auto& position : row_positions_) position += count;
    }

    void advance_row(std::int64_t row, std::int64_t count = 1) {
        if (row < 0 || row >= batch_size_) {
            throw std::out_of_range("KV cache row is outside the batch");
        }
        auto& position = row_positions_[static_cast<std::size_t>(row)];
        if (count <= 0 || count > max_sequence_length_ - position) {
            throw std::out_of_range("KV cache row advance is outside capacity");
        }
        position += count;
    }

    void reset() {
        std::fill(row_positions_.begin(), row_positions_.end(), 0);
        for (auto& layer_state : layers_) layer_state = {};
    }

    // Clears one batch row across the full backing capacity without changing
    // the shared logical position. This is a storage-ownership primitive;
    // per-slot positions are a separate scheduler capability.
    void clear_row(std::int64_t row);
    // Clears one row and returns only that row's logical position to zero.
    void reset_row(std::int64_t row);

private:
    std::int64_t max_sequence_length_;
    std::int64_t batch_size_ = 1;
    std::vector<std::int64_t> row_positions_;
    std::vector<DType> layer_dtypes_;
    std::vector<LayerState> layers_;
};

}  // namespace microllm::inference
