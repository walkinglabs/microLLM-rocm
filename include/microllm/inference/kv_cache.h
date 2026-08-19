#pragma once

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <vector>

#include <microllm/core/tensor.h>

namespace microllm::inference {

class KVCache {
public:
    struct LayerState {
        Tensor key;
        Tensor value;
    };

    KVCache(std::int64_t layers, std::int64_t max_sequence_length)
        : max_sequence_length_(max_sequence_length),
          layers_(static_cast<std::size_t>(layers)) {
        if (layers <= 0 || max_sequence_length <= 0) {
            throw std::invalid_argument("KV cache dimensions must be positive");
        }
    }

    [[nodiscard]] std::int64_t position() const noexcept { return position_; }
    [[nodiscard]] std::int64_t max_sequence_length() const noexcept {
        return max_sequence_length_;
    }
    [[nodiscard]] std::size_t layer_count() const noexcept { return layers_.size(); }
    [[nodiscard]] const LayerState& layer(std::size_t index) const { return layers_.at(index); }
    [[nodiscard]] LayerState& mutable_layer(std::size_t index) { return layers_.at(index); }

    void advance() {
        if (position_ >= max_sequence_length_) throw std::out_of_range("KV cache is full");
        ++position_;
    }

    void reset() {
        position_ = 0;
        for (auto& layer_state : layers_) layer_state = {};
    }

private:
    std::int64_t max_sequence_length_;
    std::int64_t position_ = 0;
    std::vector<LayerState> layers_;
};

}  // namespace microllm::inference
