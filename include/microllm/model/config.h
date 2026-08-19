#pragma once

#include <cstdint>
#include <string>

namespace microllm::model {

struct ModelConfig {
    std::int64_t vocabulary_size = 0;
    std::int64_t dimension = 0;
    std::int64_t layers = 0;
    std::int64_t heads = 0;
    std::int64_t kv_heads = 0;
    std::int64_t ffn_dimension = 0;
    std::int64_t max_sequence_length = 0;
    float rope_base = 10000.0F;
    bool tie_embeddings = false;

    void validate() const;
    [[nodiscard]] std::int64_t head_dimension() const;
    [[nodiscard]] std::int64_t kv_dimension() const;
    [[nodiscard]] std::uint64_t parameter_count() const;
    [[nodiscard]] std::uint64_t weight_bytes(std::uint64_t bytes_per_parameter) const;
    [[nodiscard]] std::string summary() const;

    [[nodiscard]] static ModelConfig model_s();
    [[nodiscard]] static ModelConfig model_m();
};

}  // namespace microllm::model
