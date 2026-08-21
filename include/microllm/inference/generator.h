#pragma once

#include <cstdint>
#include <random>
#include <vector>

#include <microllm/model/model.h>

namespace microllm::inference {

struct GenerationConfig {
    std::int64_t max_new_tokens = 32;
    float temperature = 1.0F;
    std::int64_t top_k = 0;
    std::uint64_t seed = 1;
    DType kv_cache_dtype = DType::Float32;
    // Empty means every layer uses kv_cache_dtype. Otherwise the vector must
    // contain one FP32/BF16 entry per model layer.
    std::vector<DType> kv_cache_layer_dtypes;
    // Generation stops after appending any listed token. Tokens must be unique
    // vocabulary IDs. An empty vector disables content-based early stopping.
    std::vector<std::int32_t> stop_tokens;
};

[[nodiscard]] std::int32_t sample_token(const std::vector<float>& logits,
                                        float temperature, std::int64_t top_k,
                                        std::mt19937_64& generator);

[[nodiscard]] std::vector<std::int32_t> generate(
    model::TransformerModel& model, const std::vector<std::int32_t>& prompt,
    const GenerationConfig& config = {});

// Static cross-request batch. Prompts may contain different tokens but must
// share a non-zero length and one GenerationConfig. Returns full sequences in
// input row order; stop tokens may make row lengths differ.
[[nodiscard]] std::vector<std::vector<std::int32_t>> generate_batch(
    model::TransformerModel& model,
    const std::vector<std::vector<std::int32_t>>& prompts,
    const GenerationConfig& config = {});

}  // namespace microllm::inference
