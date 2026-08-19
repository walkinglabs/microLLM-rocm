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
};

[[nodiscard]] std::int32_t sample_token(const std::vector<float>& logits,
                                        float temperature, std::int64_t top_k,
                                        std::mt19937_64& generator);

[[nodiscard]] std::vector<std::int32_t> generate(
    model::TransformerModel& model, const std::vector<std::int32_t>& prompt,
    const GenerationConfig& config = {});

}  // namespace microllm::inference
