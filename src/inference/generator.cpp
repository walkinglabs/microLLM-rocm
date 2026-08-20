#include <microllm/inference/generator.h>

#include <algorithm>
#include <cmath>
#include <numeric>
#include <stdexcept>

#include <microllm/inference/kv_cache.h>
#include <microllm/ops/ops.h>

namespace microllm::inference {

std::int32_t sample_token(const std::vector<float>& logits, float temperature,
                          std::int64_t top_k, std::mt19937_64& generator) {
    if (logits.empty()) throw std::invalid_argument("cannot sample empty logits");
    if (temperature < 0.0F || !std::isfinite(temperature)) {
        throw std::invalid_argument("sampling temperature must be finite and non-negative");
    }
    if (top_k < 0 || top_k > static_cast<std::int64_t>(logits.size())) {
        throw std::invalid_argument("top_k is outside the vocabulary");
    }
    for (const auto value : logits) {
        if (!std::isfinite(value)) throw std::invalid_argument("sampling logits must be finite");
    }
    if (temperature == 0.0F || top_k == 1) {
        return static_cast<std::int32_t>(
            std::distance(logits.begin(), std::max_element(logits.begin(), logits.end())));
    }

    std::vector<std::size_t> indices(logits.size());
    std::iota(indices.begin(), indices.end(), 0);
    const auto candidate_count = top_k == 0 ? indices.size() : static_cast<std::size_t>(top_k);
    std::partial_sort(indices.begin(), indices.begin() + static_cast<std::ptrdiff_t>(candidate_count),
                      indices.end(), [&logits](std::size_t left, std::size_t right) {
                          return logits[left] > logits[right];
                      });
    indices.resize(candidate_count);
    const auto maximum = logits[indices.front()] / temperature;
    std::vector<double> weights;
    weights.reserve(candidate_count);
    for (const auto index : indices) {
        weights.push_back(std::exp(static_cast<double>(logits[index] / temperature - maximum)));
    }
    std::discrete_distribution<std::size_t> distribution(weights.begin(), weights.end());
    return static_cast<std::int32_t>(indices[distribution(generator)]);
}

std::vector<std::int32_t> generate(model::TransformerModel& model,
                                   const std::vector<std::int32_t>& prompt,
                                   const GenerationConfig& config) {
    if (prompt.empty()) throw std::invalid_argument("generation prompt cannot be empty");
    if (config.max_new_tokens < 0) {
        throw std::invalid_argument("max_new_tokens must be non-negative");
    }
    if (static_cast<std::int64_t>(prompt.size()) + config.max_new_tokens >
        model.config().max_sequence_length) {
        throw std::invalid_argument("prompt plus generated tokens exceeds model context");
    }
    for (const auto token : prompt) {
        if (token < 0 || token >= model.config().vocabulary_size) {
            throw std::out_of_range("prompt token is outside the model vocabulary");
        }
    }
    if (config.max_new_tokens == 0) return prompt;

    // Reserve exactly this request's upper bound.  The model may support a much
    // larger context, but allocating that full theoretical capacity for a short
    // generation needlessly turns stable cache addresses into a memory penalty.
    KVCache cache(model.config().layers,
                  static_cast<std::int64_t>(prompt.size()) + config.max_new_tokens);
    Tensor logits;
    for (const auto token : prompt) {
        logits = model.forward_cached(Tensor::from_int32_vector({token}, {1, 1}), cache);
    }
    std::mt19937_64 generator(config.seed);
    auto tokens = prompt;
    for (std::int64_t generated = 0; generated < config.max_new_tokens; ++generated) {
        Tensor next_tensor;
        std::int32_t next = 0;
        if (logits.device().is_hip() &&
            (config.temperature == 0.0F || config.top_k == 1)) {
            next_tensor = ops::argmax(logits);
            next = next_tensor.to_int32_vector().front();
            if (next < 0) throw std::invalid_argument("sampling logits must be finite");
        } else {
            next = sample_token(logits.to_vector(), config.temperature, config.top_k,
                                generator);
        }
        tokens.push_back(next);
        if (generated + 1 < config.max_new_tokens) {
            if (!next_tensor.defined()) {
                next_tensor = Tensor::from_int32_vector({next}, {1, 1});
            }
            logits = model.forward_cached(next_tensor, cache);
        }
    }
    return tokens;
}

}  // namespace microllm::inference
