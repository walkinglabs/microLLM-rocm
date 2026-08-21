#include <microllm/inference/generator.h>

#include <algorithm>
#include <cmath>
#include <numeric>
#include <stdexcept>
#include <utility>

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
    auto layer_dtypes = config.kv_cache_layer_dtypes;
    if (layer_dtypes.empty()) {
        layer_dtypes.assign(static_cast<std::size_t>(model.config().layers),
                            config.kv_cache_dtype);
    } else if (layer_dtypes.size() !=
               static_cast<std::size_t>(model.config().layers)) {
        throw std::invalid_argument(
            "generation KV cache policy must contain one dtype per layer");
    }
    KVCache cache(std::move(layer_dtypes),
                  static_cast<std::int64_t>(prompt.size()) + config.max_new_tokens);
    auto logits = model.forward_prefill_cached(
        Tensor::from_int32_vector(
            prompt, {1, static_cast<std::int64_t>(prompt.size())}),
        cache);
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

std::vector<std::vector<std::int32_t>> generate_batch(
    model::TransformerModel& model,
    const std::vector<std::vector<std::int32_t>>& prompts,
    const GenerationConfig& config) {
    if (prompts.empty() || prompts.front().empty()) {
        throw std::invalid_argument("batched generation requires non-empty prompts");
    }
    if (config.max_new_tokens < 0 || config.temperature < 0.0F ||
        !std::isfinite(config.temperature) || config.top_k < 0 ||
        config.top_k > model.config().vocabulary_size) {
        throw std::invalid_argument("batched generation configuration is invalid");
    }
    const auto prompt_length = prompts.front().size();
    for (const auto& prompt : prompts) {
        if (prompt.size() != prompt_length) {
            throw std::invalid_argument("batched generation prompts must have equal length");
        }
        for (const auto token : prompt) {
            if (token < 0 || token >= model.config().vocabulary_size) {
                throw std::out_of_range("batched prompt token is outside the vocabulary");
            }
        }
    }
    if (static_cast<std::int64_t>(prompt_length) + config.max_new_tokens >
        model.config().max_sequence_length) {
        throw std::invalid_argument("batched generation exceeds model context");
    }
    if (config.max_new_tokens == 0) return prompts;
    auto layer_dtypes = config.kv_cache_layer_dtypes;
    if (layer_dtypes.empty()) {
        layer_dtypes.assign(static_cast<std::size_t>(model.config().layers),
                            config.kv_cache_dtype);
    } else if (layer_dtypes.size() !=
               static_cast<std::size_t>(model.config().layers)) {
        throw std::invalid_argument(
            "batched generation KV policy must contain one dtype per layer");
    }
    const auto batch = static_cast<std::int64_t>(prompts.size());
    std::vector<std::int32_t> flat;
    flat.reserve(prompts.size() * prompt_length);
    for (const auto& prompt : prompts) flat.insert(flat.end(), prompt.begin(), prompt.end());
    KVCache cache(std::move(layer_dtypes),
                  static_cast<std::int64_t>(prompt_length) + config.max_new_tokens,
                  batch);
    auto logits = model.forward_prefill_cached(
        Tensor::from_int32_vector(
            flat, {batch, static_cast<std::int64_t>(prompt_length)}),
        cache);
    auto result = prompts;
    std::vector<std::mt19937_64> random;
    random.reserve(prompts.size());
    for (std::size_t row = 0; row < prompts.size(); ++row) {
        random.emplace_back(config.seed);
    }
    for (std::int64_t generated = 0; generated < config.max_new_tokens; ++generated) {
        Tensor next_tensor;
        std::vector<std::int32_t> selected;
        if (logits.device().is_hip() &&
            (config.temperature == 0.0F || config.top_k == 1)) {
            next_tensor = ops::argmax_last_dim(logits);
            selected = next_tensor.to_int32_vector();
        } else {
            const auto values = logits.to_vector();
            const auto vocabulary = static_cast<std::size_t>(model.config().vocabulary_size);
            selected.reserve(prompts.size());
            for (std::size_t row = 0; row < prompts.size(); ++row) {
                const auto begin = values.begin() +
                                   static_cast<std::ptrdiff_t>(row * vocabulary);
                selected.push_back(sample_token(
                    std::vector<float>(begin, begin + static_cast<std::ptrdiff_t>(vocabulary)),
                    config.temperature, config.top_k, random[row]));
            }
        }
        if (selected.size() != prompts.size() ||
            std::any_of(selected.begin(), selected.end(),
                        [](std::int32_t token) { return token < 0; })) {
            throw std::invalid_argument("batched generation logits are non-finite");
        }
        for (std::size_t row = 0; row < result.size(); ++row) {
            result[row].push_back(selected[row]);
        }
        if (generated + 1 < config.max_new_tokens) {
            if (!next_tensor.defined()) {
                next_tensor = Tensor::from_int32_vector(selected, {batch, 1});
            }
            logits = model.forward_cached(next_tensor, cache);
        }
    }
    return result;
}

}  // namespace microllm::inference
