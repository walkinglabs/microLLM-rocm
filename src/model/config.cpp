#include <microllm/model/config.h>

#include <limits>
#include <cmath>
#include <sstream>
#include <stdexcept>

namespace microllm::model {
namespace {

std::uint64_t checked_product(std::uint64_t left, std::uint64_t right) {
    if (left != 0 && right > std::numeric_limits<std::uint64_t>::max() / left) {
        throw std::overflow_error("model parameter count overflow");
    }
    return left * right;
}

std::uint64_t checked_add(std::uint64_t left, std::uint64_t right) {
    if (right > std::numeric_limits<std::uint64_t>::max() - left) {
        throw std::overflow_error("model parameter count overflow");
    }
    return left + right;
}

}  // namespace

void ModelConfig::validate() const {
    if (vocabulary_size <= 0 || dimension <= 0 || layers <= 0 || heads <= 0 ||
        kv_heads <= 0 || ffn_dimension <= 0 || max_sequence_length <= 0) {
        throw std::invalid_argument("all integral model dimensions must be positive");
    }
    if (attention_head_dimension < 0) {
        throw std::invalid_argument("attention head dimension cannot be negative");
    }
    if (attention_head_dimension == 0 && dimension % heads != 0) {
        throw std::invalid_argument("dimension must divide into heads when head dimension is derived");
    }
    if (heads % kv_heads != 0) throw std::invalid_argument("heads must divide into kv_heads groups");
    if (head_dimension() % 2 != 0) throw std::invalid_argument("RoPE head dimension must be even");
    if (!(rope_base > 0.0F)) throw std::invalid_argument("RoPE base must be positive");
    if (!std::isfinite(rms_norm_epsilon) || rms_norm_epsilon <= 0.0F) {
        throw std::invalid_argument("RMSNorm epsilon must be finite and positive");
    }
    if (linear_precision == LinearPrecision::Float8E4M3FNUZ &&
        (!std::isfinite(fp8_activation_scale) || fp8_activation_scale <= 0.0F ||
         !std::isfinite(fp8_activation_minimum_scale) ||
         fp8_activation_minimum_scale <= 0.0F ||
         !std::isfinite(fp8_weight_scale) || fp8_weight_scale <= 0.0F)) {
        throw std::invalid_argument("FP8 Linear scales must be finite and positive");
    }
    if (linear_precision != LinearPrecision::Float8E4M3FNUZ &&
        fp8_diagnostic_mode != Fp8DiagnosticMode::Full) {
        throw std::invalid_argument(
            "FP8 diagnostic mode requires FP8 Linear precision");
    }
    if (fp8_weight_scale_scope != Fp8WeightScaleScope::AllLinear &&
        (linear_precision != LinearPrecision::Float8E4M3FNUZ ||
         fp8_weight_scale_mode != Fp8WeightScaleMode::OutputChannelAmax)) {
        throw std::invalid_argument(
            "FP8 weight scale scope requires output-channel FP8 weights");
    }
    std::int64_t previous = -1;
    for (const auto layer : fp8_fp32_layers) {
        if (linear_precision != LinearPrecision::Float8E4M3FNUZ ||
            layer < 0 || layer >= layers || layer <= previous) {
            throw std::invalid_argument(
                "FP8 FP32 layers must be strictly increasing and in range");
        }
        previous = layer;
    }
}

std::int64_t ModelConfig::head_dimension() const {
    if (attention_head_dimension > 0) return attention_head_dimension;
    if (heads <= 0 || dimension % heads != 0) {
        throw std::invalid_argument("invalid head configuration");
    }
    return dimension / heads;
}

std::int64_t ModelConfig::kv_dimension() const {
    if (kv_heads <= 0) throw std::invalid_argument("kv_heads must be positive");
    return head_dimension() * kv_heads;
}

std::int64_t ModelConfig::query_dimension() const {
    if (heads <= 0 || head_dimension() >
            std::numeric_limits<std::int64_t>::max() / heads) {
        throw std::overflow_error("query dimension overflow");
    }
    return heads * head_dimension();
}

std::uint64_t ModelConfig::parameter_count() const {
    validate();
    const auto vocab = static_cast<std::uint64_t>(vocabulary_size);
    const auto dim = static_cast<std::uint64_t>(dimension);
    const auto layer_count = static_cast<std::uint64_t>(layers);
    const auto feed_forward = static_cast<std::uint64_t>(ffn_dimension);
    const auto key_value = static_cast<std::uint64_t>(kv_dimension());
    const auto query_width = static_cast<std::uint64_t>(query_dimension());

    const auto embedding = checked_product(vocab, dim);
    const auto query_and_output = checked_product(
        2, checked_product(dim, query_width));
    const auto key_and_value = checked_product(2, checked_product(dim, key_value));
    const auto attention = checked_add(query_and_output, key_and_value);
    const auto attention_biases = attention_bias
        ? checked_add(query_width, checked_product(2, key_value)) : 0;
    const auto qk_norm_parameters = qk_norm
        ? checked_product(2, static_cast<std::uint64_t>(head_dimension())) : 0;
    const auto ffn = checked_product(3, checked_product(dim, feed_forward));
    const auto norms = checked_product(2, dim);
    const auto per_layer = checked_add(
        checked_add(checked_add(checked_add(attention, attention_biases),
                                qk_norm_parameters), ffn), norms);
    auto total = checked_add(embedding, checked_product(layer_count, per_layer));
    total = checked_add(total, dim);  // final RMSNorm
    if (!tie_embeddings) total = checked_add(total, checked_product(dim, vocab));
    return total;
}

std::uint64_t ModelConfig::weight_bytes(std::uint64_t bytes_per_parameter) const {
    if (bytes_per_parameter == 0) throw std::invalid_argument("bytes per parameter must be positive");
    return checked_product(parameter_count(), bytes_per_parameter);
}

std::string ModelConfig::summary() const {
    std::ostringstream output;
    output << "vocab=" << vocabulary_size << ",dim=" << dimension << ",layers=" << layers
           << ",heads=" << heads << ",kv_heads=" << kv_heads << ",ffn=" << ffn_dimension
           << ",head_dim=" << head_dimension()
           << ",max_seq=" << max_sequence_length << ",rope_base=" << rope_base
           << ",tie_embeddings=" << (tie_embeddings ? "true" : "false")
           << ",linear_precision="
           << (linear_precision == LinearPrecision::Float32
                   ? "fp32"
                   : linear_precision == LinearPrecision::BFloat16
                         ? "bf16_fp32_master" : "fp8_e4m3_fnuz")
           << ",fp8_weight_scale_mode="
           << (fp8_weight_scale_mode == Fp8WeightScaleMode::Fixed
                   ? "fixed"
                   : fp8_weight_scale_mode == Fp8WeightScaleMode::TensorAmax
                         ? "tensor_amax"
                         : fp8_weight_scale_mode ==
                                   Fp8WeightScaleMode::DeviceTensorAmax
                               ? "device_tensor_amax" : "output_channel_amax")
           << ",fp8_weight_scale_scope="
           << (fp8_weight_scale_scope == Fp8WeightScaleScope::AllLinear
                   ? "all_linear" : "attention_output_only")
           << ",fp8_activation_scale_mode="
           << (fp8_activation_scale_mode == Fp8ActivationScaleMode::Fixed
                   ? "fixed"
                   : fp8_activation_scale_mode == Fp8ActivationScaleMode::TensorAmax
                         ? "tensor_amax" : "ffn_outer_row")
           << ",fp8_activation_minimum_scale=" << fp8_activation_minimum_scale
           << ",fp8_diagnostic_mode="
           << (fp8_diagnostic_mode == Fp8DiagnosticMode::Full
                   ? "full"
                   : fp8_diagnostic_mode == Fp8DiagnosticMode::WeightOnly
                         ? "weight_only"
                         : fp8_diagnostic_mode == Fp8DiagnosticMode::ActivationOnly
                               ? "activation_only" : "both_roundtrip")
           << ",fp8_fp32_layers=";
    for (std::size_t index = 0; index < fp8_fp32_layers.size(); ++index) {
        if (index != 0) output << ':';
        output << fp8_fp32_layers[index];
    }
    output
           << ",rms_eps=" << rms_norm_epsilon
           << ",attention_bias=" << (attention_bias ? "true" : "false")
           << ",qk_norm=" << (qk_norm ? "true" : "false")
           << ",rope_layout=" << (rope_layout == RopeLayout::Interleaved ? "interleaved" : "split_half")
           << ",parameters=" << parameter_count();
    return output.str();
}

ModelConfig ModelConfig::model_s() {
    return {.vocabulary_size = 8192,
            .dimension = 384,
            .layers = 6,
            .heads = 6,
            .kv_heads = 6,
            .ffn_dimension = 832,
            .max_sequence_length = 512,
            .rope_base = 10000.0F,
            .tie_embeddings = false};
}

ModelConfig ModelConfig::model_m() {
    return {.vocabulary_size = 8192,
            .dimension = 512,
            .layers = 8,
            .heads = 8,
            .kv_heads = 8,
            .ffn_dimension = 1184,
            .max_sequence_length = 512,
            .rope_base = 10000.0F,
            .tie_embeddings = false};
}

}  // namespace microllm::model
