#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace microllm::model {

enum class LinearPrecision { Float32, BFloat16, Float8E4M3FNUZ };
enum class Fp8WeightScaleMode {
    Fixed, TensorAmax, DeviceTensorAmax, OutputChannelAmax
};
enum class Fp8WeightScaleScope { AllLinear, AttentionOnly };
enum class Fp8ActivationScaleMode { Fixed, TensorAmax, FfnOuterRow };
// Full uses native FP8 GEMM. The other modes are deliberately slow,
// inference-only counterfactuals that isolate one source of quantization error.
enum class Fp8DiagnosticMode { Full, WeightOnly, ActivationOnly, BothRoundtrip };
enum class RopeLayout { Interleaved, SplitHalf };

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
    LinearPrecision linear_precision = LinearPrecision::Float32;
    float fp8_activation_scale = 0.025F;
    float fp8_activation_minimum_scale = 1.0e-4F;
    float fp8_weight_scale = 0.005F;
    Fp8WeightScaleMode fp8_weight_scale_mode = Fp8WeightScaleMode::Fixed;
    Fp8WeightScaleScope fp8_weight_scale_scope = Fp8WeightScaleScope::AllLinear;
    Fp8ActivationScaleMode fp8_activation_scale_mode =
        Fp8ActivationScaleMode::Fixed;
    Fp8DiagnosticMode fp8_diagnostic_mode = Fp8DiagnosticMode::Full;
    std::vector<std::int64_t> fp8_fp32_layers = {};
    float rms_norm_epsilon = 1.0e-5F;
    bool attention_bias = false;
    RopeLayout rope_layout = RopeLayout::Interleaved;

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
