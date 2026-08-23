#include <microllm/model/model.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <memory>
#include <limits>
#include <optional>
#include <random>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>

#include <microllm/ops/ops.h>
#include <microllm/profiling/trace.h>
#include <microllm/runtime/memory.h>
#include <microllm/runtime/runtime.h>

namespace microllm::model {
namespace {

using autograd::Value;

constexpr float kFp8E4M3FnuzMaximum = 240.0F;

float tensor_amax_scale(const Tensor& tensor, float zero_fallback) {
    const auto values = tensor.to_vector();
    float maximum = 0.0F;
    for (const auto value : values) {
        if (!std::isfinite(value)) {
            throw std::invalid_argument(
                "FP8 tensor-amax weight preparation requires finite values");
        }
        maximum = std::max(maximum, std::abs(value));
    }
    const auto result = maximum / kFp8E4M3FnuzMaximum;
    return result > 0.0F ? result : zero_fallback;
}

void trace_detail(const std::string& prefix, const char* suffix,
                  const Tensor& tensor) {
    if (prefix.empty()) return;
    auto* trace = profiling::TraceSession::current();
    if (trace != nullptr) {
        trace->record(profiling::TraceKind::Layer,
                      prefix + "." + suffix, tensor);
    }
}

Tensor clone_tensor(const Tensor& source, Device target) {
    const auto packed = source.is_contiguous() ? source : source.contiguous();
    Tensor copy(packed.shape(), packed.dtype(), target);
    runtime::copy_bytes(copy.data(), target, packed.data(), packed.device(),
                        static_cast<std::size_t>(packed.numel()) * dtype_size(packed.dtype()));
    return copy;
}

Tensor prepare_cached_sequence(Tensor& cached, const Tensor& current,
                               std::int64_t position, std::int64_t capacity,
                               DType cache_dtype) {
    if (current.dtype() != DType::Float32 || current.ndim() != 4) {
        throw std::invalid_argument("cached K/V tensors must be float32 rank four");
    }
    if (position < 0 || position >= capacity || current.shape()[0] <= 0 ||
        current.shape()[2] != 1) {
        throw std::out_of_range("cached K/V position is outside the preallocated capacity");
    }
    const auto packed = current.is_contiguous() ? current : current.contiguous();
    const auto heads = current.shape()[1];
    const auto width = current.shape()[3];
    if (!cached.defined()) {
        if (position != 0) throw std::invalid_argument("KV cache must start at position zero");
        Tensor backing({current.shape()[0], heads, capacity, width},
                       cache_dtype, current.device());
        cached = Tensor::from_storage(
            backing.storage(), {current.shape()[0], heads, 1, width},
            backing.strides(), 0, cache_dtype);
    } else if (cached.ndim() != 4 || cached.shape()[0] != current.shape()[0] ||
               cached.shape()[1] != heads || cached.shape()[2] != position ||
               cached.shape()[3] != width || cached.device() != current.device() ||
               cached.dtype() != cache_dtype) {
        throw std::invalid_argument("cached and current K/V shapes are incompatible");
    } else {
        cached = Tensor::from_storage(
            cached.storage(), {current.shape()[0], heads, position + 1, width},
            cached.strides(), cached.storage_offset(), cached.dtype());
    }
    return packed;
}

Tensor prepare_cached_active_sequence(
    Tensor& cached, const Tensor& current, std::int64_t cache_batches,
    const std::vector<std::int64_t>& positions, std::int64_t capacity,
    DType cache_dtype) {
    if (current.dtype() != DType::Float32 || current.ndim() != 4 ||
        current.shape()[0] != static_cast<std::int64_t>(positions.size()) ||
        current.shape()[0] <= 0 || current.shape()[2] != 1 ||
        cache_batches <= 0) {
        throw std::invalid_argument(
            "active cached K/V tensors must be FP32 [A,H,1,D]");
    }
    const auto maximum = *std::max_element(positions.begin(), positions.end());
    if (positions.empty() || maximum < 0 || maximum >= capacity ||
        std::any_of(positions.begin(), positions.end(),
                    [capacity](std::int64_t position) {
                        return position < 0 || position >= capacity;
                    })) {
        throw std::out_of_range("active cached K/V positions exceed capacity");
    }
    const auto packed = current.is_contiguous() ? current : current.contiguous();
    const auto heads = current.shape()[1];
    const auto width = current.shape()[3];
    const auto target_prefix = maximum + 1;
    if (!cached.defined()) {
        if (maximum != 0) {
            throw std::invalid_argument(
                "nonzero active positions require initialized KV storage");
        }
        Tensor backing({cache_batches, heads, capacity, width}, cache_dtype,
                       current.device());
        cached = Tensor::from_storage(
            backing.storage(), {cache_batches, heads, target_prefix, width},
            backing.strides(), 0, cache_dtype);
    } else {
        if (cached.ndim() != 4 || cached.shape()[0] != cache_batches ||
            cached.shape()[1] != heads || cached.shape()[3] != width ||
            cached.shape()[2] > capacity || cached.device() != current.device() ||
            cached.dtype() != cache_dtype || cached.stride(3) != 1 ||
            cached.stride(2) != width ||
            cached.stride(1) != capacity * width ||
            cached.stride(0) != heads * capacity * width) {
            throw std::invalid_argument(
                "active cached and current K/V layouts are incompatible");
        }
        cached = Tensor::from_storage(
            cached.storage(),
            {cache_batches, heads,
             std::max(cached.shape()[2], target_prefix), width},
            cached.strides(), cached.storage_offset(), cached.dtype());
    }
    return packed;
}

void prepare_cached_prefix(Tensor& cached, const Tensor& current,
                           std::int64_t capacity, DType cache_dtype) {
    if (current.dtype() != DType::Float32 || current.ndim() != 4 ||
        current.shape()[0] <= 0 || current.shape()[2] <= 0 ||
        current.shape()[2] > capacity || cached.defined()) {
        throw std::invalid_argument(
            "KV prefix requires an empty cache and a sequence within capacity");
    }
    const auto contiguous = current.is_contiguous() ? current : current.contiguous();
    const auto packed = contiguous.dtype() == cache_dtype
                            ? contiguous
                            : ops::cast(contiguous, cache_dtype);
    Tensor backing({current.shape()[0], current.shape()[1], capacity,
                    current.shape()[3]},
                   cache_dtype, current.device());
    cached = Tensor::from_storage(
        backing.storage(), current.shape(), backing.strides(), 0, cache_dtype);
    const auto batches = static_cast<std::size_t>(current.shape()[0]);
    const auto heads = static_cast<std::size_t>(current.shape()[1]);
    const auto sequence = static_cast<std::size_t>(current.shape()[2]);
    const auto width = static_cast<std::size_t>(current.shape()[3]);
    const auto element_bytes = dtype_size(cache_dtype);
    const auto head_bytes = sequence * width * element_bytes;
    auto* destination = static_cast<std::byte*>(cached.storage().data());
    const auto* source = static_cast<const std::byte*>(packed.data());
    for (std::size_t batch = 0; batch < batches; ++batch) {
        for (std::size_t head = 0; head < heads; ++head) {
            const auto instance = batch * heads + head;
            runtime::copy_bytes(
                destination + instance * static_cast<std::size_t>(capacity) *
                                  width * element_bytes,
                cached.device(), source + instance * head_bytes,
                packed.device(), head_bytes);
        }
    }
}

void ensure_batched_cache_tensor(Tensor& cached, std::int64_t batches,
                                 std::int64_t heads, std::int64_t prefix,
                                 std::int64_t capacity, std::int64_t width,
                                 DType dtype, Device device) {
    if (!cached.defined()) {
        if (prefix != 0) {
            throw std::invalid_argument(
                "non-empty row positions require initialized KV storage");
        }
        Tensor backing({batches, heads, capacity, width}, dtype, device);
        cached = Tensor::from_storage(
            backing.storage(), {batches, heads, 0, width}, backing.strides(), 0,
            dtype);
        return;
    }
    if (cached.ndim() != 4 || cached.shape()[0] != batches ||
        cached.shape()[1] != heads || cached.shape()[3] != width ||
        cached.shape()[2] > capacity ||
        cached.dtype() != dtype || cached.device() != device ||
        cached.stride(3) != 1 || cached.stride(2) != width ||
        cached.stride(1) != capacity * width ||
        cached.stride(0) != heads * capacity * width || prefix > capacity) {
        throw std::invalid_argument("divergent-row KV storage is incompatible");
    }
    cached = Tensor::from_storage(
        cached.storage(), {batches, heads, prefix, width}, cached.strides(),
        cached.storage_offset(), dtype);
}

Tensor cache_row_view(const Tensor& cached, std::int64_t row,
                      std::int64_t prefix) {
    return Tensor::from_storage(
        cached.storage(), {1, cached.shape()[1], prefix, cached.shape()[3]},
        cached.strides(), cached.storage_offset() + row * cached.stride(0),
        cached.dtype());
}

void copy_cache_prefix_to_row(Tensor& destination, const Tensor& source,
                              std::int64_t row, std::int64_t prefix) {
    if (!destination.defined() || !source.defined() ||
        destination.ndim() != 4 || source.ndim() != 4 ||
        source.shape()[0] != 1 || destination.shape()[1] != source.shape()[1] ||
        destination.shape()[3] != source.shape()[3] ||
        source.shape()[2] != prefix || destination.dtype() != source.dtype() ||
        destination.device() != source.device() || row < 0 ||
        row >= destination.shape()[0]) {
        throw std::invalid_argument("row prefill cache copy is incompatible");
    }
    const auto bytes = static_cast<std::size_t>(
        prefix * source.shape()[3]) * dtype_size(source.dtype());
    auto* destination_data = static_cast<std::byte*>(destination.data());
    const auto* source_data = static_cast<const std::byte*>(source.data());
    const auto element_bytes = dtype_size(source.dtype());
    for (std::int64_t head = 0; head < source.shape()[1]; ++head) {
        runtime::copy_bytes(
            destination_data + static_cast<std::size_t>(
                                   row * destination.stride(0) +
                                   head * destination.stride(1)) *
                                   element_bytes,
            destination.device(),
            source_data + static_cast<std::size_t>(head * source.stride(1)) *
                              element_bytes,
            source.device(), bytes);
    }
}

Tensor random_tensor(Shape shape, std::mt19937_64& generator, float standard_deviation) {
    std::normal_distribution<float> distribution(0.0F, standard_deviation);
    std::vector<float> values(static_cast<std::size_t>(checked_numel(shape)));
    for (auto& value : values) value = distribution(generator);
    return Tensor::from_vector(values, std::move(shape));
}

template <typename Predicate>
Bf16WeightPreparationReport prepare_bf16_weights(
    const NamedValues& named, std::size_t expected_count, Device device,
    Predicate&& selected) {
    struct Prepared {
        Value* parameter = nullptr;
        Tensor bf16;
    };
    std::vector<Prepared> prepared;
    Bf16WeightPreparationReport report;
    for (const auto& [name, parameter] : named) {
        if (!selected(name)) continue;
        if (parameter->data().dtype() != DType::Float32 ||
            !parameter->data().is_contiguous()) {
            throw std::logic_error(
                "BF16 inference preparation requires contiguous FP32 source weights");
        }
        const auto elements = static_cast<std::uint64_t>(parameter->data().numel());
        prepared.push_back({parameter, ops::cast(parameter->data(), DType::BFloat16)});
        ++report.converted_tensors;
        report.fp32_bytes_released += elements * sizeof(float);
        report.bf16_bytes_retained += elements * sizeof(std::uint16_t);
    }
    if (report.converted_tensors != expected_count) {
        throw std::logic_error("model exposed an unexpected BF16 inference weight count");
    }
    runtime::synchronize(device);
    for (auto& item : prepared) {
        *item.parameter = Value(std::move(item.bf16), false);
    }
    return report;
}

Value parameter(Shape shape, std::mt19937_64& generator, float standard_deviation,
                ParameterInitialization initialization) {
    return Value(initialization == ParameterInitialization::Random
                     ? random_tensor(std::move(shape), generator, standard_deviation)
                     : Tensor(std::move(shape)),
                 true);
}

class Linear {
public:
    Linear(std::int64_t input, std::int64_t output, std::mt19937_64& generator,
           const ModelConfig& config, ParameterInitialization initialization,
           bool with_bias = false, bool ffn_linear = false,
           bool attention_linear = false)
        : weight_(parameter({input, output}, generator,
                            1.0F / std::sqrt(static_cast<float>(input)), initialization)),
          precision_(config.linear_precision),
          activation_scale_(config.fp8_activation_scale),
          activation_minimum_scale_(config.fp8_activation_minimum_scale),
          weight_scale_(config.fp8_weight_scale),
          weight_scale_mode_(
              config.fp8_weight_scale_mode ==
                          Fp8WeightScaleMode::OutputChannelAmax &&
                      config.fp8_weight_scale_scope ==
                          Fp8WeightScaleScope::AttentionOnly &&
                      !attention_linear
                  ? Fp8WeightScaleMode::DeviceTensorAmax
                  : config.fp8_weight_scale_mode),
          diagnostic_mode_(config.fp8_diagnostic_mode),
          activation_scale_mode_(
              config.fp8_activation_scale_mode ==
                      Fp8ActivationScaleMode::FfnOuterRow
                  ? ffn_linear ? Fp8ActivationScaleMode::FfnOuterRow
                               : Fp8ActivationScaleMode::Fixed
                  : config.fp8_activation_scale_mode),
          has_bias_(with_bias) {
        if (has_bias_) {
            bias_ = Value(Tensor({output}), true);
            if (initialization == ParameterInitialization::Random) {
                bias_.mutable_data().fill(0.0F);
            }
        }
    }

    Value forward_without_bias(const Value& input) {
        if (precision_ == LinearPrecision::BFloat16) {
            return bf16_training_weight_.defined()
                       ? autograd::bf16_matmul(input, weight_, bf16_training_weight_)
                       : autograd::bf16_matmul(input, weight_);
        }
        if (precision_ == LinearPrecision::Float8E4M3FNUZ) {
            if (diagnostic_mode_ != Fp8DiagnosticMode::Full) {
                throw std::logic_error(
                    "FP8 diagnostic modes are graph-free inference-only");
            }
            if (weight_scale_mode_ != Fp8WeightScaleMode::Fixed ||
                activation_scale_mode_ != Fp8ActivationScaleMode::Fixed) {
                throw std::logic_error(
                    "FP8 tensor-amax scale is inference-only");
            }
            return autograd::fp8_matmul(input, weight_, activation_scale_, weight_scale_);
        }
        return autograd::matmul(input, weight_);
    }
    Value forward(const Value& input) {
        auto output = forward_without_bias(input);
        return has_bias_ ? autograd::add_bias(output, bias_) : output;
    }
    [[nodiscard]] bool shares_dynamic_activation() const noexcept {
        return precision_ == LinearPrecision::Float8E4M3FNUZ &&
               diagnostic_mode_ != Fp8DiagnosticMode::WeightOnly &&
               activation_scale_mode_ != Fp8ActivationScaleMode::Fixed;
    }
    [[nodiscard]] ops::ScaledTensor quantize_activation(const Tensor& input) const {
        if (!shares_dynamic_activation()) {
            throw std::logic_error(
                "shared FP8 activation requires a dynamic scale mode");
        }
        return activation_scale_mode_ == Fp8ActivationScaleMode::FfnOuterRow
                   ? ops::quantize_fp8_rows_dynamic(
                         input, DType::Float8E4M3FNUZ,
                         activation_minimum_scale_)
                   : ops::quantize_fp8_dynamic(
                         input, DType::Float8E4M3FNUZ,
                         activation_minimum_scale_);
    }
    Tensor forward_scaled_input_without_bias(
        const ops::ScaledTensor& scaled_input) {
        if (precision_ != LinearPrecision::Float8E4M3FNUZ) {
            throw std::logic_error(
                "scaled FP8 input requires an FP8 Linear");
        }
        if (diagnostic_mode_ == Fp8DiagnosticMode::WeightOnly) {
            throw std::logic_error(
                "weight-only FP8 diagnostic does not quantize activations");
        }
        if (diagnostic_mode_ == Fp8DiagnosticMode::ActivationOnly) {
            return ops::matmul_with_implementation(
                ops::dequantize_fp8(scaled_input, DType::Float32),
                weight_.data(), ops::MatmulImplementation::Auto);
        }
        if (diagnostic_mode_ == Fp8DiagnosticMode::BothRoundtrip) {
            return ops::matmul_with_implementation(
                ops::dequantize_fp8(scaled_input, DType::Float32),
                ops::dequantize_fp8(scaled_weight(), DType::Float32),
                ops::MatmulImplementation::Auto);
        }
        return ops::fp8_matmul(scaled_input, scaled_weight(), DType::Float32);
    }
    [[nodiscard]] ops::ScaledTensor scaled_weight() const {
        ops::ScaledTensor result;
        if (fp8_inference_scale_.defined()) {
            result = {weight_.data(), fp8_inference_scale_, weight_scale_,
                      fp8_inference_host_scale_available_,
                      fp8_inference_scale_mode_};
        } else if (weight_scale_mode_ ==
                   Fp8WeightScaleMode::OutputChannelAmax) {
            result = ops::quantize_fp8_columns_dynamic(
                weight_.data(), DType::Float8E4M3FNUZ, weight_scale_);
        } else if (weight_scale_mode_ == Fp8WeightScaleMode::DeviceTensorAmax) {
            result = ops::quantize_fp8_dynamic(
                weight_.data(), DType::Float8E4M3FNUZ, weight_scale_);
        } else {
            const auto lazy_weight_scale =
                weight_scale_mode_ == Fp8WeightScaleMode::TensorAmax
                    ? tensor_amax_scale(weight_.data(), weight_scale_)
                    : weight_scale_;
            result = ops::quantize_fp8(
                weight_.data(), DType::Float8E4M3FNUZ, lazy_weight_scale);
        }
        return result;
    }
    Tensor forward_scaled_input(const ops::ScaledTensor& scaled_input) {
        auto output = forward_scaled_input_without_bias(scaled_input);
        return has_bias_ ? ops::add_bias(output, bias_.data()) : output;
    }
    Tensor forward_tensor_without_bias(const Tensor& input) {
        if (precision_ == LinearPrecision::BFloat16) {
            return ops::bf16_matmul(
                input, bf16_training_weight_.defined()
                           ? bf16_training_weight_
                           : ops::cast(weight_.data(), DType::BFloat16));
        }
        if (precision_ == LinearPrecision::Float8E4M3FNUZ) {
            if (diagnostic_mode_ == Fp8DiagnosticMode::WeightOnly) {
                return ops::matmul_with_implementation(
                    input,
                    ops::dequantize_fp8(scaled_weight(), DType::Float32),
                    ops::MatmulImplementation::Auto);
            }
            ops::ScaledTensor scaled_input;
            if (activation_scale_mode_ == Fp8ActivationScaleMode::TensorAmax) {
                scaled_input = ops::quantize_fp8_dynamic(
                    input, DType::Float8E4M3FNUZ, activation_minimum_scale_);
            } else if (activation_scale_mode_ ==
                       Fp8ActivationScaleMode::FfnOuterRow) {
                scaled_input = ops::quantize_fp8_rows_dynamic(
                    input, DType::Float8E4M3FNUZ, activation_minimum_scale_);
            } else if (fp8_inference_activation_scale_.defined()) {
                scaled_input = ops::quantize_fp8_with_scale(
                    input, DType::Float8E4M3FNUZ, activation_scale_,
                    fp8_inference_activation_scale_);
            } else {
                scaled_input = ops::quantize_fp8(
                    input, DType::Float8E4M3FNUZ, activation_scale_);
            }
            return forward_scaled_input_without_bias(scaled_input);
        }
        if (weight_.data().dtype() == DType::BFloat16) {
            return ops::bf16_matmul(input, weight_.data());
        }
        return ops::matmul_with_implementation(input, weight_.data(),
                                               ops::MatmulImplementation::Auto);
    }
    Tensor forward_tensor(const Tensor& input) {
        auto output = forward_tensor_without_bias(input);
        return has_bias_ ? ops::add_bias(output, bias_.data()) : output;
    }
    Value& weight() noexcept { return weight_; }
    [[nodiscard]] const Tensor& weight_data() const noexcept { return weight_.data(); }
    [[nodiscard]] bool has_bias() const noexcept { return has_bias_; }
    [[nodiscard]] bool is_fp8() const noexcept {
        return precision_ == LinearPrecision::Float8E4M3FNUZ;
    }
    Value& bias() noexcept { return bias_; }
    std::pair<Value*, Tensor*> prepare_bf16_training_mirror() {
        if (precision_ != LinearPrecision::BFloat16 ||
            weight_.data().dtype() != DType::Float32 ||
            bf16_training_weight_.defined()) {
            throw std::logic_error("Linear BF16 training mirror preparation is invalid");
        }
        bf16_training_weight_ = ops::cast(weight_.data(), DType::BFloat16);
        return {&weight_, &bf16_training_weight_};
    }
    void move_bf16_training_mirror(Device device) {
        if (bf16_training_weight_.defined()) {
            bf16_training_weight_ = bf16_training_weight_.to(device);
        }
    }
    [[nodiscard]] ops::ScaledTensor prepare_fp8_inference_candidate() const {
        if (precision_ != LinearPrecision::Float8E4M3FNUZ ||
            diagnostic_mode_ == Fp8DiagnosticMode::ActivationOnly ||
            weight_.data().dtype() != DType::Float32 ||
            fp8_inference_scale_.defined()) {
            throw std::logic_error("Linear FP8 inference preparation is invalid");
        }
        if (weight_scale_mode_ == Fp8WeightScaleMode::DeviceTensorAmax) {
            return ops::quantize_fp8_dynamic(
                weight_.data(), DType::Float8E4M3FNUZ, weight_scale_);
        }
        if (weight_scale_mode_ == Fp8WeightScaleMode::OutputChannelAmax) {
            return ops::quantize_fp8_columns_dynamic(
                weight_.data(), DType::Float8E4M3FNUZ, weight_scale_);
        }
        const auto scale = weight_scale_mode_ == Fp8WeightScaleMode::TensorAmax
                               ? tensor_amax_scale(weight_.data(), weight_scale_)
                               : weight_scale_;
        return ops::quantize_fp8(
            weight_.data(), DType::Float8E4M3FNUZ, scale);
    }
    [[nodiscard]] Tensor prepare_fp8_activation_scale_candidate() const {
        if (diagnostic_mode_ == Fp8DiagnosticMode::WeightOnly ||
            activation_scale_mode_ != Fp8ActivationScaleMode::Fixed) {
            return {};
        }
        auto result = Tensor::from_vector(
            {activation_scale_}, {}, DType::Float32);
        return weight_.data().device().is_hip()
                   ? result.to(weight_.data().device())
                   : result;
    }
    void commit_fp8_inference_candidate(
        ops::ScaledTensor candidate, Tensor activation_scale) {
        if (diagnostic_mode_ == Fp8DiagnosticMode::ActivationOnly) {
            throw std::logic_error(
                "activation-only diagnostic cannot replace FP32 weights");
        }
        weight_scale_ = candidate.scale_value;
        fp8_inference_host_scale_available_ = candidate.host_scale_available;
        fp8_inference_scale_mode_ = candidate.scale_mode;
        fp8_inference_scale_ = std::move(candidate.scale);
        fp8_inference_activation_scale_ = std::move(activation_scale);
        weight_ = Value(std::move(candidate.values), false);
    }
    void commit_fp8_activation_only_candidate(Tensor activation_scale) {
        if (diagnostic_mode_ != Fp8DiagnosticMode::ActivationOnly ||
            weight_.data().dtype() != DType::Float32 ||
            fp8_inference_scale_.defined() ||
            fp8_inference_activation_scale_.defined()) {
            throw std::logic_error(
                "activation-only FP8 inference preparation is invalid");
        }
        fp8_inference_activation_scale_ = std::move(activation_scale);
    }
    void move_fp8_inference_scale(Device device) {
        if (fp8_inference_scale_.defined()) {
            fp8_inference_scale_ = fp8_inference_scale_.to(device);
        }
        if (fp8_inference_activation_scale_.defined()) {
            fp8_inference_activation_scale_ =
                fp8_inference_activation_scale_.to(device);
        }
    }

private:
    Value weight_;
    LinearPrecision precision_ = LinearPrecision::Float32;
    float activation_scale_ = 1.0F;
    float activation_minimum_scale_ = 1.0e-4F;
    float weight_scale_ = 1.0F;
    Fp8WeightScaleMode weight_scale_mode_ = Fp8WeightScaleMode::Fixed;
    Fp8DiagnosticMode diagnostic_mode_ = Fp8DiagnosticMode::Full;
    Fp8ActivationScaleMode activation_scale_mode_ =
        Fp8ActivationScaleMode::Fixed;
    bool has_bias_ = false;
    Value bias_;
    Tensor bf16_training_weight_;
    Tensor fp8_inference_scale_;
    bool fp8_inference_host_scale_available_ = true;
    ops::Fp8ScaleMode fp8_inference_scale_mode_ = ops::Fp8ScaleMode::Scalar;
    Tensor fp8_inference_activation_scale_;
};

class Norm {
public:
    explicit Norm(std::int64_t dimension, float epsilon,
                  ParameterInitialization initialization)
        : weight_(Tensor({dimension}), true), epsilon_(epsilon) {
        if (initialization == ParameterInitialization::Random) {
            weight_.mutable_data().fill(1.0F);
        }
    }

    Value forward(const Value& input) { return autograd::rms_norm(input, weight_, epsilon_); }
    Tensor forward_tensor(const Tensor& input) {
        return ops::rms_norm(input, weight_.data(), epsilon_);
    }
    ops::TensorPair add_forward_tensor(const Tensor& left, const Tensor& right) {
        return ops::add_rms_norm(left, right, weight_.data(), epsilon_);
    }
    Value& weight() noexcept { return weight_; }

private:
    Value weight_;
    float epsilon_ = 1.0e-5F;
};

class Attention {
public:
    Attention(const ModelConfig& config, std::mt19937_64& generator,
              ParameterInitialization initialization)
        : config_(config),
          query_(config.dimension, config.dimension, generator, config, initialization,
                 config.attention_bias, false, true),
          key_(config.dimension, config.kv_dimension(), generator, config, initialization,
               config.attention_bias, false, true),
          value_(config.dimension, config.kv_dimension(), generator, config, initialization,
                 config.attention_bias, false, true),
          output_(config.dimension, config.dimension, generator, config, initialization,
                  false, false, true) {}

    Value forward(const Value& input) {
        if (input.data().ndim() != 3) throw std::invalid_argument("attention input must be BxTxD");
        const auto batch = input.data().shape()[0];
        const auto sequence = input.data().shape()[1];
        const auto flat = autograd::reshape(input, {batch * sequence, config_.dimension});
        const auto fuse_query_bias = config_.rope_layout == RopeLayout::SplitHalf &&
                                     query_.has_bias();
        const auto fuse_key_bias = config_.rope_layout == RopeLayout::SplitHalf &&
                                   key_.has_bias();
        auto query = autograd::reshape(
            fuse_query_bias ? query_.forward_without_bias(flat) : query_.forward(flat),
                                       {batch, sequence, config_.heads, config_.head_dimension()});
        auto key = autograd::reshape(
            fuse_key_bias ? key_.forward_without_bias(flat) : key_.forward(flat),
                                     {batch, sequence, config_.kv_heads, config_.head_dimension()});
        auto value = autograd::reshape(value_.forward(flat),
                                       {batch, sequence, config_.kv_heads, config_.head_dimension()});
        const auto transposed_query = autograd::transpose(query, 1, 2);
        const auto transposed_key = autograd::transpose(key, 1, 2);
        if (config_.rope_layout == RopeLayout::SplitHalf) {
            query = fuse_query_bias
                        ? autograd::rope_split_half_bias(
                              transposed_query, query_.bias(), 0, config_.rope_base)
                        : autograd::rope_split_half(
                              transposed_query, 2, 0, config_.rope_base);
            key = fuse_key_bias
                      ? autograd::rope_split_half_bias(
                            transposed_key, key_.bias(), 0, config_.rope_base)
                      : autograd::rope_split_half(
                            transposed_key, 2, 0, config_.rope_base);
        } else {
            query = autograd::rope(transposed_query, 2, 0, config_.rope_base);
            key = autograd::rope(transposed_key, 2, 0, config_.rope_base);
        }
        value = autograd::transpose(value, 1, 2);
        const auto repeats = config_.heads / config_.kv_heads;
        auto context = autograd::causal_gqa_attention(
            query, key, value, repeats,
            1.0F / std::sqrt(static_cast<float>(config_.head_dimension())));
        context = autograd::contiguous(autograd::transpose(context, 1, 2));
        context = autograd::reshape(context, {batch * sequence, config_.dimension});
        return autograd::reshape(output_.forward(context),
                                 {batch, sequence, config_.dimension});
    }

    Tensor forward_tensor(
        const Tensor& input,
        inference::KVCache::LayerState* prefill_cache = nullptr,
        std::int64_t cache_capacity = 0,
        DType cache_dtype = DType::Float32,
        const std::string& trace_prefix = {}) {
        if (input.ndim() != 3) throw std::invalid_argument("attention input must be BxTxD");
        const auto batch = input.shape()[0];
        const auto sequence = input.shape()[1];
        const auto flat = input.reshape({batch * sequence, config_.dimension});
        Tensor query_projection;
        Tensor key_projection;
        Tensor value_projection;
        if (query_.weight_data().dtype() == DType::BFloat16) {
            const auto projections = ops::bf16_qkv_projection(
                flat, query_.weight_data(), key_.weight_data(), value_.weight_data());
            query_projection = query_.has_bias()
                                   ? ops::add_bias(projections.first, query_.bias().data())
                                   : projections.first;
            key_projection = key_.has_bias()
                                 ? ops::add_bias(projections.second, key_.bias().data())
                                 : projections.second;
            value_projection = value_.has_bias()
                                   ? ops::add_bias(projections.third, value_.bias().data())
                                   : projections.third;
        } else if (query_.shares_dynamic_activation() &&
                   key_.shares_dynamic_activation() &&
                   value_.shares_dynamic_activation()) {
            const auto scaled = query_.quantize_activation(flat);
            query_projection = query_.forward_scaled_input(scaled);
            key_projection = key_.forward_scaled_input(scaled);
            value_projection = value_.forward_scaled_input(scaled);
        } else {
            query_projection = query_.forward_tensor(flat);
            key_projection = key_.forward_tensor(flat);
            value_projection = value_.forward_tensor(flat);
        }
        trace_detail(trace_prefix, "q_projection", query_projection);
        trace_detail(trace_prefix, "k_projection", key_projection);
        trace_detail(trace_prefix, "v_projection", value_projection);
        auto query = query_projection
                         .reshape({batch, sequence, config_.heads, config_.head_dimension()})
                         .transpose(1, 2)
                         .contiguous();
        auto key = key_projection
                       .reshape({batch, sequence, config_.kv_heads,
                                 config_.head_dimension()})
                       .transpose(1, 2)
                       .contiguous();
        auto value = value_projection
                         .reshape({batch, sequence, config_.kv_heads,
                                   config_.head_dimension()})
                         .transpose(1, 2)
                         .contiguous();
        if (config_.rope_layout == RopeLayout::SplitHalf) {
            query = ops::rope_split_half(query, 2, 0, config_.rope_base);
            key = ops::rope_split_half(key, 2, 0, config_.rope_base);
        } else {
            query = ops::rope(query, 2, 0, config_.rope_base);
            key = ops::rope(key, 2, 0, config_.rope_base);
        }
        trace_detail(trace_prefix, "q_rope", query);
        trace_detail(trace_prefix, "k_rope", key);
        trace_detail(trace_prefix, "value", value);
        if (prefill_cache != nullptr) {
            prepare_cached_prefix(prefill_cache->key, key, cache_capacity,
                                  cache_dtype);
            prepare_cached_prefix(prefill_cache->value, value, cache_capacity,
                                  cache_dtype);
        }
        const auto repeats = config_.heads / config_.kv_heads;
        auto context = ops::causal_gqa_attention(
                           query, key, value, repeats,
                           1.0F / std::sqrt(
                                      static_cast<float>(config_.head_dimension())))
                           .transpose(1, 2)
                           .contiguous()
                           .reshape({batch * sequence, config_.dimension});
        trace_detail(trace_prefix, "context", context);
        auto output = output_.forward_tensor(context).reshape(
            {batch, sequence, config_.dimension});
        trace_detail(trace_prefix, "output", output);
        return output;
    }

    Tensor forward_cached(const Tensor& input, inference::KVCache::LayerState& cache,
                          std::int64_t position, std::int64_t cache_capacity,
                          DType cache_dtype) {
        if (input.shape().size() != 3 || input.shape()[0] <= 0 ||
            input.shape()[1] != 1) {
            throw std::invalid_argument("cached attention expects a non-empty Bx1 token step");
        }
        const auto batch = input.shape()[0];
        const auto flat = input.reshape({batch, config_.dimension});
        const auto fuse_query_bias = config_.rope_layout == RopeLayout::SplitHalf &&
                                     query_.has_bias();
        const auto fuse_key_bias = config_.rope_layout == RopeLayout::SplitHalf &&
                                   key_.has_bias();
        Tensor query_projection;
        Tensor key_projection;
        Tensor value_projection;
        if (query_.weight_data().dtype() == DType::BFloat16) {
            const auto projections = ops::bf16_qkv_projection(
                flat, query_.weight_data(), key_.weight_data(), value_.weight_data());
            query_projection = fuse_query_bias
                                   ? projections.first
                                   : query_.has_bias()
                                         ? ops::add_bias(projections.first, query_.bias().data())
                                         : projections.first;
            key_projection = fuse_key_bias
                                 ? projections.second
                                 : key_.has_bias()
                                       ? ops::add_bias(projections.second, key_.bias().data())
                                       : projections.second;
            value_projection = value_.has_bias()
                                   ? ops::add_bias(projections.third, value_.bias().data())
                                   : projections.third;
        } else {
            query_projection = fuse_query_bias ? query_.forward_tensor_without_bias(flat)
                                                : query_.forward_tensor(flat);
            key_projection = fuse_key_bias ? key_.forward_tensor_without_bias(flat)
                                            : key_.forward_tensor(flat);
            value_projection = value_.forward_tensor(flat);
        }
        auto query = query_projection
                         .reshape({batch, 1, config_.heads, config_.head_dimension()})
                         .transpose(1, 2);
        auto key = key_projection
                       .reshape({batch, 1, config_.kv_heads, config_.head_dimension()})
                       .transpose(1, 2);
        auto value = value_projection
                         .reshape({batch, 1, config_.kv_heads, config_.head_dimension()})
                         .transpose(1, 2);
        if (config_.rope_layout == RopeLayout::SplitHalf) {
            query = fuse_query_bias
                        ? ops::rope_split_half_bias(
                              query, query_.bias().data(), position, config_.rope_base)
                        : ops::rope_split_half(query, 2, position, config_.rope_base);
            key = fuse_key_bias
                      ? ops::rope_split_half_bias(
                            key, key_.bias().data(), position, config_.rope_base)
                      : ops::rope_split_half(key, 2, position, config_.rope_base);
        } else {
            query = ops::rope(query, 2, position, config_.rope_base);
            key = ops::rope(key, 2, position, config_.rope_base);
        }
        const auto packed_key =
            prepare_cached_sequence(cache.key, key, position, cache_capacity,
                                    cache_dtype);
        const auto packed_value =
            prepare_cached_sequence(cache.value, value, position, cache_capacity,
                                    cache_dtype);
        ops::kv_cache_store_pair_(cache.key, cache.value, packed_key, packed_value,
                                  position);
        const auto repeats = config_.heads / config_.kv_heads;
        auto context = ops::cached_gqa_attention(
                           query, cache.key, cache.value, repeats,
                           1.0F / std::sqrt(static_cast<float>(config_.head_dimension())))
                           .transpose(1, 2)
                           .contiguous()
                           .reshape({batch, config_.dimension});
        return output_.forward_tensor(context).reshape(
            {batch, 1, config_.dimension});
    }

    Tensor forward_cached_positions(
        const Tensor& input, inference::KVCache::LayerState& cache,
        const Tensor& positions, const Tensor& cache_rows,
        const std::vector<std::int64_t>& host_positions,
        std::int64_t cache_batches, std::int64_t cache_capacity,
        DType cache_dtype) {
        if (input.shape().size() != 3 || input.shape()[0] <= 0 ||
            input.shape()[1] != 1 ||
            positions.shape() != Shape({input.shape()[0]}) ||
            cache_rows.shape() != positions.shape()) {
            throw std::invalid_argument(
                "positioned cached attention expects active Bx1 rows");
        }
        const auto batch = input.shape()[0];
        const auto flat = input.reshape({batch, config_.dimension});
        const auto fuse_query_bias = config_.rope_layout == RopeLayout::SplitHalf &&
                                     query_.has_bias();
        const auto fuse_key_bias = config_.rope_layout == RopeLayout::SplitHalf &&
                                   key_.has_bias();
        Tensor query_projection;
        Tensor key_projection;
        Tensor value_projection;
        if (query_.weight_data().dtype() == DType::BFloat16) {
            const auto projections = ops::bf16_qkv_projection(
                flat, query_.weight_data(), key_.weight_data(),
                value_.weight_data());
            query_projection = fuse_query_bias
                                   ? projections.first
                                   : query_.has_bias()
                                         ? ops::add_bias(projections.first,
                                                         query_.bias().data())
                                         : projections.first;
            key_projection = fuse_key_bias
                                 ? projections.second
                                 : key_.has_bias()
                                       ? ops::add_bias(projections.second,
                                                       key_.bias().data())
                                       : projections.second;
            value_projection = value_.has_bias()
                                   ? ops::add_bias(projections.third,
                                                   value_.bias().data())
                                   : projections.third;
        } else {
            query_projection = fuse_query_bias
                                   ? query_.forward_tensor_without_bias(flat)
                                   : query_.forward_tensor(flat);
            key_projection = fuse_key_bias
                                 ? key_.forward_tensor_without_bias(flat)
                                 : key_.forward_tensor(flat);
            value_projection = value_.forward_tensor(flat);
        }
        auto query = query_projection
                         .reshape({batch, 1, config_.heads,
                                   config_.head_dimension()})
                         .transpose(1, 2);
        auto key = key_projection
                       .reshape({batch, 1, config_.kv_heads,
                                 config_.head_dimension()})
                       .transpose(1, 2);
        auto value = value_projection
                         .reshape({batch, 1, config_.kv_heads,
                                   config_.head_dimension()})
                         .transpose(1, 2);
        if (config_.rope_layout == RopeLayout::SplitHalf) {
            query = fuse_query_bias
                        ? ops::rope_split_half_bias_positions(
                              query, query_.bias().data(), positions,
                              config_.rope_base)
                        : ops::rope_split_half_positions(
                              query, positions, config_.rope_base);
            key = fuse_key_bias
                      ? ops::rope_split_half_bias_positions(
                            key, key_.bias().data(), positions,
                            config_.rope_base)
                      : ops::rope_split_half_positions(
                            key, positions, config_.rope_base);
        } else {
            query = ops::rope_positions(query, positions, config_.rope_base);
            key = ops::rope_positions(key, positions, config_.rope_base);
        }
        const auto packed_key = prepare_cached_active_sequence(
            cache.key, key, cache_batches, host_positions, cache_capacity,
            cache_dtype);
        const auto packed_value = prepare_cached_active_sequence(
            cache.value, value, cache_batches, host_positions, cache_capacity,
            cache_dtype);
        ops::kv_cache_store_pair_positions_(
            cache.key, cache.value, packed_key, packed_value, positions,
            cache_rows);
        const auto repeats = config_.heads / config_.kv_heads;
        auto context = ops::cached_gqa_attention_positions(
                           query, cache.key, cache.value, positions, cache_rows,
                           repeats,
                           1.0F / std::sqrt(static_cast<float>(
                                      config_.head_dimension())))
                           .transpose(1, 2)
                           .contiguous()
                           .reshape({batch, config_.dimension});
        return output_.forward_tensor(context).reshape(
            {batch, 1, config_.dimension});
    }

    void append_named(const std::string& prefix, NamedValues& values) {
        values.emplace_back(prefix + ".q_proj.weight", &query_.weight());
        if (query_.has_bias()) values.emplace_back(prefix + ".q_proj.bias", &query_.bias());
        values.emplace_back(prefix + ".k_proj.weight", &key_.weight());
        if (key_.has_bias()) values.emplace_back(prefix + ".k_proj.bias", &key_.bias());
        values.emplace_back(prefix + ".v_proj.weight", &value_.weight());
        if (value_.has_bias()) values.emplace_back(prefix + ".v_proj.bias", &value_.bias());
        values.emplace_back(prefix + ".o_proj.weight", &output_.weight());
    }

    void append_bf16_training_mirrors(Bf16TrainingMirrors& mirrors) {
        for (auto* linear : {&query_, &key_, &value_, &output_}) {
            mirrors.push_back(linear->prepare_bf16_training_mirror());
        }
    }
    void move_bf16_training_mirrors(Device device) {
        for (auto* linear : {&query_, &key_, &value_, &output_}) {
            linear->move_bf16_training_mirror(device);
        }
    }
    void append_fp8_inference_linears(std::vector<Linear*>& linears) {
        for (auto* linear : {&query_, &key_, &value_, &output_}) {
            if (linear->is_fp8()) linears.push_back(linear);
        }
    }
    void move_fp8_inference_scales(Device device) {
        for (auto* linear : {&query_, &key_, &value_, &output_}) {
            linear->move_fp8_inference_scale(device);
        }
    }

private:
    ModelConfig config_;
    Linear query_;
    Linear key_;
    Linear value_;
    Linear output_;
};

class FeedForward {
public:
    FeedForward(const ModelConfig& config, std::mt19937_64& generator,
                ParameterInitialization initialization)
        : config_(config),
          gate_(config.dimension, config.ffn_dimension, generator, config,
                initialization, false, true),
          up_(config.dimension, config.ffn_dimension, generator, config,
              initialization, false, true),
          down_(config.ffn_dimension, config.dimension, generator, config,
                initialization, false, true) {}

    Value forward(const Value& input) {
        const auto batch = input.data().shape()[0];
        const auto sequence = input.data().shape()[1];
        const auto flat = autograd::reshape(input, {batch * sequence, config_.dimension});
        // Keep projection evaluation order explicit so traces and failure attribution are stable.
        const auto gate = gate_.forward(flat);
        const auto up = up_.forward(flat);
        const auto activated = autograd::swiglu(gate, up);
        return autograd::reshape(down_.forward(activated),
                                 {batch, sequence, config_.dimension});
    }

    Tensor forward_tensor(const Tensor& input,
                          const std::string& trace_prefix = {}) {
        if (input.ndim() != 3) throw std::invalid_argument("FFN input must be BxTxD");
        const auto batch = input.shape()[0];
        const auto sequence = input.shape()[1];
        const auto flat = input.reshape({batch * sequence, config_.dimension});
        Tensor output;
        if (gate_.weight_data().dtype() == DType::BFloat16) {
            if (up_.weight_data().dtype() != DType::BFloat16 ||
                down_.weight_data().dtype() != DType::BFloat16) {
                throw std::logic_error("FFN inference weights have mixed preparation state");
            }
            if (trace_prefix.empty()) {
                output = ops::bf16_ffn(
                    flat, gate_.weight_data(), up_.weight_data(),
                    down_.weight_data());
            } else {
                const auto diagnostics = ops::bf16_ffn_diagnostics(
                    flat, gate_.weight_data(), up_.weight_data(),
                    down_.weight_data());
                trace_detail(trace_prefix, "input_bf16",
                             diagnostics.input_bf16);
                trace_detail(trace_prefix, "gate", diagnostics.gate);
                trace_detail(trace_prefix, "up", diagnostics.up);
                trace_detail(trace_prefix, "activated",
                             diagnostics.activated);
                trace_detail(trace_prefix, "down", diagnostics.output);
                output = diagnostics.output;
            }
        } else {
            Tensor gate;
            Tensor up;
            if (gate_.shares_dynamic_activation() &&
                up_.shares_dynamic_activation()) {
                const auto scaled = gate_.quantize_activation(flat);
                gate = gate_.forward_scaled_input(scaled);
                up = up_.forward_scaled_input(scaled);
            } else {
                gate = gate_.forward_tensor(flat);
                up = up_.forward_tensor(flat);
            }
            const auto activated = ops::swiglu(gate, up);
            auto* trace = profiling::TraceSession::current();
            if (trace != nullptr &&
                trace->options().record_all_layer_details) {
                trace_detail(trace_prefix, "gate", gate);
                trace_detail(trace_prefix, "up", up);
                trace_detail(trace_prefix, "activated", activated);
            }
            output = down_.forward_tensor(activated);
        }
        auto reshaped = output.reshape({batch, sequence, config_.dimension});
        trace_detail(trace_prefix, "output", reshaped);
        return reshaped;
    }

    void append_named(const std::string& prefix, NamedValues& values) {
        values.emplace_back(prefix + ".gate_proj.weight", &gate_.weight());
        values.emplace_back(prefix + ".up_proj.weight", &up_.weight());
        values.emplace_back(prefix + ".down_proj.weight", &down_.weight());
    }

    void append_bf16_training_mirrors(Bf16TrainingMirrors& mirrors) {
        for (auto* linear : {&gate_, &up_, &down_}) {
            mirrors.push_back(linear->prepare_bf16_training_mirror());
        }
    }
    void move_bf16_training_mirrors(Device device) {
        for (auto* linear : {&gate_, &up_, &down_}) {
            linear->move_bf16_training_mirror(device);
        }
    }
    void append_fp8_inference_linears(std::vector<Linear*>& linears) {
        for (auto* linear : {&gate_, &up_, &down_}) {
            if (linear->is_fp8()) linears.push_back(linear);
        }
    }
    void move_fp8_inference_scales(Device device) {
        for (auto* linear : {&gate_, &up_, &down_}) {
            linear->move_fp8_inference_scale(device);
        }
    }

private:
    ModelConfig config_;
    Linear gate_;
    Linear up_;
    Linear down_;
};

class Block {
public:
    Block(const ModelConfig& config, std::mt19937_64& generator,
          ParameterInitialization initialization)
        : attention_norm_(config.dimension, config.rms_norm_epsilon, initialization),
          attention_(config, generator, initialization),
          ffn_norm_(config.dimension, config.rms_norm_epsilon, initialization),
          feed_forward_(config, generator, initialization) {}

    Value forward(const Value& input) {
        auto hidden = autograd::add(input, attention_.forward(attention_norm_.forward(input)));
        return autograd::add(hidden, feed_forward_.forward(ffn_norm_.forward(hidden)));
    }

    Tensor forward_tensor(const Tensor& input,
                          const std::string& trace_prefix = {}) {
        auto attention_input = attention_norm_.forward_tensor(input);
        trace_detail(trace_prefix, "attention_norm", attention_input);
        auto attention = attention_.forward_tensor(
            attention_input, nullptr, 0, DType::Float32,
            trace_prefix.empty() ? std::string{} : trace_prefix + ".attention");
        auto hidden = ops::add(input, attention);
        trace_detail(trace_prefix, "attention_residual", hidden);
        auto ffn_input = ffn_norm_.forward_tensor(hidden);
        trace_detail(trace_prefix, "ffn_norm", ffn_input);
        auto ffn = feed_forward_.forward_tensor(
            ffn_input,
            trace_prefix.empty() ? std::string{} : trace_prefix + ".ffn");
        return ops::add(hidden, ffn);
    }

    Tensor forward_prefill_cached(const Tensor& input,
                                  inference::KVCache::LayerState& cache,
                                  std::int64_t cache_capacity,
                                  DType cache_dtype) {
        auto hidden = ops::add(
            input, attention_.forward_tensor(attention_norm_.forward_tensor(input),
                                             &cache, cache_capacity, cache_dtype));
        return ops::add(hidden, feed_forward_.forward_tensor(
                                    ffn_norm_.forward_tensor(hidden)));
    }

    Tensor forward_cached(const Tensor& input, inference::KVCache::LayerState& cache,
                          std::int64_t position, std::int64_t cache_capacity,
                          DType cache_dtype) {
        auto attention = attention_.forward_cached(attention_norm_.forward_tensor(input), cache,
                                                   position, cache_capacity, cache_dtype);
        auto residual_and_norm = ffn_norm_.add_forward_tensor(input, attention);
        return ops::add(residual_and_norm.first,
                        feed_forward_.forward_tensor(residual_and_norm.second));
    }

    Tensor forward_cached_positions(
        const Tensor& input, inference::KVCache::LayerState& cache,
        const Tensor& positions, const Tensor& cache_rows,
        const std::vector<std::int64_t>& host_positions,
        std::int64_t cache_batches, std::int64_t cache_capacity,
        DType cache_dtype) {
        auto attention = attention_.forward_cached_positions(
            attention_norm_.forward_tensor(input), cache, positions, cache_rows,
            host_positions, cache_batches, cache_capacity, cache_dtype);
        auto residual_and_norm = ffn_norm_.add_forward_tensor(input, attention);
        return ops::add(residual_and_norm.first,
                        feed_forward_.forward_tensor(residual_and_norm.second));
    }

    void append_named(const std::string& prefix, NamedValues& values) {
        values.emplace_back(prefix + ".attention_norm.weight", &attention_norm_.weight());
        attention_.append_named(prefix + ".attention", values);
        values.emplace_back(prefix + ".ffn_norm.weight", &ffn_norm_.weight());
        feed_forward_.append_named(prefix + ".feed_forward", values);
    }

    void append_bf16_training_mirrors(Bf16TrainingMirrors& mirrors) {
        attention_.append_bf16_training_mirrors(mirrors);
        feed_forward_.append_bf16_training_mirrors(mirrors);
    }
    void move_bf16_training_mirrors(Device device) {
        attention_.move_bf16_training_mirrors(device);
        feed_forward_.move_bf16_training_mirrors(device);
    }
    void append_fp8_inference_linears(std::vector<Linear*>& linears) {
        attention_.append_fp8_inference_linears(linears);
        feed_forward_.append_fp8_inference_linears(linears);
    }
    void move_fp8_inference_scales(Device device) {
        attention_.move_fp8_inference_scales(device);
        feed_forward_.move_fp8_inference_scales(device);
    }


private:
    Norm attention_norm_;
    Attention attention_;
    Norm ffn_norm_;
    FeedForward feed_forward_;
};

}  // namespace

struct TransformerModel::Impl {
    Impl(ModelConfig model_config, std::uint64_t seed,
         ParameterInitialization initialization)
        : config(std::move(model_config)),
          generator(seed),
          token_embedding(parameter({config.vocabulary_size, config.dimension}, generator,
                                    0.02F, initialization)),
          final_norm(config.dimension, config.rms_norm_epsilon, initialization) {
        config.validate();
        blocks.reserve(static_cast<std::size_t>(config.layers));
        for (std::int64_t layer = 0; layer < config.layers; ++layer) {
            auto block_config = config;
            if (std::binary_search(config.fp8_fp32_layers.begin(),
                                   config.fp8_fp32_layers.end(), layer)) {
                block_config.linear_precision = LinearPrecision::Float32;
                block_config.fp8_weight_scale_mode = Fp8WeightScaleMode::Fixed;
                block_config.fp8_activation_scale_mode =
                    Fp8ActivationScaleMode::Fixed;
            }
            blocks.push_back(std::make_unique<Block>(
                block_config, generator, initialization));
        }
        if (!config.tie_embeddings) {
            output_head = std::make_unique<Linear>(config.dimension, config.vocabulary_size,
                                                   generator, config, initialization);
        }
    }

    ModelConfig config;
    std::mt19937_64 generator;
    Value token_embedding;
    std::vector<std::unique_ptr<Block>> blocks;
    Norm final_norm;
    std::unique_ptr<Linear> output_head;
    bool bf16_ffn_prepared = false;
    bool bf16_attention_prepared = false;
    bool bf16_training_mirrors_prepared = false;
    bool fp8_inference_prepared = false;
    bool parameters_initialized = true;
};

TransformerModel::TransformerModel(ModelConfig config, std::uint64_t seed,
                                   ParameterInitialization initialization)
    : impl_(std::make_unique<Impl>(std::move(config), seed, initialization)) {
    impl_->parameters_initialized = initialization == ParameterInitialization::Random;
    if (parameter_count() != impl_->config.parameter_count()) {
        throw std::logic_error("constructed model parameter count does not match ModelConfig");
    }
}
TransformerModel::~TransformerModel() = default;
TransformerModel::TransformerModel(TransformerModel&&) noexcept = default;
TransformerModel& TransformerModel::operator=(TransformerModel&&) noexcept = default;
const ModelConfig& TransformerModel::config() const noexcept { return impl_->config; }

Device TransformerModel::device() {
    return impl_->token_embedding.data().device();
}

void TransformerModel::to(Device target) {
    for (auto* value : parameters()) {
        value->mutable_data() = impl_->parameters_initialized
                                    ? value->data().to(target)
                                    : Tensor(value->data().shape(), value->data().dtype(), target);
        value->zero_grad();
    }
    if (impl_->bf16_training_mirrors_prepared) {
        for (auto& block : impl_->blocks) block->move_bf16_training_mirrors(target);
        if (impl_->output_head) impl_->output_head->move_bf16_training_mirror(target);
    }
    if (impl_->fp8_inference_prepared) {
        for (auto& block : impl_->blocks) block->move_fp8_inference_scales(target);
        if (impl_->output_head) impl_->output_head->move_fp8_inference_scale(target);
    }
}

Value TransformerModel::forward(const Tensor& token_ids) {
    if (!impl_->parameters_initialized) {
        throw std::logic_error("model parameters must be loaded before forward");
    }
    if (impl_->bf16_ffn_prepared || impl_->bf16_attention_prepared ||
        impl_->fp8_inference_prepared) {
        throw std::logic_error(
            "autograd forward is unavailable after one-way inference preparation; "
            "use forward_inference or forward_cached");
    }
    if (token_ids.dtype() != DType::Int32 || token_ids.ndim() != 2) {
        throw std::invalid_argument("model token IDs must be an int32 BxT tensor");
    }
    if (token_ids.shape()[1] > impl_->config.max_sequence_length) {
        throw std::invalid_argument("token sequence exceeds configured maximum");
    }
    const auto model_tokens = token_ids.device() == device() ? token_ids : token_ids.to(device());
    auto* trace = profiling::TraceSession::current();
    if (trace != nullptr) trace->record(profiling::TraceKind::Input, "model.tokens", model_tokens);
    profiling::TraceTimer model_timer(profiling::TraceKind::Model, "model.forward", device());

    profiling::TraceTimer embedding_timer(profiling::TraceKind::Layer,
                                           "model.embedding", device());
    auto hidden = autograd::embedding(impl_->token_embedding, model_tokens);
    embedding_timer.finish(hidden.data());
    for (std::size_t layer = 0; layer < impl_->blocks.size(); ++layer) {
        profiling::TraceTimer block_timer(
            profiling::TraceKind::Layer,
            "model.blocks." + std::to_string(layer), device());
        hidden = impl_->blocks[layer]->forward(hidden);
        block_timer.finish(hidden.data());
    }
    profiling::TraceTimer norm_timer(profiling::TraceKind::Layer,
                                      "model.final_norm", device());
    hidden = impl_->final_norm.forward(hidden);
    norm_timer.finish(hidden.data());
    const auto batch = token_ids.shape()[0];
    const auto sequence = token_ids.shape()[1];
    const auto flat = autograd::reshape(hidden, {batch * sequence, impl_->config.dimension});
    Value logits;
    if (impl_->config.tie_embeddings) {
        logits = autograd::matmul(flat, impl_->token_embedding, false, true);
    } else {
        logits = impl_->output_head->forward(flat);
    }
    auto output = autograd::reshape(logits, {batch, sequence, impl_->config.vocabulary_size});
    if (trace != nullptr) trace->record(profiling::TraceKind::Output, "model.logits", output.data());
    model_timer.finish(output.data());
    return output;
}

Tensor TransformerModel::forward_inference(const Tensor& token_ids) {
    return forward_inference_impl(token_ids, false);
}

Tensor TransformerModel::forward_inference_last_logits(const Tensor& token_ids) {
    return forward_inference_impl(token_ids, true);
}

Tensor TransformerModel::forward_inference_impl(const Tensor& token_ids,
                                                 bool last_logits_only) {
    if (!impl_->parameters_initialized) {
        throw std::logic_error("model parameters must be loaded before inference");
    }
    if (token_ids.dtype() != DType::Int32 || token_ids.ndim() != 2) {
        throw std::invalid_argument("model token IDs must be an int32 BxT tensor");
    }
    if (token_ids.shape()[1] > impl_->config.max_sequence_length) {
        throw std::invalid_argument("token sequence exceeds configured maximum");
    }
    const auto model_tokens = token_ids.device() == device() ? token_ids : token_ids.to(device());
    auto* trace = profiling::TraceSession::current();
    if (trace != nullptr) {
        trace->record(profiling::TraceKind::Input, "inference.tokens", model_tokens);
    }
    profiling::TraceTimer model_timer(
        profiling::TraceKind::Model, "inference.forward", device());
    profiling::TraceTimer embedding_timer(
        profiling::TraceKind::Layer, "inference.embedding", device());
    auto hidden = ops::embedding(impl_->token_embedding.data(), model_tokens);
    embedding_timer.finish(hidden);
    for (std::size_t layer = 0; layer < impl_->blocks.size(); ++layer) {
        profiling::TraceTimer block_timer(
            profiling::TraceKind::Layer,
            "inference.blocks." + std::to_string(layer), device());
        const auto detail_prefix =
            trace != nullptr &&
                    (layer == 0 || trace->options().record_all_layer_details)
                ? "inference.blocks." + std::to_string(layer)
                : std::string{};
        hidden = impl_->blocks[layer]->forward_tensor(hidden, detail_prefix);
        block_timer.finish(hidden);
    }
    profiling::TraceTimer norm_timer(
        profiling::TraceKind::Layer, "inference.final_norm", device());
    hidden = impl_->final_norm.forward_tensor(hidden);
    norm_timer.finish(hidden);
    const auto batch = token_ids.shape()[0];
    const auto sequence = token_ids.shape()[1];
    const auto selected = last_logits_only
                              ? hidden.slice(1, sequence - 1, sequence).contiguous()
                              : hidden;
    const auto positions = last_logits_only ? 1 : sequence;
    const auto flat = selected.reshape({batch * positions, impl_->config.dimension});
    Tensor logits;
    if (impl_->config.tie_embeddings) {
        logits = ops::matmul_with_implementation(
            flat, impl_->token_embedding.data(), ops::MatmulImplementation::Auto,
            false, true);
    } else {
        logits = impl_->output_head->forward_tensor(flat);
    }
    auto output = logits.reshape(
        {batch, positions, impl_->config.vocabulary_size});
    if (trace != nullptr) {
        trace->record(profiling::TraceKind::Output, "inference.logits", output);
    }
    model_timer.finish(output);
    return output;
}

Tensor TransformerModel::forward_prefill_cached(
    const Tensor& token_ids, inference::KVCache& cache) {
    if (!impl_->parameters_initialized) {
        throw std::logic_error("model parameters must be loaded before cached prefill");
    }
    if (token_ids.dtype() != DType::Int32 || token_ids.ndim() != 2 ||
        token_ids.shape()[0] <= 0 || token_ids.shape()[1] <= 0) {
        throw std::invalid_argument("cached prefill expects a non-empty BxT int32 sequence");
    }
    const auto batch = token_ids.shape()[0];
    const auto sequence = token_ids.shape()[1];
    if (cache.layer_count() != impl_->blocks.size() ||
        cache.batch_size() != batch || cache.position() != 0 ||
        sequence > cache.max_sequence_length() ||
        cache.max_sequence_length() > impl_->config.max_sequence_length) {
        throw std::invalid_argument("cached prefill requires an empty matching KV cache");
    }
    try {
        const auto model_tokens = token_ids.device() == device()
                                      ? token_ids : token_ids.to(device());
        auto hidden = ops::embedding(impl_->token_embedding.data(), model_tokens);
        for (std::size_t layer = 0; layer < impl_->blocks.size(); ++layer) {
            hidden = impl_->blocks[layer]->forward_prefill_cached(
                hidden, cache.mutable_layer(layer), cache.max_sequence_length(),
                cache.layer_dtype(layer));
        }
        hidden = impl_->final_norm.forward_tensor(hidden);
        const auto last = hidden.slice(1, sequence - 1, sequence)
                              .contiguous()
                              .reshape({batch, impl_->config.dimension});
        Tensor logits;
        if (impl_->config.tie_embeddings) {
            logits = ops::matmul_with_implementation(
                last, impl_->token_embedding.data(), ops::MatmulImplementation::Auto,
                false, true);
        } else {
            logits = impl_->output_head->forward_tensor(last);
        }
        cache.advance(sequence);
        return logits.reshape({batch, 1, impl_->config.vocabulary_size});
    } catch (...) {
        cache.reset();
        throw;
    }
}

Tensor TransformerModel::forward_prefill_cached_row(
    const Tensor& token_ids, inference::KVCache& cache, std::int64_t row) {
    if (!impl_->parameters_initialized) {
        throw std::logic_error("model parameters must be loaded before cached prefill");
    }
    if (token_ids.dtype() != DType::Int32 || token_ids.ndim() != 2 ||
        token_ids.shape()[0] != 1 || token_ids.shape()[1] <= 0) {
        throw std::invalid_argument("row prefill expects a non-empty 1xT int32 sequence");
    }
    if (row < 0 || row >= cache.batch_size()) {
        throw std::out_of_range("row prefill target is outside the KV cache batch");
    }
    const auto sequence = token_ids.shape()[1];
    if (cache.layer_count() != impl_->blocks.size() ||
        cache.row_position(row) != 0 || sequence > cache.max_sequence_length() ||
        cache.max_sequence_length() > impl_->config.max_sequence_length) {
        throw std::invalid_argument("row prefill requires an empty matching cache row");
    }
    if (cache.batch_size() == 1) return forward_prefill_cached(token_ids, cache);

    const auto maximum_prefix = *std::max_element(
        cache.row_positions().begin(), cache.row_positions().end());
    const auto target_prefix = std::max(maximum_prefix, sequence);
    std::vector<DType> layer_dtypes;
    layer_dtypes.reserve(cache.layer_count());
    for (std::size_t layer = 0; layer < cache.layer_count(); ++layer) {
        const auto& state = cache.layer(layer);
        if (state.key.defined() != state.value.defined() ||
            (maximum_prefix > 0 && !state.key.defined())) {
            throw std::invalid_argument("row prefill shared KV storage is incomplete");
        }
        layer_dtypes.push_back(cache.layer_dtype(layer));
    }

    inference::KVCache local_cache(
        layer_dtypes, cache.max_sequence_length(), 1);
    const auto logits = forward_prefill_cached(token_ids, local_cache);
    const auto kv_heads = impl_->config.kv_heads;
    const auto width = impl_->config.head_dimension();
    const auto model_device = device();
    for (std::size_t layer = 0; layer < cache.layer_count(); ++layer) {
        auto& shared = cache.mutable_layer(layer);
        if (!shared.key.defined()) {
            ensure_batched_cache_tensor(
                shared.key, cache.batch_size(), kv_heads, 0,
                cache.max_sequence_length(), width, cache.layer_dtype(layer),
                model_device);
        }
        if (!shared.value.defined()) {
            ensure_batched_cache_tensor(
                shared.value, cache.batch_size(), kv_heads, 0,
                cache.max_sequence_length(), width, cache.layer_dtype(layer),
                model_device);
        }
        ensure_batched_cache_tensor(
            shared.key, cache.batch_size(), kv_heads, target_prefix,
            cache.max_sequence_length(), width, cache.layer_dtype(layer),
            model_device);
        ensure_batched_cache_tensor(
            shared.value, cache.batch_size(), kv_heads, target_prefix,
            cache.max_sequence_length(), width, cache.layer_dtype(layer),
            model_device);
        copy_cache_prefix_to_row(shared.key, local_cache.layer(layer).key, row,
                                 sequence);
        copy_cache_prefix_to_row(shared.value, local_cache.layer(layer).value, row,
                                 sequence);
    }
    cache.advance_row(row, sequence);
    return logits;
}

Tensor TransformerModel::forward_prefill_cached_rows(
    const Tensor& token_ids, inference::KVCache& cache,
    const std::vector<std::int64_t>& active_rows) {
    if (!impl_->parameters_initialized) {
        throw std::logic_error("model parameters must be loaded before cached prefill");
    }
    if (token_ids.dtype() != DType::Int32 || token_ids.ndim() != 2 ||
        token_ids.shape()[0] != static_cast<std::int64_t>(active_rows.size()) ||
        token_ids.shape()[0] <= 0 || token_ids.shape()[1] <= 0) {
        throw std::invalid_argument(
            "row prefill batch expects non-empty AxT tokens and A rows");
    }
    const auto active = token_ids.shape()[0];
    const auto sequence = token_ids.shape()[1];
    if (cache.layer_count() != impl_->blocks.size() ||
        sequence > cache.max_sequence_length() ||
        cache.max_sequence_length() > impl_->config.max_sequence_length) {
        throw std::invalid_argument(
            "row prefill batch does not match model or cache capacity");
    }
    auto all_rows = active == cache.batch_size();
    for (std::size_t index = 0; index < active_rows.size(); ++index) {
        const auto row = active_rows[index];
        if (row < 0 || row >= cache.batch_size() ||
            (index > 0 && active_rows[index - 1] >= row) ||
            cache.row_position(row) != 0) {
            throw std::invalid_argument(
                "row prefill targets must be unique increasing empty rows");
        }
        all_rows = all_rows && row == static_cast<std::int64_t>(index);
    }
    auto storage_is_empty = true;
    for (std::size_t layer = 0; layer < cache.layer_count(); ++layer) {
        const auto& state = cache.layer(layer);
        storage_is_empty = storage_is_empty &&
                           !state.key.defined() && !state.value.defined();
    }
    if (all_rows && cache.positions_uniform() && storage_is_empty) {
        return forward_prefill_cached(token_ids, cache);
    }

    const auto maximum_prefix = *std::max_element(
        cache.row_positions().begin(), cache.row_positions().end());
    const auto target_prefix = std::max(maximum_prefix, sequence);
    std::vector<DType> layer_dtypes;
    layer_dtypes.reserve(cache.layer_count());
    for (std::size_t layer = 0; layer < cache.layer_count(); ++layer) {
        const auto& state = cache.layer(layer);
        if (state.key.defined() != state.value.defined() ||
            (maximum_prefix > 0 && !state.key.defined())) {
            throw std::invalid_argument(
                "row prefill batch shared KV storage is incomplete");
        }
        layer_dtypes.push_back(cache.layer_dtype(layer));
    }
    inference::KVCache local_cache(
        layer_dtypes, cache.max_sequence_length(), active);
    const auto logits = forward_prefill_cached(token_ids, local_cache);
    const auto kv_heads = impl_->config.kv_heads;
    const auto width = impl_->config.head_dimension();
    const auto model_device = device();
    for (std::size_t layer = 0; layer < cache.layer_count(); ++layer) {
        auto& shared = cache.mutable_layer(layer);
        if (!shared.key.defined()) {
            ensure_batched_cache_tensor(
                shared.key, cache.batch_size(), kv_heads, 0,
                cache.max_sequence_length(), width, cache.layer_dtype(layer),
                model_device);
        }
        if (!shared.value.defined()) {
            ensure_batched_cache_tensor(
                shared.value, cache.batch_size(), kv_heads, 0,
                cache.max_sequence_length(), width, cache.layer_dtype(layer),
                model_device);
        }
        ensure_batched_cache_tensor(
            shared.key, cache.batch_size(), kv_heads, target_prefix,
            cache.max_sequence_length(), width, cache.layer_dtype(layer),
            model_device);
        ensure_batched_cache_tensor(
            shared.value, cache.batch_size(), kv_heads, target_prefix,
            cache.max_sequence_length(), width, cache.layer_dtype(layer),
            model_device);
        for (std::int64_t index = 0; index < active; ++index) {
            const auto source_key = cache_row_view(
                local_cache.layer(layer).key, index, sequence);
            const auto source_value = cache_row_view(
                local_cache.layer(layer).value, index, sequence);
            copy_cache_prefix_to_row(
                shared.key, source_key,
                active_rows[static_cast<std::size_t>(index)], sequence);
            copy_cache_prefix_to_row(
                shared.value, source_value,
                active_rows[static_cast<std::size_t>(index)], sequence);
        }
    }
    for (const auto row : active_rows) cache.advance_row(row, sequence);
    return logits;
}

Value TransformerModel::loss(const Tensor& token_ids, const Tensor& targets) {
    if (targets.shape() != token_ids.shape()) {
        throw std::invalid_argument("language-model targets must match token shape");
    }
    const auto model_targets = targets.device() == device() ? targets : targets.to(device());
    return autograd::cross_entropy(forward(token_ids), model_targets);
}

Tensor TransformerModel::forward_cached(const Tensor& token_id, inference::KVCache& cache) {
    if (!impl_->parameters_initialized) {
        throw std::logic_error("model parameters must be loaded before cached inference");
    }
    if (token_id.dtype() != DType::Int32 || token_id.ndim() != 2 ||
        token_id.shape()[0] <= 0 || token_id.shape()[1] != 1) {
        throw std::invalid_argument("cached forward expects int32 tokens with shape Bx1");
    }
    const auto batch = token_id.shape()[0];
    if (cache.layer_count() != impl_->blocks.size() ||
        cache.batch_size() != batch ||
        cache.max_sequence_length() > impl_->config.max_sequence_length) {
        throw std::invalid_argument("KV cache does not match model configuration");
    }
    if (cache.position() >= cache.max_sequence_length()) {
        throw std::out_of_range("KV cache has reached maximum sequence length");
    }
    const auto model_device = device();
    const auto device_token = token_id.device() == model_device ? token_id : token_id.to(model_device);
    auto hidden = ops::embedding(impl_->token_embedding.data(), device_token);
    for (std::size_t layer = 0; layer < impl_->blocks.size(); ++layer) {
        hidden = impl_->blocks[layer]->forward_cached(hidden, cache.mutable_layer(layer),
                                                      cache.position(),
                                                      cache.max_sequence_length(),
                                                      cache.layer_dtype(layer));
    }
    hidden = impl_->final_norm.forward_tensor(hidden);
    const auto flat = hidden.reshape({batch, impl_->config.dimension});
    Tensor logits;
    if (impl_->config.tie_embeddings) {
        logits = ops::matmul_with_implementation(
            flat, impl_->token_embedding.data(), ops::MatmulImplementation::Auto,
            false, true);
    } else {
        logits = impl_->output_head->forward_tensor(flat);
    }
    cache.advance();
    return logits.reshape({batch, 1, impl_->config.vocabulary_size});
}

Tensor TransformerModel::forward_cached_rows(const Tensor& token_ids,
                                             inference::KVCache& cache) {
    if (!impl_->parameters_initialized) {
        throw std::logic_error("model parameters must be loaded before cached inference");
    }
    if (token_ids.dtype() != DType::Int32 || token_ids.ndim() != 2 ||
        token_ids.shape()[0] <= 0 || token_ids.shape()[1] != 1) {
        throw std::invalid_argument(
            "divergent cached forward expects int32 tokens with shape Bx1");
    }
    const auto batch = token_ids.shape()[0];
    if (cache.layer_count() != impl_->blocks.size() ||
        cache.batch_size() != batch ||
        cache.max_sequence_length() > impl_->config.max_sequence_length) {
        throw std::invalid_argument("divergent KV cache does not match model configuration");
    }
    if (cache.positions_uniform()) return forward_cached(token_ids, cache);
    if (std::any_of(cache.row_positions().begin(), cache.row_positions().end(),
                    [&cache](std::int64_t position) {
                        return position < 0 || position >= cache.max_sequence_length();
                    })) {
        throw std::out_of_range("a KV cache row has reached maximum sequence length");
    }

    const auto maximum_prefix = *std::max_element(
        cache.row_positions().begin(), cache.row_positions().end());
    const auto kv_heads = impl_->config.kv_heads;
    const auto width = impl_->config.head_dimension();
    const auto model_device = device();
    std::vector<DType> layer_dtypes;
    layer_dtypes.reserve(cache.layer_count());
    for (std::size_t layer = 0; layer < cache.layer_count(); ++layer) {
        auto& state = cache.mutable_layer(layer);
        if (state.key.defined() != state.value.defined()) {
            throw std::invalid_argument("divergent KV cache has an incomplete layer");
        }
        const auto dtype = cache.layer_dtype(layer);
        layer_dtypes.push_back(dtype);
        ensure_batched_cache_tensor(
            state.key, batch, kv_heads, maximum_prefix,
            cache.max_sequence_length(), width, dtype, model_device);
        ensure_batched_cache_tensor(
            state.value, batch, kv_heads, maximum_prefix,
            cache.max_sequence_length(), width, dtype, model_device);
    }

    const auto resize_shared_views = [&]() {
        const auto prefix = *std::max_element(
            cache.row_positions().begin(), cache.row_positions().end());
        for (std::size_t layer = 0; layer < cache.layer_count(); ++layer) {
            auto& state = cache.mutable_layer(layer);
            ensure_batched_cache_tensor(
                state.key, batch, kv_heads, prefix, cache.max_sequence_length(),
                width, cache.layer_dtype(layer), model_device);
            ensure_batched_cache_tensor(
                state.value, batch, kv_heads, prefix, cache.max_sequence_length(),
                width, cache.layer_dtype(layer), model_device);
        }
    };

    Tensor output({batch, 1, impl_->config.vocabulary_size},
                  DType::Float32, model_device);
    try {
        for (std::int64_t row = 0; row < batch; ++row) {
            const auto position = cache.row_position(row);
            inference::KVCache row_cache(
                layer_dtypes, cache.max_sequence_length(), 1);
            if (position > 0) row_cache.advance(position);
            for (std::size_t layer = 0; layer < cache.layer_count(); ++layer) {
                const auto& shared = cache.layer(layer);
                auto& local = row_cache.mutable_layer(layer);
                local.key = cache_row_view(shared.key, row, position);
                local.value = cache_row_view(shared.value, row, position);
            }
            const auto row_token = token_ids.slice(0, row, row + 1).contiguous();
            const auto row_logits = forward_cached(row_token, row_cache);
            runtime::copy_bytes(
                static_cast<float*>(output.data()) +
                    row * impl_->config.vocabulary_size,
                output.device(), row_logits.data(), row_logits.device(),
                static_cast<std::size_t>(impl_->config.vocabulary_size) *
                    sizeof(float));
            cache.advance_row(row);
        }
    } catch (...) {
        resize_shared_views();
        throw;
    }
    resize_shared_views();
    return output;
}

Tensor TransformerModel::forward_cached_active_rows(
    const Tensor& token_ids, inference::KVCache& cache,
    const std::vector<std::int64_t>& active_rows) {
    if (!impl_->parameters_initialized) {
        throw std::logic_error("model parameters must be loaded before cached inference");
    }
    if (token_ids.dtype() != DType::Int32 || token_ids.ndim() != 2 ||
        token_ids.shape()[0] != static_cast<std::int64_t>(active_rows.size()) ||
        token_ids.shape()[1] != 1 || active_rows.empty()) {
        throw std::invalid_argument(
            "active cached forward expects Ax1 int32 tokens and A rows");
    }
    if (cache.layer_count() != impl_->blocks.size() ||
        cache.max_sequence_length() > impl_->config.max_sequence_length) {
        throw std::invalid_argument(
            "active-row KV cache does not match model configuration");
    }
    for (std::size_t index = 0; index < active_rows.size(); ++index) {
        const auto row = active_rows[index];
        if (row < 0 || row >= cache.batch_size() ||
            (index > 0 && active_rows[index - 1] >= row)) {
            throw std::invalid_argument(
                "active cache rows must be unique, increasing and in range");
        }
        if (cache.row_position(row) >= cache.max_sequence_length()) {
            throw std::out_of_range(
                "an active KV cache row reached maximum sequence length");
        }
    }
    auto all_rows =
        static_cast<std::int64_t>(active_rows.size()) == cache.batch_size();
    for (std::size_t index = 0; index < active_rows.size(); ++index) {
        all_rows = all_rows &&
                   active_rows[index] == static_cast<std::int64_t>(index);
    }
    if (all_rows && cache.positions_uniform()) {
        return forward_cached(token_ids, cache);
    }

    const auto model_device = device();
    for (std::size_t layer = 0; layer < cache.layer_count(); ++layer) {
        const auto& state = cache.layer(layer);
        if (state.key.defined() != state.value.defined()) {
            throw std::invalid_argument("active-row KV cache has an incomplete layer");
        }
        if (!state.key.defined() && std::any_of(
                active_rows.begin(), active_rows.end(), [&cache](std::int64_t row) {
                    return cache.row_position(row) != 0;
                })) {
            throw std::invalid_argument(
                "nonzero active rows require initialized KV storage");
        }
    }
    std::vector<std::int32_t> position_values;
    std::vector<std::int32_t> row_values;
    std::vector<std::int64_t> host_positions;
    position_values.reserve(active_rows.size());
    row_values.reserve(active_rows.size());
    host_positions.reserve(active_rows.size());
    for (const auto row : active_rows) {
        const auto position = cache.row_position(row);
        if (position > std::numeric_limits<std::int32_t>::max() ||
            row > std::numeric_limits<std::int32_t>::max()) {
            throw std::out_of_range(
                "active row or position exceeds the Int32 Kernel contract");
        }
        position_values.push_back(static_cast<std::int32_t>(position));
        row_values.push_back(static_cast<std::int32_t>(row));
        host_positions.push_back(position);
    }
    Tensor model_tokens;
    Tensor positions;
    Tensor rows;
    const auto active = static_cast<std::int64_t>(active_rows.size());
    if (model_device.is_hip() && token_ids.device().is_cpu()) {
        const auto token_values = token_ids.to_int32_vector();
        std::vector<std::int32_t> packed_values;
        packed_values.reserve(token_values.size() + position_values.size() +
                              row_values.size());
        packed_values.insert(packed_values.end(), token_values.begin(),
                             token_values.end());
        packed_values.insert(packed_values.end(), position_values.begin(),
                             position_values.end());
        packed_values.insert(packed_values.end(), row_values.begin(),
                             row_values.end());
        const auto packed = Tensor::from_int32_vector(
                                packed_values, {3, active})
                                .to(model_device);
        model_tokens = packed.slice(0, 0, 1).reshape({active, 1});
        positions = packed.slice(0, 1, 2).reshape({active});
        rows = packed.slice(0, 2, 3).reshape({active});
    } else {
        model_tokens = token_ids.device() == model_device
                           ? token_ids
                           : token_ids.to(model_device);
        positions = Tensor::from_int32_vector(position_values, {active});
        rows = Tensor::from_int32_vector(row_values, {active});
        if (model_device.is_hip()) {
            positions = positions.to(model_device);
            rows = rows.to(model_device);
        }
    }
    auto hidden = ops::embedding(impl_->token_embedding.data(), model_tokens);
    for (std::size_t layer = 0; layer < impl_->blocks.size(); ++layer) {
        hidden = impl_->blocks[layer]->forward_cached_positions(
            hidden, cache.mutable_layer(layer), positions, rows,
            host_positions, cache.batch_size(), cache.max_sequence_length(),
            cache.layer_dtype(layer));
    }
    hidden = impl_->final_norm.forward_tensor(hidden);
    const auto flat = hidden.reshape(
        {static_cast<std::int64_t>(active_rows.size()), impl_->config.dimension});
    Tensor logits;
    if (impl_->config.tie_embeddings) {
        logits = ops::matmul_with_implementation(
            flat, impl_->token_embedding.data(),
            ops::MatmulImplementation::Auto, false, true);
    } else {
        logits = impl_->output_head->forward_tensor(flat);
    }
    for (const auto row : active_rows) cache.advance_row(row);
    return logits.reshape(
        {static_cast<std::int64_t>(active_rows.size()), 1,
         impl_->config.vocabulary_size});
}

NamedValues TransformerModel::named_parameters() {
    NamedValues values;
    values.emplace_back("token_embedding.weight", &impl_->token_embedding);
    for (std::size_t layer = 0; layer < impl_->blocks.size(); ++layer) {
        impl_->blocks[layer]->append_named("blocks." + std::to_string(layer), values);
    }
    values.emplace_back("final_norm.weight", &impl_->final_norm.weight());
    if (impl_->output_head) values.emplace_back("output_head.weight", &impl_->output_head->weight());
    return values;
}

std::vector<Value*> TransformerModel::parameters() {
    const auto named = named_parameters();
    std::vector<Value*> values;
    values.reserve(named.size());
    for (const auto& [name, value] : named) {
        (void)name;
        values.push_back(value);
    }
    return values;
}

std::uint64_t TransformerModel::parameter_count() {
    std::uint64_t count = 0;
    for (const auto* value : parameters()) count += static_cast<std::uint64_t>(value->data().numel());
    return count;
}

Bf16FfnPreparationReport TransformerModel::prepare_bf16_ffn_inference() {
    if (impl_->bf16_ffn_prepared) {
        throw std::logic_error("BF16 FFN inference preparation is one-way and already complete");
    }
    if (impl_->config.linear_precision != LinearPrecision::Float32) {
        throw std::logic_error("BF16 inference preparation requires FP32 Linear policy");
    }
    // Transactional helper keeps every FP32 source alive until all casts finish.
    const auto report = prepare_bf16_weights(
        named_parameters(), impl_->blocks.size() * 3U, device(),
        [](const std::string& name) {
            return name.find(".feed_forward.") != std::string::npos;
        });
    impl_->bf16_ffn_prepared = true;
    return report;
}

bool TransformerModel::bf16_ffn_inference_prepared() const noexcept {
    return impl_->bf16_ffn_prepared;
}

Bf16WeightPreparationReport TransformerModel::prepare_bf16_attention_inference() {
    if (impl_->bf16_attention_prepared) {
        throw std::logic_error(
            "BF16 Attention inference preparation is one-way and already complete");
    }
    if (impl_->config.linear_precision != LinearPrecision::Float32) {
        throw std::logic_error("BF16 inference preparation requires FP32 Linear policy");
    }
    const auto report = prepare_bf16_weights(
        named_parameters(), impl_->blocks.size() * 4U, device(),
        [](const std::string& name) {
            return name.find(".attention.") != std::string::npos &&
                   name.ends_with(".weight");
        });
    impl_->bf16_attention_prepared = true;
    return report;
}

bool TransformerModel::bf16_attention_inference_prepared() const noexcept {
    return impl_->bf16_attention_prepared;
}

Fp8WeightPreparationReport TransformerModel::prepare_fp8_inference_weights() {
    if (!impl_->parameters_initialized ||
        impl_->config.linear_precision != LinearPrecision::Float8E4M3FNUZ ||
        impl_->fp8_inference_prepared || impl_->bf16_ffn_prepared ||
        impl_->bf16_attention_prepared || impl_->bf16_training_mirrors_prepared) {
        throw std::logic_error("FP8 inference preparation is invalid for model state");
    }
    std::vector<Linear*> linears;
    const auto expected =
        (impl_->blocks.size() - impl_->config.fp8_fp32_layers.size()) * 7U +
        (impl_->output_head && impl_->output_head->is_fp8() ? 1U : 0U);
    linears.reserve(expected);
    for (auto& block : impl_->blocks) {
        block->append_fp8_inference_linears(linears);
    }
    if (impl_->output_head && impl_->output_head->is_fp8()) {
        linears.push_back(impl_->output_head.get());
    }
    if (linears.size() != expected) {
        throw std::logic_error("FP8 inference Linear count changed");
    }
    Fp8WeightPreparationReport report;
    report.linears_covered = linears.size();
    if (impl_->config.fp8_diagnostic_mode ==
        Fp8DiagnosticMode::ActivationOnly) {
        std::vector<Tensor> activation_scales;
        activation_scales.reserve(linears.size());
        for (const auto* linear : linears) {
            activation_scales.push_back(
                linear->prepare_fp8_activation_scale_candidate());
            if (activation_scales.back().defined()) {
                report.scale_bytes_retained += sizeof(float);
            }
        }
        runtime::synchronize(device());
        for (std::size_t index = 0; index < linears.size(); ++index) {
            linears[index]->commit_fp8_activation_only_candidate(
                std::move(activation_scales[index]));
        }
        impl_->fp8_inference_prepared = true;
        return report;
    }
    struct Candidate {
        ops::ScaledTensor weight;
        Tensor activation_scale;
    };
    std::vector<Candidate> candidates;
    candidates.reserve(linears.size());
    for (const auto* linear : linears) {
        const auto elements = static_cast<std::uint64_t>(
            linear->weight_data().numel());
        candidates.push_back({
            linear->prepare_fp8_inference_candidate(),
            linear->prepare_fp8_activation_scale_candidate()});
        const auto& prepared_weight = candidates.back().weight;
        if (prepared_weight.host_scale_available) {
            const auto actual_scale = prepared_weight.scale_value;
            report.minimum_weight_scale = report.converted_tensors == 0
                                              ? actual_scale
                                              : std::min(report.minimum_weight_scale,
                                                         actual_scale);
            report.maximum_weight_scale = std::max(
                report.maximum_weight_scale, actual_scale);
        } else {
            report.host_scale_summary_available = false;
            if (linear->weight_data().device().is_hip()) {
                ++report.device_amax_tensors;
                report.device_weight_bytes_scanned += elements * sizeof(float);
            } else if (impl_->config.fp8_weight_scale_mode ==
                       Fp8WeightScaleMode::OutputChannelAmax) {
                report.weight_bytes_scanned += elements * sizeof(float);
            }
        }
        ++report.converted_tensors;
        report.fp32_bytes_released += elements * sizeof(float);
        report.fp8_bytes_retained += elements;
        report.scale_bytes_retained +=
            static_cast<std::uint64_t>(prepared_weight.scale.numel()) *
            sizeof(float);
        if (candidates.back().activation_scale.defined()) {
            report.scale_bytes_retained += sizeof(float);
        }
        if (impl_->config.fp8_weight_scale_mode == Fp8WeightScaleMode::TensorAmax) {
            report.weight_bytes_scanned += elements * sizeof(float);
        }
    }
    runtime::synchronize(device());
    if (!report.host_scale_summary_available) {
        report.minimum_weight_scale = 0.0F;
        report.maximum_weight_scale = 0.0F;
    }
    for (std::size_t index = 0; index < linears.size(); ++index) {
        linears[index]->commit_fp8_inference_candidate(
            std::move(candidates[index].weight),
            std::move(candidates[index].activation_scale));
    }
    impl_->fp8_inference_prepared = true;
    return report;
}

bool TransformerModel::fp8_inference_weights_prepared() const noexcept {
    return impl_->fp8_inference_prepared;
}

Bf16TrainingMirrors TransformerModel::prepare_bf16_training_mirrors() {
    if (impl_->config.linear_precision != LinearPrecision::BFloat16 ||
        impl_->bf16_training_mirrors_prepared || impl_->bf16_ffn_prepared ||
        impl_->bf16_attention_prepared) {
        throw std::logic_error("BF16 training mirror preparation is invalid for model state");
    }
    Bf16TrainingMirrors mirrors;
    const auto expected = impl_->blocks.size() * 7U + (impl_->output_head ? 1U : 0U);
    mirrors.reserve(expected);
    for (auto& block : impl_->blocks) block->append_bf16_training_mirrors(mirrors);
    if (impl_->output_head) {
        mirrors.push_back(impl_->output_head->prepare_bf16_training_mirror());
    }
    if (mirrors.size() != expected) {
        throw std::logic_error("BF16 training mirror count does not match model Linears");
    }
    impl_->bf16_training_mirrors_prepared = true;
    return mirrors;
}

bool TransformerModel::bf16_training_mirrors_prepared() const noexcept {
    return impl_->bf16_training_mirrors_prepared;
}

io::StateDict TransformerModel::state_dict(Device target) {
    io::StateDict state;
    for (const auto& [name, parameter] : named_parameters()) {
        auto copy = Tensor::from_vector(parameter->data().to_vector(), parameter->data().shape());
        if (target != Device::cpu()) copy = copy.to(target);
        state.emplace(name, std::move(copy));
    }
    return state;
}

LoadWeightsReport TransformerModel::load_state_dict(const io::StateDict& state,
                                                     const LoadWeightsOptions& options) {
    if (impl_->bf16_ffn_prepared || impl_->bf16_attention_prepared ||
        impl_->bf16_training_mirrors_prepared || impl_->fp8_inference_prepared) {
        throw std::logic_error(
            "load weights before preparing derived inference or training weights");
    }
    LoadWeightsReport report;
    const auto named = named_parameters();
    std::set<std::string> target_names;
    for (const auto& [name, parameter] : named) {
        (void)parameter;
        target_names.insert(name);
    }
    for (const auto& [target, source] : options.mapping) {
        (void)source;
        if (!target_names.contains(target)) {
            report.incompatible.push_back("mapping target is not a model parameter: " + target);
        }
    }

    struct Prepared {
        std::string name;
        Value* parameter = nullptr;
        Tensor tensor;
    };
    std::vector<Prepared> prepared;
    std::set<std::string> consumed;
    for (const auto& [target_name, parameter] : named) {
        const auto mapping = options.mapping.find(target_name);
        const auto source_name = mapping == options.mapping.end()
                                     ? target_name
                                     : mapping->second.name;
        const auto transform = mapping == options.mapping.end()
                                   ? WeightTransform::Identity
                                   : mapping->second.transform;
        const auto found = state.find(source_name);
        if (found == state.end()) {
            report.missing.push_back(target_name + " <- " + source_name);
            continue;
        }
        consumed.insert(source_name);
        auto source = found->second;
        if (!source.defined() || source.dtype() != DType::Float32) {
            report.incompatible.push_back(target_name + " requires a defined float32 source");
            continue;
        }
        if (transform == WeightTransform::Transpose2D) {
            if (source.ndim() != 2) {
                report.incompatible.push_back(target_name + " transpose requires rank two");
                continue;
            }
            source = source.transpose(0, 1).contiguous();
        }
        if (source.shape() != parameter->data().shape()) {
            std::ostringstream message;
            message << target_name << " shape mismatch: source=[";
            for (std::size_t index = 0; index < source.shape().size(); ++index) {
                if (index != 0) message << ',';
                message << source.shape()[index];
            }
            message << "] target=[";
            for (std::size_t index = 0; index < parameter->data().shape().size(); ++index) {
                if (index != 0) message << ',';
                message << parameter->data().shape()[index];
            }
            message << ']';
            report.incompatible.push_back(message.str());
            continue;
        }
        auto copy = transform == WeightTransform::Transpose2D
                        ? source.device() == parameter->data().device()
                              ? std::move(source)
                              : source.to(parameter->data().device())
                        : clone_tensor(source, parameter->data().device());
        prepared.push_back({target_name, parameter, std::move(copy)});
    }
    for (const auto& [name, tensor] : state) {
        (void)tensor;
        if (!consumed.contains(name)) report.unexpected.push_back(name);
    }

    if (options.strict && !report.complete()) {
        std::ostringstream message;
        message << "strict weight load failed";
        for (const auto& missing : report.missing) message << "\nmissing: " << missing;
        for (const auto& unexpected : report.unexpected) message << "\nunexpected: " << unexpected;
        for (const auto& incompatible : report.incompatible) {
            message << "\nincompatible: " << incompatible;
        }
        throw std::invalid_argument(message.str());
    }
    for (auto& item : prepared) {
        item.parameter->mutable_data() = std::move(item.tensor);
        item.parameter->zero_grad();
        report.loaded.push_back(std::move(item.name));
    }
    if (report.complete()) impl_->parameters_initialized = true;
    return report;
}

LoadWeightsReport TransformerModel::load_safetensors(
    const std::filesystem::path& path, const LoadWeightsOptions& options) {
    if (impl_->parameters_initialized || !device().is_hip()) {
        return load_state_dict(io::load_safetensors(path, device()), options);
    }

    const auto metadata = io::inspect_safetensors(path);
    std::map<std::string, const io::SafetensorsTensorInfo*> source_info;
    for (const auto& info : metadata) source_info.emplace(info.name, &info);
    const auto named = named_parameters();
    std::set<std::string> target_names;
    for (const auto& [name, parameter] : named) {
        (void)parameter;
        target_names.insert(name);
    }
    LoadWeightsReport report;
    for (const auto& [target, source] : options.mapping) {
        (void)source;
        if (!target_names.contains(target)) {
            report.incompatible.push_back("mapping target is not a model parameter: " + target);
        }
    }
    struct StreamingTarget {
        std::string name;
        Value* parameter = nullptr;
        WeightTransform transform = WeightTransform::Identity;
    };
    std::map<std::string, std::vector<StreamingTarget>> targets_by_source;
    std::set<std::string> consumed;
    for (const auto& [target_name, parameter] : named) {
        const auto mapping = options.mapping.find(target_name);
        const auto source_name = mapping == options.mapping.end()
                                     ? target_name : mapping->second.name;
        const auto transform = mapping == options.mapping.end()
                                   ? WeightTransform::Identity : mapping->second.transform;
        const auto found = source_info.find(source_name);
        if (found == source_info.end()) {
            report.missing.push_back(target_name + " <- " + source_name);
            continue;
        }
        auto expected_shape = found->second->shape;
        if (transform == WeightTransform::Transpose2D) {
            if (expected_shape.size() != 2) {
                report.incompatible.push_back(target_name + " transpose requires rank two");
                continue;
            }
            std::swap(expected_shape[0], expected_shape[1]);
        }
        if (expected_shape != parameter->data().shape()) {
            report.incompatible.push_back(target_name + " shape mismatch in safetensors header");
            continue;
        }
        consumed.insert(source_name);
        targets_by_source[source_name].push_back({target_name, parameter, transform});
    }
    for (const auto& info : metadata) {
        if (!consumed.contains(info.name)) report.unexpected.push_back(info.name);
    }
    if (options.strict && !report.complete()) {
        std::ostringstream message;
        message << "strict weight load failed before streaming payload";
        for (const auto& missing : report.missing) message << "\nmissing: " << missing;
        for (const auto& unexpected : report.unexpected) message << "\nunexpected: " << unexpected;
        for (const auto& incompatible : report.incompatible) {
            message << "\nincompatible: " << incompatible;
        }
        throw std::invalid_argument(message.str());
    }

    const auto dtype_slot = [](DType dtype) -> std::size_t {
        if (dtype == DType::Float32) return 0;
        if (dtype == DType::Float16) return 1;
        if (dtype == DType::BFloat16) return 2;
        throw std::invalid_argument("streamed weight dtype is unsupported");
    };
    std::array<std::int64_t, 3> maximum_elements{};
    std::array<bool, 3> needs_staging{};
    for (const auto& info : metadata) {
        const auto targets = targets_by_source.find(info.name);
        if (targets == targets_by_source.end()) continue;
        const auto needs = info.dtype != DType::Float32 || std::any_of(
            targets->second.begin(), targets->second.end(), [](const auto& target) {
                return target.transform == WeightTransform::Transpose2D;
            });
        if (!needs) continue;
        const auto slot = dtype_slot(info.dtype);
        needs_staging[slot] = true;
        maximum_elements[slot] = std::max(maximum_elements[slot], checked_numel(info.shape));
    }
    std::array<Tensor, 3> staging;
    for (std::size_t slot = 0; slot < staging.size(); ++slot) {
        if (!needs_staging[slot]) continue;
        const auto dtype = slot == 0 ? DType::Float32
                         : slot == 1 ? DType::Float16 : DType::BFloat16;
        staging[slot] = Tensor({maximum_elements[slot]}, dtype, device());
    }

    io::visit_safetensors(path, [&](const io::SafetensorsTensorInfo& info,
                                    std::span<const std::byte> bytes) {
        const auto found = targets_by_source.find(info.name);
        if (found == targets_by_source.end()) return;
        const auto requires_staging = info.dtype != DType::Float32 || std::any_of(
            found->second.begin(), found->second.end(), [](const auto& target) {
                return target.transform == WeightTransform::Transpose2D;
            });
        Tensor source;
        if (requires_staging) {
            const auto slot = dtype_slot(info.dtype);
            source = Tensor::from_storage(staging[slot].storage(), info.shape,
                                          contiguous_strides(info.shape), 0, info.dtype);
            runtime::copy_bytes(source.data(), device(), bytes.data(), Device::cpu(),
                                bytes.size());
        }
        for (auto& target : found->second) {
            if (!requires_staging) {
                runtime::copy_bytes(target.parameter->mutable_data().data(), device(),
                                    bytes.data(), Device::cpu(), bytes.size());
            } else if (target.transform == WeightTransform::Transpose2D) {
                ops::cast_transpose_2d_out_(source, target.parameter->mutable_data());
            } else {
                ops::cast_out_(source, target.parameter->mutable_data());
            }
            target.parameter->zero_grad();
            report.loaded.push_back(target.name);
        }
    });
    runtime::synchronize(device());
    if (report.complete()) impl_->parameters_initialized = true;
    return report;
}

LoadWeightsReport TransformerModel::load_safetensors_files(
    const std::vector<std::filesystem::path>& paths,
    const LoadWeightsOptions& options) {
    return load_state_dict(io::load_safetensors_files(paths, device()), options);
}

LoadWeightsReport TransformerModel::load_safetensors_index(
    const std::filesystem::path& index_path, const LoadWeightsOptions& options) {
    return load_state_dict(io::load_safetensors_index(index_path, device()), options);
}

void TransformerModel::save_safetensors(
    const std::filesystem::path& path,
    const io::SafetensorsSaveOptions& options) {
    io::save_safetensors(path, state_dict(), options);
}

WeightMapping qwen_style_weight_mapping(const ModelConfig& config) {
    config.validate();
    WeightMapping mapping;
    mapping.emplace("token_embedding.weight",
                    WeightSource{"model.embed_tokens.weight", WeightTransform::Identity});
    for (std::int64_t layer = 0; layer < config.layers; ++layer) {
        const auto target = "blocks." + std::to_string(layer);
        const auto source = "model.layers." + std::to_string(layer);
        mapping.emplace(target + ".attention_norm.weight",
                        WeightSource{source + ".input_layernorm.weight",
                                     WeightTransform::Identity});
        mapping.emplace(target + ".attention.q_proj.weight",
                        WeightSource{source + ".self_attn.q_proj.weight",
                                     WeightTransform::Transpose2D});
        mapping.emplace(target + ".attention.k_proj.weight",
                        WeightSource{source + ".self_attn.k_proj.weight",
                                     WeightTransform::Transpose2D});
        mapping.emplace(target + ".attention.v_proj.weight",
                        WeightSource{source + ".self_attn.v_proj.weight",
                                     WeightTransform::Transpose2D});
        if (config.attention_bias) {
            mapping.emplace(target + ".attention.q_proj.bias",
                            WeightSource{source + ".self_attn.q_proj.bias",
                                         WeightTransform::Identity});
            mapping.emplace(target + ".attention.k_proj.bias",
                            WeightSource{source + ".self_attn.k_proj.bias",
                                         WeightTransform::Identity});
            mapping.emplace(target + ".attention.v_proj.bias",
                            WeightSource{source + ".self_attn.v_proj.bias",
                                         WeightTransform::Identity});
        }
        mapping.emplace(target + ".attention.o_proj.weight",
                        WeightSource{source + ".self_attn.o_proj.weight",
                                     WeightTransform::Transpose2D});
        mapping.emplace(target + ".ffn_norm.weight",
                        WeightSource{source + ".post_attention_layernorm.weight",
                                     WeightTransform::Identity});
        mapping.emplace(target + ".feed_forward.gate_proj.weight",
                        WeightSource{source + ".mlp.gate_proj.weight",
                                     WeightTransform::Transpose2D});
        mapping.emplace(target + ".feed_forward.up_proj.weight",
                        WeightSource{source + ".mlp.up_proj.weight",
                                     WeightTransform::Transpose2D});
        mapping.emplace(target + ".feed_forward.down_proj.weight",
                        WeightSource{source + ".mlp.down_proj.weight",
                                     WeightTransform::Transpose2D});
    }
    mapping.emplace("final_norm.weight",
                    WeightSource{"model.norm.weight", WeightTransform::Identity});
    if (!config.tie_embeddings) {
        mapping.emplace("output_head.weight",
                        WeightSource{"lm_head.weight", WeightTransform::Transpose2D});
    }
    return mapping;
}

}  // namespace microllm::model
