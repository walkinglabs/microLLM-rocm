#include <microllm/ops/ops.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include <microllm/runtime/memory.h>

#if MICROLLM_HAS_HIP
#include "hip/kernels.h"
#endif

namespace microllm::ops {
namespace {

void require_float(const Tensor& tensor, const char* name) {
    if (!tensor.defined()) throw std::invalid_argument(std::string(name) + " is undefined");
    if (tensor.dtype() != DType::Float32) {
        throw std::invalid_argument(std::string(name) + " must be float32");
    }
}

void require_forward_float(const Tensor& tensor, const char* name) {
    if (!tensor.defined()) throw std::invalid_argument(std::string(name) + " is undefined");
    if (!is_floating_point(tensor.dtype())) {
        throw std::invalid_argument(std::string(name) + " must have a floating dtype");
    }
}

void require_same_dtype(const Tensor& left, const Tensor& right) {
    if (left.dtype() != right.dtype()) {
        throw std::invalid_argument("tensor dtypes must match; cast explicitly before the operator");
    }
}

void require_readable_hip_dtype(const Tensor& tensor) {
    if (tensor.device().is_hip() && tensor.dtype() != DType::Float32) {
        throw std::runtime_error(
            "readable HIP low-precision kernels are not enabled yet; use CPU reference or cast");
    }
}

void require_same_device(const Tensor& left, const Tensor& right) {
    if (left.device() != right.device()) throw std::invalid_argument("tensor devices must match");
}

void require_contiguous(const Tensor& tensor, const char* name) {
    if (!tensor.is_contiguous()) {
        throw std::invalid_argument(std::string(name) + " must be contiguous for HIP execution");
    }
}

void require_same_shape(const Tensor& left, const Tensor& right) {
    if (left.shape() != right.shape()) throw std::invalid_argument("tensor shapes must match");
}

Tensor from_values(std::vector<float> values, Shape shape,
                   DType dtype = DType::Float32) {
    return Tensor::from_vector(values, std::move(shape), dtype);
}

std::int64_t positive_dim(const Tensor& input, std::int64_t dim) {
    if (dim < 0) dim += input.ndim();
    if (dim < 0 || dim >= input.ndim()) throw std::out_of_range("operator dimension out of range");
    return dim;
}

float sigmoid(float value) {
    if (value >= 0.0F) return 1.0F / (1.0F + std::exp(-value));
    const auto exponential = std::exp(value);
    return exponential / (1.0F + exponential);
}

[[maybe_unused]] bool is_aligned(const void* pointer, std::uintptr_t alignment) {
    return reinterpret_cast<std::uintptr_t>(pointer) % alignment == 0;
}

}  // namespace

Tensor cast(const Tensor& input, DType output_dtype,
            [[maybe_unused]] const OpContext& context) {
    require_forward_float(input, "input");
    if ((input.dtype() != DType::Float32 && input.dtype() != DType::Float16 &&
         input.dtype() != DType::BFloat16) ||
        (output_dtype != DType::Float32 && output_dtype != DType::Float16 &&
         output_dtype != DType::BFloat16)) {
        throw std::invalid_argument("cast supports FP32, FP16, and BF16");
    }
    if (input.dtype() == output_dtype) return input;
    if (input.device().is_cpu()) return input.cast(output_dtype);
    require_contiguous(input, "input");
    Tensor output(input.shape(), output_dtype, input.device());
#if MICROLLM_HAS_HIP
    hip::launch_cast(input.data(), input.dtype(), output.data(), output_dtype,
                     input.numel(), context.native_stream(input.device()));
    return output;
#else
    throw std::runtime_error("microLLM was built without HIP operator support");
#endif
}

void cast_out_(const Tensor& input, Tensor& output,
               [[maybe_unused]] const OpContext& context) {
    require_forward_float(input, "input");
    require_forward_float(output, "output");
    require_same_shape(input, output);
    require_same_device(input, output);
    if (!input.is_contiguous() || !output.is_contiguous()) {
        throw std::invalid_argument("cast_out requires contiguous tensors");
    }
    if (input.device().is_cpu()) {
        const auto converted = input.cast(output.dtype());
        runtime::copy_bytes(output.data(), output.device(), converted.data(),
                            converted.device(), static_cast<std::size_t>(output.numel()) *
                                                    dtype_size(output.dtype()));
        return;
    }
#if MICROLLM_HAS_HIP
    hip::launch_cast(input.data(), input.dtype(), output.data(), output.dtype(),
                     input.numel(), context.native_stream(input.device()));
#else
    throw std::runtime_error("microLLM was built without HIP operator support");
#endif
}

void cast_transpose_2d_out_(const Tensor& input, Tensor& output,
                            [[maybe_unused]] const OpContext& context) {
    require_forward_float(input, "input");
    require_forward_float(output, "output");
    require_same_device(input, output);
    if (input.ndim() != 2 || output.ndim() != 2 ||
        output.shape() != Shape({input.shape()[1], input.shape()[0]})) {
        throw std::invalid_argument("cast transpose output must reverse a rank-two input shape");
    }
    if (!input.is_contiguous() || !output.is_contiguous()) {
        throw std::invalid_argument("cast transpose requires contiguous tensors");
    }
    if (input.device().is_cpu()) {
        const auto converted = input.transpose(0, 1).contiguous().cast(output.dtype());
        runtime::copy_bytes(output.data(), output.device(), converted.data(),
                            converted.device(), static_cast<std::size_t>(output.numel()) *
                                                    dtype_size(output.dtype()));
        return;
    }
#if MICROLLM_HAS_HIP
    hip::launch_cast_transpose_2d(input.data(), input.dtype(), output.data(),
                                  output.dtype(), input.shape()[0], input.shape()[1],
                                  context.native_stream(input.device()));
#else
    throw std::runtime_error("microLLM was built without HIP operator support");
#endif
}

void adamw_update_(Tensor& parameter, const Tensor& gradient,
                   Tensor& first_moment, Tensor& second_moment,
                   float learning_rate, float beta1, float beta2,
                   float epsilon, float weight_decay,
                   float first_correction, float second_correction,
                   [[maybe_unused]] const OpContext& context,
                   AdamWImplementation implementation) {
    require_float(parameter, "parameter");
    require_float(gradient, "gradient");
    require_float(first_moment, "first_moment");
    require_float(second_moment, "second_moment");
    require_same_shape(parameter, gradient);
    require_same_shape(parameter, first_moment);
    require_same_shape(parameter, second_moment);
    require_same_device(parameter, gradient);
    require_same_device(parameter, first_moment);
    require_same_device(parameter, second_moment);
    if (!parameter.is_contiguous() || !gradient.is_contiguous() ||
        !first_moment.is_contiguous() || !second_moment.is_contiguous()) {
        throw std::invalid_argument("AdamW update requires contiguous tensors");
    }
    if (!(learning_rate > 0.0F) || beta1 < 0.0F || beta1 >= 1.0F ||
        beta2 < 0.0F || beta2 >= 1.0F || !(epsilon > 0.0F) ||
        weight_decay < 0.0F || !(first_correction > 0.0F) ||
        !(second_correction > 0.0F)) {
        throw std::invalid_argument("AdamW update hyperparameters are invalid");
    }
    if (parameter.device().is_hip()) {
#if MICROLLM_HAS_HIP
        const auto aligned = is_aligned(parameter.data(), 16) &&
                             is_aligned(gradient.data(), 16) &&
                             is_aligned(first_moment.data(), 16) &&
                             is_aligned(second_moment.data(), 16);
        if (implementation == AdamWImplementation::Vectorized && !aligned) {
            throw std::invalid_argument("vectorized AdamW requires 16-byte aligned tensors");
        }
        const auto vectorized = implementation == AdamWImplementation::Vectorized;
        if (vectorized) {
            hip::launch_adamw_update_vectorized(
                static_cast<float*>(parameter.data()),
                static_cast<const float*>(gradient.data()),
                static_cast<float*>(first_moment.data()),
                static_cast<float*>(second_moment.data()), nullptr,
                parameter.numel(), learning_rate, beta1, beta2, epsilon,
                weight_decay, first_correction, second_correction,
                context.native_stream(parameter.device()));
        } else {
            hip::launch_adamw_update(
                static_cast<float*>(parameter.data()),
                static_cast<const float*>(gradient.data()),
                static_cast<float*>(first_moment.data()),
                static_cast<float*>(second_moment.data()), nullptr,
                parameter.numel(), learning_rate,
                beta1, beta2, epsilon, weight_decay, first_correction,
                second_correction, context.native_stream(parameter.device()));
        }
        return;
#else
        throw std::runtime_error("microLLM was built without HIP operator support");
#endif
    }
    if (implementation == AdamWImplementation::Vectorized) {
        throw std::invalid_argument("vectorized AdamW requires a HIP tensor");
    }
    auto* values = parameter.data_float();
    const auto* gradients = gradient.data_float();
    auto* first = first_moment.data_float();
    auto* second = second_moment.data_float();
    for (std::int64_t index = 0; index < parameter.numel(); ++index) {
        const auto offset = static_cast<std::size_t>(index);
        const auto grad = gradients[offset];
        first[offset] = beta1 * first[offset] + (1.0F - beta1) * grad;
        second[offset] = beta2 * second[offset] + (1.0F - beta2) * grad * grad;
        values[offset] *= 1.0F - learning_rate * weight_decay;
        values[offset] -= learning_rate * (first[offset] / first_correction) /
                          (std::sqrt(second[offset] / second_correction) + epsilon);
    }
}

void adamw_update_bf16_mirror_(Tensor& parameter, const Tensor& gradient,
                               Tensor& first_moment, Tensor& second_moment,
                               Tensor& bf16_mirror, float learning_rate,
                               float beta1, float beta2, float epsilon,
                               float weight_decay, float first_correction,
                               float second_correction,
                               const OpContext& context,
                               AdamWImplementation implementation) {
    if (bf16_mirror.dtype() != DType::BFloat16 ||
        bf16_mirror.shape() != parameter.shape() ||
        bf16_mirror.device() != parameter.device() ||
        !bf16_mirror.is_contiguous()) {
        throw std::invalid_argument(
            "AdamW BF16 mirror must match parameter shape/device and be contiguous");
    }
    if (parameter.device().is_cpu()) {
        adamw_update_(parameter, gradient, first_moment, second_moment,
                      learning_rate, beta1, beta2, epsilon, weight_decay,
                      first_correction, second_correction, context, implementation);
        bf16_mirror = parameter.cast(DType::BFloat16);
        return;
    }
    require_float(parameter, "parameter");
    require_float(gradient, "gradient");
    require_float(first_moment, "first_moment");
    require_float(second_moment, "second_moment");
    require_same_shape(parameter, gradient);
    require_same_shape(parameter, first_moment);
    require_same_shape(parameter, second_moment);
    require_same_device(parameter, gradient);
    require_same_device(parameter, first_moment);
    require_same_device(parameter, second_moment);
    if (!parameter.is_contiguous() || !gradient.is_contiguous() ||
        !first_moment.is_contiguous() || !second_moment.is_contiguous()) {
        throw std::invalid_argument("AdamW update requires contiguous tensors");
    }
    if (!(learning_rate > 0.0F) || beta1 < 0.0F || beta1 >= 1.0F ||
        beta2 < 0.0F || beta2 >= 1.0F || !(epsilon > 0.0F) ||
        weight_decay < 0.0F || !(first_correction > 0.0F) ||
        !(second_correction > 0.0F)) {
        throw std::invalid_argument("AdamW update hyperparameters are invalid");
    }
#if MICROLLM_HAS_HIP
    const auto aligned = is_aligned(parameter.data(), 16) &&
                         is_aligned(gradient.data(), 16) &&
                         is_aligned(first_moment.data(), 16) &&
                         is_aligned(second_moment.data(), 16);
    if (implementation == AdamWImplementation::Vectorized && !aligned) {
        throw std::invalid_argument("vectorized AdamW requires 16-byte aligned tensors");
    }
    const auto vectorized = implementation == AdamWImplementation::Vectorized;
    if (vectorized) {
        hip::launch_adamw_update_vectorized(
            static_cast<float*>(parameter.data()),
            static_cast<const float*>(gradient.data()),
            static_cast<float*>(first_moment.data()),
            static_cast<float*>(second_moment.data()), bf16_mirror.data(),
            parameter.numel(), learning_rate, beta1, beta2, epsilon,
            weight_decay, first_correction, second_correction,
            context.native_stream(parameter.device()));
    } else {
        hip::launch_adamw_update(
            static_cast<float*>(parameter.data()),
            static_cast<const float*>(gradient.data()),
            static_cast<float*>(first_moment.data()),
            static_cast<float*>(second_moment.data()), bf16_mirror.data(),
            parameter.numel(), learning_rate, beta1, beta2, epsilon,
            weight_decay, first_correction, second_correction,
            context.native_stream(parameter.device()));
    }
#else
    throw std::runtime_error("microLLM was built without HIP operator support");
#endif
}

ScaledTensor quantize_fp8_with_scale(
    const Tensor& input, DType fp8_dtype, float scale,
    const Tensor& scale_tensor,
    [[maybe_unused]] const OpContext& context) {
    if (!input.defined() || (input.dtype() != DType::Float32 &&
                             input.dtype() != DType::Float16 &&
                             input.dtype() != DType::BFloat16)) {
        throw std::invalid_argument("FP8 quantize input must be FP32, FP16, or BF16");
    }
    if (!is_fp8_fnuz(fp8_dtype)) {
        throw std::invalid_argument("MI300 FP8 quantize requires an FNUZ dtype");
    }
    if (!std::isfinite(scale) || scale <= 0.0F) {
        throw std::invalid_argument("FP8 quantize scale must be finite and positive");
    }
    if (!input.is_contiguous()) throw std::invalid_argument("FP8 quantize requires contiguous input");
    if (!scale_tensor.defined() || scale_tensor.dtype() != DType::Float32 ||
        scale_tensor.numel() != 1 || scale_tensor.device() != input.device() ||
        !scale_tensor.is_contiguous()) {
        throw std::invalid_argument(
            "FP8 quantize scale Tensor must be one contiguous FP32 value on the input device");
    }
    Tensor output(input.shape(), fp8_dtype, input.device());
    if (input.device().is_cpu()) {
        auto values = input.to_vector();
        for (auto& value : values) value /= scale;
        output = Tensor::from_vector(values, input.shape(), fp8_dtype);
    } else {
#if MICROLLM_HAS_HIP
        hip::launch_quantize_fp8(input.data(), input.dtype(), output.data(), fp8_dtype,
                                 input.numel(), 1.0F / scale,
                                 context.native_stream(input.device()));
#else
        throw std::runtime_error("microLLM was built without HIP operator support");
#endif
    }
    return {std::move(output), scale_tensor, scale};
}

ScaledTensor quantize_fp8(const Tensor& input, DType fp8_dtype, float scale,
                          const OpContext& context) {
    auto scale_tensor = Tensor::from_vector({scale}, {}, DType::Float32);
    if (input.device().is_hip()) scale_tensor = scale_tensor.to(input.device());
    return quantize_fp8_with_scale(
        input, fp8_dtype, scale, scale_tensor, context);
}

ScaledTensor quantize_fp8_dynamic(
    const Tensor& input, DType fp8_dtype, float minimum_scale,
    [[maybe_unused]] const OpContext& context) {
    if (!input.defined() || (input.dtype() != DType::Float32 &&
                             input.dtype() != DType::Float16 &&
                             input.dtype() != DType::BFloat16) ||
        !input.is_contiguous()) {
        throw std::invalid_argument(
            "dynamic FP8 quantize requires contiguous FP32, FP16, or BF16 input");
    }
    if (!is_fp8_fnuz(fp8_dtype) || !std::isfinite(minimum_scale) ||
        minimum_scale <= 0.0F) {
        throw std::invalid_argument(
            "dynamic FP8 quantize requires an FNUZ dtype and positive minimum scale");
    }
    if (input.device().is_cpu()) {
        const auto values = input.to_vector();
        float maximum = 0.0F;
        for (const auto value : values) {
            if (!std::isfinite(value)) {
                throw std::invalid_argument(
                    "dynamic FP8 quantize requires finite input");
            }
            maximum = std::max(maximum, std::abs(value));
        }
        return quantize_fp8(
            input, fp8_dtype, std::max(maximum / 240.0F, minimum_scale),
            context);
    }
    Tensor scale(Shape{}, DType::Float32, input.device());
    Tensor output(input.shape(), fp8_dtype, input.device());
#if MICROLLM_HAS_HIP
    const auto partial_count = std::min<std::int64_t>(
        1024, std::max<std::int64_t>(1, (input.numel() + 255) / 256));
    Tensor partial_maxima({partial_count}, DType::Float32, input.device());
    hip::launch_quantize_fp8_dynamic(
        input.data(), input.dtype(), output.data(), fp8_dtype,
        static_cast<float*>(scale.data()),
        static_cast<float*>(partial_maxima.data()), partial_count,
        input.numel(), minimum_scale,
        context.native_stream(input.device()));
    return {std::move(output), std::move(scale), minimum_scale, false};
#else
    throw std::runtime_error("microLLM was built without HIP operator support");
#endif
}

ScaledTensor quantize_fp8_rows_dynamic(
    const Tensor& input, DType fp8_dtype, float minimum_scale,
    [[maybe_unused]] const OpContext& context) {
    if (!input.defined() || input.ndim() != 2 || !input.is_contiguous() ||
        (input.dtype() != DType::Float32 && input.dtype() != DType::Float16 &&
         input.dtype() != DType::BFloat16) || !is_fp8_fnuz(fp8_dtype) ||
        !std::isfinite(minimum_scale) || minimum_scale <= 0.0F) {
        throw std::invalid_argument(
            "row FP8 quantize requires contiguous rank-two floating input, FNUZ output, and positive minimum scale");
    }
    const auto rows = input.shape()[0];
    const auto columns = input.shape()[1];
    if (input.device().is_cpu()) {
        const auto values = input.to_vector();
        std::vector<float> scales(static_cast<std::size_t>(rows), minimum_scale);
        auto quantized = values;
        for (std::int64_t row = 0; row < rows; ++row) {
            float maximum = 0.0F;
            for (std::int64_t column = 0; column < columns; ++column) {
                const auto value = values[static_cast<std::size_t>(row * columns + column)];
                if (!std::isfinite(value)) throw std::invalid_argument("row FP8 quantize requires finite input");
                maximum = std::max(maximum, std::abs(value));
            }
            scales[static_cast<std::size_t>(row)] =
                std::max(maximum / 240.0F, minimum_scale);
            for (std::int64_t column = 0; column < columns; ++column) {
                quantized[static_cast<std::size_t>(row * columns + column)] /=
                    scales[static_cast<std::size_t>(row)];
            }
        }
        return {Tensor::from_vector(quantized, input.shape(), fp8_dtype),
                Tensor::from_vector(scales, {rows}), minimum_scale, false,
                Fp8ScaleMode::OuterRow};
    }
    Tensor scales({rows}, DType::Float32, input.device());
    Tensor output(input.shape(), fp8_dtype, input.device());
#if MICROLLM_HAS_HIP
    hip::launch_quantize_fp8_rows_dynamic(
        input.data(), input.dtype(), output.data(), fp8_dtype,
        static_cast<float*>(scales.data()), rows, columns, minimum_scale,
        context.native_stream(input.device()));
    return {std::move(output), std::move(scales), minimum_scale, false,
            Fp8ScaleMode::OuterRow};
#else
    throw std::runtime_error("microLLM was built without HIP operator support");
#endif
}

Tensor dequantize_fp8(const ScaledTensor& input, DType output_dtype,
                      [[maybe_unused]] const OpContext& context) {
    if (!input.values.defined() || !is_fp8_fnuz(input.values.dtype())) {
        throw std::invalid_argument("FP8 dequantize requires FNUZ values");
    }
    if (output_dtype != DType::Float32 && output_dtype != DType::Float16 &&
        output_dtype != DType::BFloat16) {
        throw std::invalid_argument("FP8 dequantize output must be FP32, FP16, or BF16");
    }
    const auto expected_scales = input.scale_mode == Fp8ScaleMode::OuterRow
                                     ? input.values.shape()[0] : 1;
    if (!input.scale.defined() || input.scale.dtype() != DType::Float32 ||
        input.scale.numel() != expected_scales || input.scale.device() != input.values.device() ||
        !input.scale.is_contiguous()) {
        throw std::invalid_argument(
            "FP8 dequantize scale Tensor must be one contiguous same-device FP32 value");
    }
    if (input.host_scale_available &&
        (!std::isfinite(input.scale_value) || input.scale_value <= 0.0F)) {
        throw std::invalid_argument(
            "FP8 dequantize host scale must be finite and positive when available");
    }
    Tensor output(input.values.shape(), output_dtype, input.values.device());
    if (input.values.device().is_cpu()) {
        auto values = input.values.to_vector();
        if (input.scale_mode == Fp8ScaleMode::OuterRow) {
            const auto scales = input.scale.to_vector();
            const auto columns = input.values.shape()[1];
            for (std::size_t index = 0; index < values.size(); ++index) {
                values[index] *= scales[index / static_cast<std::size_t>(columns)];
            }
        } else {
            if (!input.host_scale_available) throw std::invalid_argument("CPU scalar FP8 dequantize requires host scale");
            for (auto& value : values) value *= input.scale_value;
        }
        return Tensor::from_vector(values, input.values.shape(), output_dtype);
    }
#if MICROLLM_HAS_HIP
    if (input.scale_mode == Fp8ScaleMode::OuterRow) {
        hip::launch_dequantize_fp8_row_scales(
            input.values.data(), input.values.dtype(), output.data(), output_dtype,
            input.values.shape()[0], input.values.shape()[1],
            static_cast<const float*>(input.scale.data()),
            context.native_stream(input.values.device()));
    } else if (input.host_scale_available) {
        hip::launch_dequantize_fp8(
            input.values.data(), input.values.dtype(), output.data(),
            output_dtype, input.values.numel(), input.scale_value,
            context.native_stream(input.values.device()));
    } else {
        hip::launch_dequantize_fp8_device_scale(
            input.values.data(), input.values.dtype(), output.data(),
            output_dtype, input.values.numel(),
            static_cast<const float*>(input.scale.data()),
            context.native_stream(input.values.device()));
    }
    return output;
#else
    throw std::runtime_error("microLLM was built without HIP operator support");
#endif
}

void fill_(Tensor& tensor, float value, [[maybe_unused]] const OpContext& context) {
    require_forward_float(tensor, "tensor");
    if (tensor.device().is_cpu()) {
        tensor.fill(value);
        return;
    }
    require_contiguous(tensor, "tensor");
#if MICROLLM_HAS_HIP
    hip::launch_fill_typed(tensor.data(), tensor.dtype(), tensor.numel(), value,
                           context.native_stream(tensor.device()));
#else
    throw std::runtime_error("microLLM was built without HIP operator support");
#endif
}

Tensor add(const Tensor& left, const Tensor& right, [[maybe_unused]] const OpContext& context) {
    require_forward_float(left, "left");
    require_forward_float(right, "right");
    require_same_dtype(left, right);
    require_same_shape(left, right);
    require_same_device(left, right);
    if (left.device().is_hip()) {
        require_contiguous(left, "left");
        require_contiguous(right, "right");
        Tensor output(left.shape(), left.dtype(), left.device());
#if MICROLLM_HAS_HIP
        hip::launch_add_typed(left.data(), right.data(), output.data(), left.dtype(),
                              left.numel(), context.native_stream(left.device()));
        return output;
#else
        throw std::runtime_error("microLLM was built without HIP operator support");
#endif
    }
    auto left_values = left.to_vector();
    const auto right_values = right.to_vector();
    for (std::size_t index = 0; index < left_values.size(); ++index) {
        left_values[index] += right_values[index];
    }
    return from_values(std::move(left_values), left.shape(), left.dtype());
}

Tensor add_bias(const Tensor& input, const Tensor& bias,
                [[maybe_unused]] const OpContext& context) {
    require_float(input, "input");
    require_float(bias, "bias");
    require_same_device(input, bias);
    if (input.ndim() == 0 || bias.ndim() != 1 || input.shape().back() != bias.shape()[0]) {
        throw std::invalid_argument("bias must be rank one and match the input last dimension");
    }
    if (input.device().is_hip()) {
        require_contiguous(input, "input");
        require_contiguous(bias, "bias");
        Tensor output(input.shape(), DType::Float32, input.device());
#if MICROLLM_HAS_HIP
        hip::launch_add_bias(static_cast<const float*>(input.data()),
                             static_cast<const float*>(bias.data()),
                             static_cast<float*>(output.data()), input.numel(),
                             bias.numel(), context.native_stream(input.device()));
        return output;
#else
        throw std::runtime_error("microLLM was built without HIP operator support");
#endif
    }
    auto values = input.to_vector();
    const auto bias_values = bias.to_vector();
    const auto width = static_cast<std::size_t>(bias.numel());
    for (std::size_t index = 0; index < values.size(); ++index) {
        values[index] += bias_values[index % width];
    }
    return from_values(std::move(values), input.shape());
}

Tensor bias_gradient(const Tensor& gradient,
                     [[maybe_unused]] const OpContext& context) {
    require_float(gradient, "gradient");
    if (gradient.ndim() == 0) throw std::invalid_argument("bias gradient requires rank one or greater");
    const auto width = gradient.shape().back();
    const auto rows = gradient.numel() / width;
    if (gradient.device().is_hip()) {
        require_contiguous(gradient, "gradient");
        Tensor output({width}, DType::Float32, gradient.device());
#if MICROLLM_HAS_HIP
        hip::launch_bias_gradient(static_cast<const float*>(gradient.data()),
                                  static_cast<float*>(output.data()), rows, width,
                                  context.native_stream(gradient.device()));
        return output;
#else
        throw std::runtime_error("microLLM was built without HIP operator support");
#endif
    }
    const auto values = gradient.to_vector();
    std::vector<float> output(static_cast<std::size_t>(width), 0.0F);
    for (std::int64_t row = 0; row < rows; ++row) {
        for (std::int64_t column = 0; column < width; ++column) {
            output[static_cast<std::size_t>(column)] +=
                values[static_cast<std::size_t>(row * width + column)];
        }
    }
    return from_values(std::move(output), {width});
}

Tensor multiply(const Tensor& left, const Tensor& right,
                [[maybe_unused]] const OpContext& context) {
    require_forward_float(left, "left");
    require_forward_float(right, "right");
    require_same_dtype(left, right);
    require_same_shape(left, right);
    require_same_device(left, right);
    if (left.device().is_hip()) {
        require_contiguous(left, "left");
        require_contiguous(right, "right");
        Tensor output(left.shape(), left.dtype(), left.device());
#if MICROLLM_HAS_HIP
        hip::launch_multiply_typed(left.data(), right.data(), output.data(), left.dtype(),
                                   left.numel(), context.native_stream(left.device()));
        return output;
#else
        throw std::runtime_error("microLLM was built without HIP operator support");
#endif
    }
    auto left_values = left.to_vector();
    const auto right_values = right.to_vector();
    for (std::size_t index = 0; index < left_values.size(); ++index) {
        left_values[index] *= right_values[index];
    }
    return from_values(std::move(left_values), left.shape(), left.dtype());
}

Tensor scale(const Tensor& input, float factor, [[maybe_unused]] const OpContext& context) {
    require_forward_float(input, "input");
    if (input.device().is_hip()) {
        require_contiguous(input, "input");
        Tensor output(input.shape(), input.dtype(), input.device());
#if MICROLLM_HAS_HIP
        hip::launch_scale_typed(input.data(), output.data(), input.dtype(), input.numel(),
                                factor, context.native_stream(input.device()));
        return output;
#else
        throw std::runtime_error("microLLM was built without HIP operator support");
#endif
    }
    auto values = input.to_vector();
    for (auto& value : values) value *= factor;
    return from_values(std::move(values), input.shape(), input.dtype());
}

Tensor matmul(const Tensor& left, const Tensor& right,
              [[maybe_unused]] const OpContext& context) {
    require_forward_float(left, "left");
    require_forward_float(right, "right");
    require_same_dtype(left, right);
    require_same_device(left, right);
    if (left.ndim() < 2 || right.ndim() != left.ndim()) {
        throw std::invalid_argument("matmul requires equal ranks of at least two");
    }
    const auto rank = static_cast<std::size_t>(left.ndim());
    for (std::size_t dim = 0; dim + 2 < rank; ++dim) {
        if (left.shape()[dim] != right.shape()[dim]) {
            throw std::invalid_argument("matmul batch dimensions must match");
        }
    }
    const auto rows = left.shape()[rank - 2];
    const auto inner = left.shape()[rank - 1];
    if (right.shape()[rank - 2] != inner) {
        throw std::invalid_argument("matmul inner dimensions must match");
    }
    const auto columns = right.shape()[rank - 1];
    std::int64_t batches = 1;
    Shape output_shape(left.shape().begin(), left.shape().end() - 2);
    for (const auto dimension : output_shape) batches *= dimension;
    output_shape.push_back(rows);
    output_shape.push_back(columns);

    if (left.device().is_hip()) {
        require_contiguous(left, "left");
        require_contiguous(right, "right");
        Tensor output(output_shape, left.dtype(), left.device());
#if MICROLLM_HAS_HIP
        hip::launch_matmul_typed(left.data(), right.data(), output.data(), left.dtype(),
                                 batches, rows, inner, columns,
                                 context.native_stream(left.device()));
        return output;
#else
        throw std::runtime_error("microLLM was built without HIP operator support");
#endif
    }

    const auto left_values = left.to_vector();
    const auto right_values = right.to_vector();
    std::vector<float> output(static_cast<std::size_t>(batches * rows * columns), 0.0F);
    for (std::int64_t batch = 0; batch < batches; ++batch) {
        const auto left_base = batch * rows * inner;
        const auto right_base = batch * inner * columns;
        const auto output_base = batch * rows * columns;
        for (std::int64_t row = 0; row < rows; ++row) {
            for (std::int64_t column = 0; column < columns; ++column) {
                float sum = 0.0F;
                for (std::int64_t reduction = 0; reduction < inner; ++reduction) {
                    sum += left_values[static_cast<std::size_t>(left_base + row * inner + reduction)] *
                           right_values[static_cast<std::size_t>(right_base + reduction * columns + column)];
                }
                output[static_cast<std::size_t>(output_base + row * columns + column)] = sum;
            }
        }
    }
    return from_values(std::move(output), std::move(output_shape), left.dtype());
}

Tensor embedding(const Tensor& weight, const Tensor& indices,
                 [[maybe_unused]] const OpContext& context) {
    require_forward_float(weight, "weight");
    if (weight.ndim() != 2) throw std::invalid_argument("embedding weight must have rank two");
    if (indices.dtype() != DType::Int32) {
        throw std::invalid_argument("embedding indices must be an int32 tensor");
    }
    require_same_device(weight, indices);
    const auto vocabulary = weight.shape()[0];
    const auto width = weight.shape()[1];
    auto output_shape = indices.shape();
    output_shape.push_back(width);
    if (weight.device().is_hip()) {
        require_readable_hip_dtype(weight);
        require_contiguous(weight, "weight");
        require_contiguous(indices, "indices");
        Tensor output(output_shape, weight.dtype(), weight.device());
#if MICROLLM_HAS_HIP
        hip::launch_embedding(static_cast<const float*>(weight.data()),
                              static_cast<const std::int32_t*>(indices.data()),
                              static_cast<float*>(output.data()), indices.numel(), vocabulary,
                              width, context.native_stream(weight.device()));
        return output;
#else
        throw std::runtime_error("microLLM was built without HIP operator support");
#endif
    }
    const auto weight_values = weight.to_vector();
    const auto index_values = indices.to_int32_vector();
    std::vector<float> output(index_values.size() * static_cast<std::size_t>(width));
    for (std::size_t token = 0; token < index_values.size(); ++token) {
        const auto index = index_values[token];
        if (index < 0 || index >= vocabulary) throw std::out_of_range("embedding index out of range");
        std::copy_n(weight_values.begin() + static_cast<std::ptrdiff_t>(index * width), width,
                    output.begin() + static_cast<std::ptrdiff_t>(token * static_cast<std::size_t>(width)));
    }
    return from_values(std::move(output), std::move(output_shape), weight.dtype());
}

Tensor softmax(const Tensor& input, std::int64_t dim,
               [[maybe_unused]] const OpContext& context) {
    require_forward_float(input, "input");
    const auto normalized = positive_dim(input, dim);
    if (normalized != input.ndim() - 1) {
        throw std::invalid_argument("the readable softmax currently supports the last dimension");
    }
    const auto width = input.shape().back();
    if (width == 0) throw std::invalid_argument("softmax dimension cannot be empty");
    const auto rows = input.numel() / width;
    if (input.device().is_hip()) {
        require_contiguous(input, "input");
        Tensor output(input.shape(), input.dtype(), input.device());
#if MICROLLM_HAS_HIP
        hip::launch_softmax(static_cast<const float*>(input.data()),
                            static_cast<float*>(output.data()), rows, width,
                            context.native_stream(input.device()));
        return output;
#else
        throw std::runtime_error("microLLM was built without HIP operator support");
#endif
    }
    auto output = input.to_vector();
    for (std::int64_t row = 0; row < rows; ++row) {
        const auto begin = output.begin() + static_cast<std::ptrdiff_t>(row * width);
        const auto end = begin + width;
        const auto maximum = *std::max_element(begin, end);
        float denominator = 0.0F;
        for (auto iterator = begin; iterator != end; ++iterator) {
            *iterator = std::exp(*iterator - maximum);
            denominator += *iterator;
        }
        for (auto iterator = begin; iterator != end; ++iterator) *iterator /= denominator;
    }
    return from_values(std::move(output), input.shape(), input.dtype());
}

Tensor rms_norm(const Tensor& input, const Tensor& weight, float epsilon,
                [[maybe_unused]] const OpContext& context) {
    require_forward_float(input, "input");
    require_forward_float(weight, "weight");
    require_same_dtype(input, weight);
    require_same_device(input, weight);
    if (epsilon <= 0.0F) throw std::invalid_argument("rms_norm epsilon must be positive");
    if (weight.ndim() != 1 || input.ndim() == 0 || weight.shape()[0] != input.shape().back()) {
        throw std::invalid_argument("rms_norm weight must match the last input dimension");
    }
    const auto width = input.shape().back();
    const auto rows = input.numel() / width;
    if (input.device().is_hip()) {
        require_readable_hip_dtype(input);
        require_contiguous(input, "input");
        require_contiguous(weight, "weight");
        Tensor output(input.shape(), input.dtype(), input.device());
#if MICROLLM_HAS_HIP
        hip::launch_rms_norm(static_cast<const float*>(input.data()),
                             static_cast<const float*>(weight.data()),
                             static_cast<float*>(output.data()), rows, width, epsilon,
                             context.native_stream(input.device()));
        return output;
#else
        throw std::runtime_error("microLLM was built without HIP operator support");
#endif
    }
    const auto input_values = input.to_vector();
    const auto weights = weight.to_vector();
    std::vector<float> output(input_values.size());
    for (std::int64_t row = 0; row < rows; ++row) {
        float square_sum = 0.0F;
        for (std::int64_t column = 0; column < width; ++column) {
            const auto value = input_values[static_cast<std::size_t>(row * width + column)];
            square_sum += value * value;
        }
        const auto inverse_rms = 1.0F / std::sqrt(square_sum / static_cast<float>(width) + epsilon);
        for (std::int64_t column = 0; column < width; ++column) {
            const auto index = static_cast<std::size_t>(row * width + column);
            output[index] = input_values[index] * inverse_rms * weights[static_cast<std::size_t>(column)];
        }
    }
    return from_values(std::move(output), input.shape(), input.dtype());
}

TensorPair add_rms_norm(const Tensor& left, const Tensor& right,
                        const Tensor& weight, float epsilon,
                        [[maybe_unused]] const OpContext& context) {
    require_forward_float(left, "left");
    require_forward_float(right, "right");
    require_forward_float(weight, "weight");
    require_same_dtype(left, right);
    require_same_shape(left, right);
    require_same_device(left, right);
    require_same_device(left, weight);
    if (left.dtype() != DType::Float32 || weight.dtype() != DType::Float32 ||
        epsilon <= 0.0F || left.ndim() == 0 || weight.ndim() != 1 ||
        weight.shape()[0] != left.shape().back()) {
        throw std::invalid_argument(
            "add_rms_norm requires FP32 equal inputs and last-dimension weight");
    }
    if (left.device().is_hip()) {
        require_contiguous(left, "left");
        require_contiguous(right, "right");
        require_contiguous(weight, "weight");
        Tensor sum(left.shape(), DType::Float32, left.device());
        Tensor normalized(left.shape(), DType::Float32, left.device());
#if MICROLLM_HAS_HIP
        const auto width = left.shape().back();
        hip::launch_add_rms_norm(
            static_cast<const float*>(left.data()),
            static_cast<const float*>(right.data()),
            static_cast<const float*>(weight.data()),
            static_cast<float*>(sum.data()), static_cast<float*>(normalized.data()),
            left.numel() / width, width, epsilon,
            context.native_stream(left.device()));
        return {std::move(sum), std::move(normalized)};
#else
        throw std::runtime_error("microLLM was built without HIP operator support");
#endif
    }
    auto sum = add(left, right, context);
    return {sum, rms_norm(sum, weight, epsilon, context)};
}

Tensor silu(const Tensor& input, [[maybe_unused]] const OpContext& context) {
    require_forward_float(input, "input");
    if (input.device().is_hip()) {
        require_contiguous(input, "input");
        Tensor output(input.shape(), input.dtype(), input.device());
#if MICROLLM_HAS_HIP
        hip::launch_silu_typed(input.data(), output.data(), input.dtype(), input.numel(),
                               context.native_stream(input.device()));
        return output;
#else
        throw std::runtime_error("microLLM was built without HIP operator support");
#endif
    }
    auto values = input.to_vector();
    for (auto& value : values) value *= sigmoid(value);
    return from_values(std::move(values), input.shape(), input.dtype());
}

Tensor swiglu(const Tensor& gate, const Tensor& up,
              [[maybe_unused]] const OpContext& context) {
    require_forward_float(gate, "gate");
    require_forward_float(up, "up");
    require_same_dtype(gate, up);
    require_same_shape(gate, up);
    require_same_device(gate, up);
    if (gate.device().is_hip()) {
        require_contiguous(gate, "gate");
        require_contiguous(up, "up");
        Tensor output(gate.shape(), gate.dtype(), gate.device());
#if MICROLLM_HAS_HIP
        hip::launch_swiglu_typed(gate.data(), up.data(), output.data(), gate.dtype(),
                                 gate.numel(), context.native_stream(gate.device()));
        return output;
#else
        throw std::runtime_error("microLLM was built without HIP operator support");
#endif
    }
    auto gate_values = gate.to_vector();
    const auto up_values = up.to_vector();
    for (std::size_t index = 0; index < gate_values.size(); ++index) {
        gate_values[index] *= sigmoid(gate_values[index]) * up_values[index];
    }
    return from_values(std::move(gate_values), gate.shape(), gate.dtype());
}

Tensor rope(const Tensor& input, std::int64_t sequence_dim, std::int64_t position_offset,
            float base, [[maybe_unused]] const OpContext& context) {
    require_forward_float(input, "input");
    if (input.ndim() < 2) throw std::invalid_argument("rope requires rank two or greater");
    const auto sequence = positive_dim(input, sequence_dim);
    if (sequence == input.ndim() - 1) {
        throw std::invalid_argument("rope sequence dimension cannot be the head dimension");
    }
    if (position_offset < 0 || base <= 0.0F) {
        throw std::invalid_argument("rope position offset and base are invalid");
    }
    const auto head_width = input.shape().back();
    if (head_width % 2 != 0) throw std::invalid_argument("rope head dimension must be even");
    const auto sequence_stride = contiguous_strides(input.shape())[static_cast<std::size_t>(sequence)];
    if (input.device().is_hip()) {
        require_readable_hip_dtype(input);
        require_contiguous(input, "input");
        Tensor output(input.shape(), input.dtype(), input.device());
#if MICROLLM_HAS_HIP
        hip::launch_rope(static_cast<const float*>(input.data()),
                         static_cast<float*>(output.data()), input.numel(), head_width,
                         input.shape()[static_cast<std::size_t>(sequence)], sequence_stride,
                         position_offset, base, context.native_stream(input.device()));
        return output;
#else
        throw std::runtime_error("microLLM was built without HIP operator support");
#endif
    }
    const auto values = input.to_vector();
    auto output = values;
    for (std::int64_t linear = 0; linear < input.numel(); linear += head_width) {
        const auto position = (linear / sequence_stride) % input.shape()[static_cast<std::size_t>(sequence)];
        for (std::int64_t pair = 0; pair < head_width / 2; ++pair) {
            const auto angle = static_cast<float>(position + position_offset) *
                               std::pow(base, -2.0F * static_cast<float>(pair) /
                                                  static_cast<float>(head_width));
            const auto cosine = std::cos(angle);
            const auto sine = std::sin(angle);
            const auto even = static_cast<std::size_t>(linear + pair * 2);
            const auto odd = even + 1;
            output[even] = values[even] * cosine - values[odd] * sine;
            output[odd] = values[even] * sine + values[odd] * cosine;
        }
    }
    return from_values(std::move(output), input.shape(), input.dtype());
}

Tensor rope_split_half(const Tensor& input, std::int64_t sequence_dim,
                       std::int64_t position_offset, float base,
                       [[maybe_unused]] const OpContext& context) {
    require_forward_float(input, "input");
    if (input.ndim() < 2) throw std::invalid_argument("rope requires rank two or greater");
    const auto sequence = positive_dim(input, sequence_dim);
    if (sequence == input.ndim() - 1 || position_offset < 0 || base <= 0.0F ||
        input.shape().back() % 2 != 0) {
        throw std::invalid_argument("split-half rope shape or configuration is invalid");
    }
    const auto head_width = input.shape().back();
    const auto sequence_stride =
        contiguous_strides(input.shape())[static_cast<std::size_t>(sequence)];
    if (input.device().is_hip()) {
        require_readable_hip_dtype(input);
        require_contiguous(input, "input");
        Tensor output(input.shape(), input.dtype(), input.device());
#if MICROLLM_HAS_HIP
        hip::launch_rope_split_half(
            static_cast<const float*>(input.data()), static_cast<float*>(output.data()),
            input.numel(), head_width, input.shape()[static_cast<std::size_t>(sequence)],
            sequence_stride, position_offset, base, context.native_stream(input.device()));
        return output;
#else
        throw std::runtime_error("microLLM was built without HIP operator support");
#endif
    }
    const auto values = input.to_vector();
    auto output = values;
    const auto half = head_width / 2;
    for (std::int64_t linear = 0; linear < input.numel(); linear += head_width) {
        const auto position =
            (linear / sequence_stride) % input.shape()[static_cast<std::size_t>(sequence)];
        for (std::int64_t pair = 0; pair < half; ++pair) {
            const auto angle = static_cast<float>(position + position_offset) *
                               std::pow(base, -2.0F * static_cast<float>(pair) /
                                                  static_cast<float>(head_width));
            const auto cosine = std::cos(angle);
            const auto sine = std::sin(angle);
            const auto first = static_cast<std::size_t>(linear + pair);
            const auto second = static_cast<std::size_t>(linear + pair + half);
            output[first] = values[first] * cosine - values[second] * sine;
            output[second] = values[first] * sine + values[second] * cosine;
        }
    }
    return from_values(std::move(output), input.shape(), input.dtype());
}

Tensor rope_split_half_bias(const Tensor& input, const Tensor& bias,
                            std::int64_t position_offset, float base,
                            [[maybe_unused]] const OpContext& context) {
    require_forward_float(input, "input");
    require_float(bias, "bias");
    require_same_device(input, bias);
    if (input.dtype() != DType::Float32 || bias.dtype() != DType::Float32 ||
        input.ndim() != 4 || bias.ndim() != 1 || input.shape()[3] % 2 != 0 ||
        bias.shape()[0] != input.shape()[1] * input.shape()[3] ||
        position_offset < 0 || base <= 0.0F) {
        throw std::invalid_argument(
            "split-half rope+bias requires FP32 [B,H,T,even-D] and bias [H*D]");
    }
    const auto heads = input.shape()[1];
    const auto sequence = input.shape()[2];
    const auto head_width = input.shape()[3];
    if (input.device().is_hip()) {
        require_contiguous(input, "input");
        require_contiguous(bias, "bias");
        Tensor output(input.shape(), DType::Float32, input.device());
#if MICROLLM_HAS_HIP
        hip::launch_rope_split_half_bias(
            static_cast<const float*>(input.data()),
            static_cast<const float*>(bias.data()),
            static_cast<float*>(output.data()), input.numel(), heads, sequence,
            head_width, position_offset, base, context.native_stream(input.device()));
        return output;
#else
        throw std::runtime_error("microLLM was built without HIP operator support");
#endif
    }
    const auto values = input.to_vector();
    const auto bias_values = bias.to_vector();
    auto output = values;
    const auto half = head_width / 2;
    for (std::int64_t batch = 0; batch < input.shape()[0]; ++batch) {
        for (std::int64_t head = 0; head < heads; ++head) {
            for (std::int64_t position = 0; position < sequence; ++position) {
                const auto row = ((batch * heads + head) * sequence + position) * head_width;
                const auto bias_row = head * head_width;
                for (std::int64_t pair = 0; pair < half; ++pair) {
                    const auto angle = static_cast<float>(position + position_offset) *
                                       std::pow(base, -2.0F * static_cast<float>(pair) /
                                                          static_cast<float>(head_width));
                    const auto cosine = std::cos(angle);
                    const auto sine = std::sin(angle);
                    const auto first = static_cast<std::size_t>(row + pair);
                    const auto second = static_cast<std::size_t>(row + pair + half);
                    const auto first_value = values[first] +
                                             bias_values[static_cast<std::size_t>(bias_row + pair)];
                    const auto second_value = values[second] +
                                              bias_values[static_cast<std::size_t>(bias_row + pair + half)];
                    output[first] = first_value * cosine - second_value * sine;
                    output[second] = first_value * sine + second_value * cosine;
                }
            }
        }
    }
    return from_values(std::move(output), input.shape());
}

Tensor rope_positions(const Tensor& input, const Tensor& positions, float base,
                      [[maybe_unused]] const OpContext& context) {
    require_float(input, "input");
    require_same_device(input, positions);
    if (positions.dtype() != DType::Int32 || input.ndim() != 4 ||
        input.shape()[0] <= 0 || input.shape()[2] != 1 ||
        input.shape()[3] % 2 != 0 || positions.shape() != Shape({input.shape()[0]}) ||
        base <= 0.0F) {
        throw std::invalid_argument(
            "positioned rope requires FP32 [B,H,1,even-D] and Int32 [B]");
    }
    require_contiguous(input, "input");
    require_contiguous(positions, "positions");
    const auto batches = input.shape()[0];
    const auto heads = input.shape()[1];
    const auto width = input.shape()[3];
    if (input.device().is_hip()) {
        Tensor output(input.shape(), DType::Float32, input.device());
#if MICROLLM_HAS_HIP
        hip::launch_rope_positions(
            static_cast<const float*>(input.data()),
            static_cast<const std::int32_t*>(positions.data()),
            static_cast<float*>(output.data()), batches, heads, width, base,
            context.native_stream(input.device()));
        return output;
#else
        throw std::runtime_error("microLLM was built without HIP operator support");
#endif
    }
    const auto values = input.to_vector();
    const auto offsets = positions.to_int32_vector();
    auto output = values;
    for (std::int64_t batch = 0; batch < batches; ++batch) {
        const auto position = offsets[static_cast<std::size_t>(batch)];
        if (position < 0) throw std::out_of_range("rope position cannot be negative");
        for (std::int64_t head = 0; head < heads; ++head) {
            const auto row = (batch * heads + head) * width;
            for (std::int64_t pair = 0; pair < width / 2; ++pair) {
                const auto angle = static_cast<float>(position) *
                                   std::pow(base, -2.0F * static_cast<float>(pair) /
                                                      static_cast<float>(width));
                const auto cosine = std::cos(angle);
                const auto sine = std::sin(angle);
                const auto first = static_cast<std::size_t>(row + pair * 2);
                const auto second = first + 1;
                output[first] = values[first] * cosine - values[second] * sine;
                output[second] = values[first] * sine + values[second] * cosine;
            }
        }
    }
    return Tensor::from_vector(output, input.shape());
}

Tensor rope_split_half_positions(const Tensor& input, const Tensor& positions,
                                 float base,
                                 [[maybe_unused]] const OpContext& context) {
    require_float(input, "input");
    require_same_device(input, positions);
    if (positions.dtype() != DType::Int32 || input.ndim() != 4 ||
        input.shape()[0] <= 0 || input.shape()[2] != 1 ||
        input.shape()[3] % 2 != 0 || positions.shape() != Shape({input.shape()[0]}) ||
        base <= 0.0F) {
        throw std::invalid_argument(
            "positioned split-half rope requires FP32 [B,H,1,even-D] and Int32 [B]");
    }
    require_contiguous(input, "input");
    require_contiguous(positions, "positions");
    const auto batches = input.shape()[0];
    const auto heads = input.shape()[1];
    const auto width = input.shape()[3];
    if (input.device().is_hip()) {
        Tensor output(input.shape(), DType::Float32, input.device());
#if MICROLLM_HAS_HIP
        hip::launch_rope_split_half_positions(
            static_cast<const float*>(input.data()),
            static_cast<const std::int32_t*>(positions.data()),
            static_cast<float*>(output.data()), batches, heads, width, base,
            context.native_stream(input.device()));
        return output;
#else
        throw std::runtime_error("microLLM was built without HIP operator support");
#endif
    }
    const auto values = input.to_vector();
    const auto offsets = positions.to_int32_vector();
    auto output = values;
    const auto half = width / 2;
    for (std::int64_t batch = 0; batch < batches; ++batch) {
        const auto position = offsets[static_cast<std::size_t>(batch)];
        if (position < 0) throw std::out_of_range("rope position cannot be negative");
        for (std::int64_t head = 0; head < heads; ++head) {
            const auto row = (batch * heads + head) * width;
            for (std::int64_t pair = 0; pair < half; ++pair) {
                const auto angle = static_cast<float>(position) *
                                   std::pow(base, -2.0F * static_cast<float>(pair) /
                                                      static_cast<float>(width));
                const auto cosine = std::cos(angle);
                const auto sine = std::sin(angle);
                const auto first = static_cast<std::size_t>(row + pair);
                const auto second = static_cast<std::size_t>(row + pair + half);
                output[first] = values[first] * cosine - values[second] * sine;
                output[second] = values[first] * sine + values[second] * cosine;
            }
        }
    }
    return Tensor::from_vector(output, input.shape());
}

Tensor rope_split_half_bias_positions(
    const Tensor& input, const Tensor& bias, const Tensor& positions,
    float base, [[maybe_unused]] const OpContext& context) {
    require_float(input, "input");
    require_float(bias, "bias");
    require_same_device(input, bias);
    require_same_device(input, positions);
    if (positions.dtype() != DType::Int32 || input.ndim() != 4 ||
        input.shape()[0] <= 0 || input.shape()[2] != 1 ||
        input.shape()[3] % 2 != 0 || bias.ndim() != 1 ||
        bias.shape()[0] != input.shape()[1] * input.shape()[3] ||
        positions.shape() != Shape({input.shape()[0]}) || base <= 0.0F) {
        throw std::invalid_argument(
            "positioned split-half rope+bias requires input [B,H,1,D], bias [H*D], positions [B]");
    }
    require_contiguous(input, "input");
    require_contiguous(bias, "bias");
    require_contiguous(positions, "positions");
    const auto batches = input.shape()[0];
    const auto heads = input.shape()[1];
    const auto width = input.shape()[3];
    if (input.device().is_hip()) {
        Tensor output(input.shape(), DType::Float32, input.device());
#if MICROLLM_HAS_HIP
        hip::launch_rope_split_half_bias_positions(
            static_cast<const float*>(input.data()),
            static_cast<const float*>(bias.data()),
            static_cast<const std::int32_t*>(positions.data()),
            static_cast<float*>(output.data()), batches, heads, width, base,
            context.native_stream(input.device()));
        return output;
#else
        throw std::runtime_error("microLLM was built without HIP operator support");
#endif
    }
    const auto values = input.to_vector();
    const auto bias_values = bias.to_vector();
    const auto offsets = positions.to_int32_vector();
    auto output = values;
    const auto half = width / 2;
    for (std::int64_t batch = 0; batch < batches; ++batch) {
        const auto position = offsets[static_cast<std::size_t>(batch)];
        if (position < 0) throw std::out_of_range("rope position cannot be negative");
        for (std::int64_t head = 0; head < heads; ++head) {
            const auto row = (batch * heads + head) * width;
            const auto bias_row = head * width;
            for (std::int64_t pair = 0; pair < half; ++pair) {
                const auto angle = static_cast<float>(position) *
                                   std::pow(base, -2.0F * static_cast<float>(pair) /
                                                      static_cast<float>(width));
                const auto cosine = std::cos(angle);
                const auto sine = std::sin(angle);
                const auto first = static_cast<std::size_t>(row + pair);
                const auto second = static_cast<std::size_t>(row + pair + half);
                const auto first_value =
                    values[first] + bias_values[static_cast<std::size_t>(bias_row + pair)];
                const auto second_value =
                    values[second] +
                    bias_values[static_cast<std::size_t>(bias_row + pair + half)];
                output[first] = first_value * cosine - second_value * sine;
                output[second] = first_value * sine + second_value * cosine;
            }
        }
    }
    return Tensor::from_vector(output, input.shape());
}

Tensor cross_entropy(const Tensor& logits, const Tensor& targets,
                     [[maybe_unused]] const OpContext& context) {
    require_forward_float(logits, "logits");
    if (targets.dtype() != DType::Int32) {
        throw std::invalid_argument("cross_entropy targets must be int32");
    }
    require_same_device(logits, targets);
    if (logits.ndim() < 1 || logits.shape().back() <= 0) {
        throw std::invalid_argument("cross_entropy logits require a non-empty class dimension");
    }
    Shape expected(logits.shape().begin(), logits.shape().end() - 1);
    if (targets.shape() != expected) throw std::invalid_argument("target shape must match logits prefix");
    const auto classes = logits.shape().back();
    const auto rows = logits.numel() / classes;
    if (rows == 0) throw std::invalid_argument("cross_entropy requires at least one target");
    if (logits.device().is_hip()) {
        require_readable_hip_dtype(logits);
        require_contiguous(logits, "logits");
        require_contiguous(targets, "targets");
        Tensor output(Shape{}, DType::Float32, logits.device());
        Tensor row_data({rows, 2}, DType::Float32, logits.device());
#if MICROLLM_HAS_HIP
        hip::launch_cross_entropy(static_cast<const float*>(logits.data()),
                                  static_cast<const std::int32_t*>(targets.data()),
                                  static_cast<float*>(output.data()),
                                  static_cast<float*>(row_data.data()), rows, classes,
                                  context.native_stream(logits.device()));
        return output;
#else
        throw std::runtime_error("microLLM was built without HIP operator support");
#endif
    }
    const auto values = logits.to_vector();
    const auto labels = targets.to_int32_vector();
    double total = 0.0;
    std::int64_t valid_rows = 0;
    for (std::int64_t row = 0; row < rows; ++row) {
        const auto label = labels[static_cast<std::size_t>(row)];
        if (label == -100) continue;
        if (label < 0 || label >= classes) throw std::out_of_range("target class out of range");
        ++valid_rows;
        float maximum = -std::numeric_limits<float>::infinity();
        for (std::int64_t column = 0; column < classes; ++column) {
            maximum = std::max(maximum, values[static_cast<std::size_t>(row * classes + column)]);
        }
        double exponential_sum = 0.0;
        for (std::int64_t column = 0; column < classes; ++column) {
            exponential_sum += std::exp(static_cast<double>(values[static_cast<std::size_t>(row * classes + column)] - maximum));
        }
        const auto selected = values[static_cast<std::size_t>(row * classes + label)];
        total += std::log(exponential_sum) + static_cast<double>(maximum - selected);
    }
    if (valid_rows == 0) throw std::invalid_argument("cross_entropy has no non-ignored targets");
    return from_values({static_cast<float>(total / static_cast<double>(valid_rows))}, {});
}

Tensor reduce_sum(const Tensor& input, [[maybe_unused]] const OpContext& context) {
    require_float(input, "input");
    if (input.device().is_hip()) {
        require_contiguous(input, "input");
        Tensor output(Shape{}, DType::Float32, input.device());
#if MICROLLM_HAS_HIP
        hip::launch_reduce_sum(static_cast<const float*>(input.data()),
                               static_cast<float*>(output.data()), input.numel(),
                               context.native_stream(input.device()));
        return output;
#else
        throw std::runtime_error("microLLM was built without HIP operator support");
#endif
    }
    double total = 0.0;
    for (const auto value : input.to_vector()) total += value;
    return from_values({static_cast<float>(total)}, {});
}

Tensor broadcast_scalar(const Tensor& scalar, Shape shape,
                        [[maybe_unused]] const OpContext& context) {
    require_float(scalar, "scalar");
    if (scalar.numel() != 1) throw std::invalid_argument("broadcast source must be scalar");
    Tensor output(std::move(shape), DType::Float32, scalar.device());
    if (scalar.device().is_hip()) {
        require_contiguous(scalar, "scalar");
#if MICROLLM_HAS_HIP
        hip::launch_broadcast_scalar(static_cast<const float*>(scalar.data()),
                                     static_cast<float*>(output.data()), output.numel(),
                                     context.native_stream(scalar.device()));
        return output;
#else
        throw std::runtime_error("microLLM was built without HIP operator support");
#endif
    }
    output.fill(scalar.to_vector()[0]);
    return output;
}

Tensor embedding_backward(const Tensor& gradient, const Tensor& indices,
                          std::int64_t vocabulary,
                          [[maybe_unused]] const OpContext& context) {
    require_float(gradient, "gradient");
    if (indices.dtype() != DType::Int32 || indices.device() != gradient.device()) {
        throw std::invalid_argument("embedding backward indices must be int32 on gradient device");
    }
    if (vocabulary <= 0 || gradient.ndim() < 1 ||
        gradient.numel() != indices.numel() * gradient.shape().back()) {
        throw std::invalid_argument("embedding backward shape or vocabulary is invalid");
    }
    const auto width = gradient.shape().back();
    Tensor output({vocabulary, width}, DType::Float32, gradient.device());
    fill_(output, 0.0F, context);
    if (gradient.device().is_hip()) {
        require_contiguous(gradient, "gradient");
        require_contiguous(indices, "indices");
#if MICROLLM_HAS_HIP
        hip::launch_embedding_backward(
            static_cast<const float*>(gradient.data()),
            static_cast<const std::int32_t*>(indices.data()),
            static_cast<float*>(output.data()), indices.numel(), vocabulary, width,
            context.native_stream(gradient.device()));
        return output;
#else
        throw std::runtime_error("microLLM was built without HIP operator support");
#endif
    }
    const auto values = gradient.to_vector();
    const auto labels = indices.to_int32_vector();
    auto result = output.to_vector();
    for (std::size_t token = 0; token < labels.size(); ++token) {
        const auto label = static_cast<std::int64_t>(labels[token]);
        if (label < 0 || label >= vocabulary) throw std::out_of_range("embedding index out of range");
        for (std::int64_t column = 0; column < width; ++column) {
            result[static_cast<std::size_t>(label * width + column)] +=
                values[token * static_cast<std::size_t>(width) +
                       static_cast<std::size_t>(column)];
        }
    }
    return from_values(std::move(result), output.shape());
}

Tensor softmax_backward(const Tensor& output, const Tensor& gradient,
                        [[maybe_unused]] const OpContext& context) {
    require_float(output, "output");
    require_float(gradient, "gradient");
    require_same_shape(output, gradient);
    require_same_device(output, gradient);
    if (output.ndim() == 0 || output.shape().back() == 0) {
        throw std::invalid_argument("softmax backward requires a non-empty final dimension");
    }
    const auto width = output.shape().back();
    const auto rows = output.numel() / width;
    if (output.device().is_hip()) {
        require_contiguous(output, "output");
        require_contiguous(gradient, "gradient");
        Tensor input_gradient(output.shape(), DType::Float32, output.device());
#if MICROLLM_HAS_HIP
        hip::launch_softmax_backward(
            static_cast<const float*>(output.data()),
            static_cast<const float*>(gradient.data()),
            static_cast<float*>(input_gradient.data()), rows, width,
            context.native_stream(output.device()));
        return input_gradient;
#else
        throw std::runtime_error("microLLM was built without HIP operator support");
#endif
    }
    const auto probabilities = output.to_vector();
    const auto output_gradient = gradient.to_vector();
    std::vector<float> input_gradient(output_gradient.size());
    for (std::int64_t row = 0; row < rows; ++row) {
        float dot = 0.0F;
        for (std::int64_t column = 0; column < width; ++column) {
            const auto index = static_cast<std::size_t>(row * width + column);
            dot += output_gradient[index] * probabilities[index];
        }
        for (std::int64_t column = 0; column < width; ++column) {
            const auto index = static_cast<std::size_t>(row * width + column);
            input_gradient[index] = probabilities[index] * (output_gradient[index] - dot);
        }
    }
    return from_values(std::move(input_gradient), output.shape());
}

TensorPair rms_norm_backward(const Tensor& input, const Tensor& weight,
                             const Tensor& gradient, float epsilon,
                             [[maybe_unused]] const OpContext& context) {
    require_float(input, "input");
    require_float(weight, "weight");
    require_float(gradient, "gradient");
    require_same_shape(input, gradient);
    require_same_device(input, weight);
    require_same_device(input, gradient);
    if (epsilon <= 0.0F || input.ndim() == 0 || weight.ndim() != 1 ||
        weight.shape()[0] != input.shape().back()) {
        throw std::invalid_argument("rms_norm backward shape or epsilon is invalid");
    }
    const auto width = input.shape().back();
    const auto rows = input.numel() / width;
    if (input.device().is_hip()) {
        require_contiguous(input, "input");
        require_contiguous(weight, "weight");
        require_contiguous(gradient, "gradient");
        Tensor input_gradient(input.shape(), DType::Float32, input.device());
        Tensor weight_gradient(weight.shape(), DType::Float32, input.device());
        Tensor row_inverse_rms({rows}, DType::Float32, input.device());
#if MICROLLM_HAS_HIP
        hip::launch_rms_norm_backward(
            static_cast<const float*>(input.data()),
            static_cast<const float*>(weight.data()),
            static_cast<const float*>(gradient.data()),
            static_cast<float*>(input_gradient.data()),
            static_cast<float*>(weight_gradient.data()),
            static_cast<float*>(row_inverse_rms.data()), rows, width, epsilon,
            context.native_stream(input.device()));
        return {std::move(input_gradient), std::move(weight_gradient)};
#else
        throw std::runtime_error("microLLM was built without HIP operator support");
#endif
    }
    const auto input_values = input.to_vector();
    const auto weight_values = weight.to_vector();
    const auto output_gradient = gradient.to_vector();
    std::vector<float> input_gradient(input_values.size());
    std::vector<float> weight_gradient(static_cast<std::size_t>(width), 0.0F);
    for (std::int64_t row = 0; row < rows; ++row) {
        float square_sum = 0.0F;
        float weighted_dot = 0.0F;
        for (std::int64_t column = 0; column < width; ++column) {
            const auto index = static_cast<std::size_t>(row * width + column);
            square_sum += input_values[index] * input_values[index];
            weighted_dot += output_gradient[index] * weight_values[static_cast<std::size_t>(column)] *
                            input_values[index];
        }
        const auto inverse_rms =
            1.0F / std::sqrt(square_sum / static_cast<float>(width) + epsilon);
        const auto correction = inverse_rms * inverse_rms * inverse_rms * weighted_dot /
                                static_cast<float>(width);
        for (std::int64_t column = 0; column < width; ++column) {
            const auto index = static_cast<std::size_t>(row * width + column);
            input_gradient[index] = output_gradient[index] *
                                        weight_values[static_cast<std::size_t>(column)] * inverse_rms -
                                    input_values[index] * correction;
            weight_gradient[static_cast<std::size_t>(column)] +=
                output_gradient[index] * input_values[index] * inverse_rms;
        }
    }
    return {from_values(std::move(input_gradient), input.shape()),
            from_values(std::move(weight_gradient), weight.shape())};
}

void kv_cache_store_(Tensor& cache, const Tensor& current, std::int64_t position,
                     [[maybe_unused]] const OpContext& context) {
    require_forward_float(cache, "cache");
    require_float(current, "current");
    require_same_device(cache, current);
    if ((cache.dtype() != DType::Float32 && cache.dtype() != DType::BFloat16) ||
        current.dtype() != DType::Float32 ||
        cache.ndim() != 4 || current.ndim() != 4 || cache.shape()[0] <= 0 ||
        cache.shape()[0] != current.shape()[0] || current.shape()[2] != 1 ||
        cache.shape()[1] != current.shape()[1] ||
        cache.shape()[3] != current.shape()[3] || position < 0 ||
        position >= cache.shape()[2] || position != cache.shape()[2] - 1 ||
        cache.stride(3) != 1 || cache.stride(2) != cache.shape()[3]) {
        throw std::invalid_argument("KV cache store shape, dtype, layout, or position is invalid");
    }
    require_contiguous(current, "current");
    const auto batches = cache.shape()[0];
    const auto heads = cache.shape()[1];
    const auto width = cache.shape()[3];
    const auto capacity = cache.stride(1) / width;
    if (capacity < cache.shape()[2]) {
        throw std::invalid_argument("KV cache backing capacity is smaller than its prefix");
    }
    if (cache.device().is_hip()) {
#if MICROLLM_HAS_HIP
        hip::launch_kv_cache_store(
            static_cast<const float*>(current.data()), cache.data(), cache.dtype(),
            batches, heads, capacity, width, position,
            context.native_stream(cache.device()));
        return;
#else
        throw std::runtime_error("microLLM was built without HIP operator support");
#endif
    }
    const auto converted = current.cast(cache.dtype());
    const auto element_bytes = dtype_size(cache.dtype());
    const auto row_bytes = static_cast<std::size_t>(width) * element_bytes;
    const auto* source = static_cast<const std::byte*>(converted.data());
    auto* destination = static_cast<std::byte*>(cache.data());
    for (std::int64_t batch = 0; batch < batches; ++batch) {
        for (std::int64_t head = 0; head < heads; ++head) {
            const auto source_offset =
                static_cast<std::size_t>((batch * heads + head) * width) * element_bytes;
            const auto destination_offset = static_cast<std::size_t>(
                batch * cache.stride(0) + head * cache.stride(1) +
                position * cache.stride(2)) * element_bytes;
            runtime::copy_bytes(destination + destination_offset, cache.device(),
                                source + source_offset, converted.device(), row_bytes);
        }
    }
}

void kv_cache_store_pair_(Tensor& key_cache, Tensor& value_cache,
                          const Tensor& current_key, const Tensor& current_value,
                          std::int64_t position,
                          [[maybe_unused]] const OpContext& context) {
    require_same_shape(key_cache, value_cache);
    require_same_dtype(key_cache, value_cache);
    require_same_shape(current_key, current_value);
    require_same_dtype(current_key, current_value);
    require_same_device(key_cache, value_cache);
    require_same_device(current_key, current_value);
    require_same_device(key_cache, current_key);
    if (!key_cache.device().is_hip()) {
        kv_cache_store_(key_cache, current_key, position, context);
        kv_cache_store_(value_cache, current_value, position, context);
        return;
    }
    if ((key_cache.dtype() != DType::Float32 &&
         key_cache.dtype() != DType::BFloat16) ||
        current_key.dtype() != DType::Float32 ||
        key_cache.ndim() != 4 || current_key.ndim() != 4 ||
        key_cache.shape()[0] <= 0 ||
        key_cache.shape()[0] != current_key.shape()[0] ||
        current_key.shape()[2] != 1 || key_cache.shape()[1] != current_key.shape()[1] ||
        key_cache.shape()[3] != current_key.shape()[3] || position < 0 ||
        position >= key_cache.shape()[2] || position != key_cache.shape()[2] - 1 ||
        key_cache.stride(3) != 1 || key_cache.stride(2) != key_cache.shape()[3] ||
        key_cache.strides() != value_cache.strides()) {
        throw std::invalid_argument("paired KV cache store contract is invalid");
    }
    require_contiguous(current_key, "current_key");
    require_contiguous(current_value, "current_value");
    [[maybe_unused]] const auto batches = key_cache.shape()[0];
    [[maybe_unused]] const auto heads = key_cache.shape()[1];
    const auto width = key_cache.shape()[3];
    const auto capacity = key_cache.stride(1) / width;
    if (capacity < key_cache.shape()[2]) {
        throw std::invalid_argument("paired KV cache capacity is too small");
    }
#if MICROLLM_HAS_HIP
    hip::launch_kv_cache_store_pair(
        static_cast<const float*>(current_key.data()),
        static_cast<const float*>(current_value.data()),
        key_cache.data(), value_cache.data(), key_cache.dtype(), batches, heads,
        capacity, width, position,
        context.native_stream(key_cache.device()));
#else
    throw std::runtime_error("microLLM was built without HIP operator support");
#endif
}

void kv_cache_store_pair_positions_(
    Tensor& key_cache, Tensor& value_cache, const Tensor& current_key,
    const Tensor& current_value, const Tensor& positions,
    const Tensor& cache_rows, [[maybe_unused]] const OpContext& context) {
    require_same_shape(key_cache, value_cache);
    require_same_dtype(key_cache, value_cache);
    require_same_shape(current_key, current_value);
    require_same_dtype(current_key, current_value);
    require_same_device(key_cache, value_cache);
    require_same_device(current_key, current_value);
    require_same_device(key_cache, current_key);
    require_same_device(key_cache, positions);
    require_same_device(key_cache, cache_rows);
    const auto active = current_key.shape().empty() ? 0 : current_key.shape()[0];
    if ((key_cache.dtype() != DType::Float32 &&
         key_cache.dtype() != DType::BFloat16) ||
        current_key.dtype() != DType::Float32 || key_cache.ndim() != 4 ||
        current_key.ndim() != 4 || key_cache.shape()[0] <= 0 || active <= 0 ||
        current_key.shape()[2] != 1 ||
        key_cache.shape()[1] != current_key.shape()[1] ||
        key_cache.shape()[3] != current_key.shape()[3] ||
        positions.dtype() != DType::Int32 || cache_rows.dtype() != DType::Int32 ||
        positions.shape() != Shape({active}) ||
        cache_rows.shape() != Shape({active}) ||
        key_cache.stride(3) != 1 ||
        key_cache.stride(2) != key_cache.shape()[3] ||
        key_cache.strides() != value_cache.strides()) {
        throw std::invalid_argument("positioned paired KV store contract is invalid");
    }
    require_contiguous(current_key, "current_key");
    require_contiguous(current_value, "current_value");
    require_contiguous(positions, "positions");
    require_contiguous(cache_rows, "cache_rows");
    const auto cache_batches = key_cache.shape()[0];
    const auto heads = key_cache.shape()[1];
    const auto logical_prefix = key_cache.shape()[2];
    const auto width = key_cache.shape()[3];
    const auto capacity = key_cache.stride(1) / width;
    if (logical_prefix <= 0 || capacity < logical_prefix) {
        throw std::invalid_argument("positioned paired KV cache capacity is invalid");
    }
    if (key_cache.device().is_hip()) {
#if MICROLLM_HAS_HIP
        hip::launch_kv_cache_store_pair_positions(
            static_cast<const float*>(current_key.data()),
            static_cast<const float*>(current_value.data()),
            static_cast<const std::int32_t*>(positions.data()),
            static_cast<const std::int32_t*>(cache_rows.data()),
            key_cache.data(), value_cache.data(), key_cache.dtype(), active,
            cache_batches, heads, capacity, logical_prefix, width,
            context.native_stream(key_cache.device()));
        return;
#else
        throw std::runtime_error("microLLM was built without HIP operator support");
#endif
    }
    const auto position_values = positions.to_int32_vector();
    const auto row_values = cache_rows.to_int32_vector();
    const auto converted_key = current_key.cast(key_cache.dtype());
    const auto converted_value = current_value.cast(value_cache.dtype());
    const auto element_bytes = dtype_size(key_cache.dtype());
    const auto row_bytes = static_cast<std::size_t>(width) * element_bytes;
    const auto* source_key = static_cast<const std::byte*>(converted_key.data());
    const auto* source_value = static_cast<const std::byte*>(converted_value.data());
    auto* destination_key = static_cast<std::byte*>(key_cache.data());
    auto* destination_value = static_cast<std::byte*>(value_cache.data());
    for (std::int64_t index = 0; index < active; ++index) {
        const auto row = row_values[static_cast<std::size_t>(index)];
        const auto position = position_values[static_cast<std::size_t>(index)];
        if (row < 0 || row >= cache_batches || position < 0 ||
            position >= logical_prefix) {
            throw std::out_of_range("positioned KV store row or position is invalid");
        }
        for (std::int64_t head = 0; head < heads; ++head) {
            const auto source_offset = static_cast<std::size_t>(
                (index * heads + head) * width) * element_bytes;
            const auto destination_offset = static_cast<std::size_t>(
                row * key_cache.stride(0) + head * key_cache.stride(1) +
                position * key_cache.stride(2)) * element_bytes;
            runtime::copy_bytes(destination_key + destination_offset,
                                key_cache.device(), source_key + source_offset,
                                converted_key.device(), row_bytes);
            runtime::copy_bytes(destination_value + destination_offset,
                                value_cache.device(), source_value + source_offset,
                                converted_value.device(), row_bytes);
        }
    }
}

Tensor cached_gqa_attention(const Tensor& query, const Tensor& key_cache,
                            const Tensor& value_cache, std::int64_t repeats,
                            float factor, [[maybe_unused]] const OpContext& context) {
    require_float(query, "query");
    require_forward_float(key_cache, "key_cache");
    require_forward_float(value_cache, "value_cache");
    require_same_device(query, key_cache);
    require_same_device(query, value_cache);
    require_same_shape(key_cache, value_cache);
    require_same_dtype(key_cache, value_cache);
    if (query.dtype() != DType::Float32 ||
        (key_cache.dtype() != DType::Float32 &&
         key_cache.dtype() != DType::BFloat16) || query.ndim() != 4 ||
        key_cache.ndim() != 4 || query.shape()[0] <= 0 || query.shape()[2] != 1 ||
        key_cache.shape()[0] != query.shape()[0] ||
        query.shape()[3] != key_cache.shape()[3] ||
        key_cache.shape()[2] <= 0 || repeats <= 0 ||
        query.shape()[1] != key_cache.shape()[1] * repeats ||
        !std::isfinite(factor) || factor <= 0.0F || key_cache.strides() != value_cache.strides()) {
        throw std::invalid_argument("cached GQA attention shape, dtype, or scale is invalid");
    }
    require_contiguous(query, "query");
    const auto batches = query.shape()[0];
    const auto heads = query.shape()[1];
    [[maybe_unused]] const auto kv_heads = key_cache.shape()[1];
    const auto sequence = key_cache.shape()[2];
    const auto width = query.shape()[3];
    const auto cache_head_stride = key_cache.stride(1);
    [[maybe_unused]] const auto cache_batch_stride = key_cache.stride(0);
    if (key_cache.stride(3) != 1 || key_cache.stride(2) != width ||
        cache_head_stride < sequence * width) {
        throw std::invalid_argument("cached GQA attention requires a dense sequence prefix");
    }
    if (query.device().is_hip()) {
#if MICROLLM_HAS_HIP
        constexpr std::int64_t kMaximumFusedSequence = 4096;
        if (sequence <= kMaximumFusedSequence) {
            Tensor output({batches, heads, 1, width}, DType::Float32, query.device());
            hip::launch_cached_attention_fused(
                static_cast<const float*>(query.data()),
                key_cache.data(), value_cache.data(), key_cache.dtype(),
                static_cast<float*>(output.data()), batches, heads, sequence,
                cache_batch_stride, cache_head_stride, width, repeats, factor,
                context.native_stream(query.device()));
            return output;
        }
        Tensor scores({batches, heads, 1, sequence}, DType::Float32, query.device());
        hip::launch_cached_attention_scores(
            static_cast<const float*>(query.data()),
            key_cache.data(), key_cache.dtype(),
            static_cast<float*>(scores.data()), batches, heads, kv_heads, sequence,
            cache_batch_stride, cache_head_stride, width, repeats, factor,
            context.native_stream(query.device()));
        const auto probabilities = softmax(scores, -1, context);
        Tensor output({batches, heads, 1, width}, DType::Float32, query.device());
        hip::launch_cached_attention_context(
            static_cast<const float*>(probabilities.data()),
            value_cache.data(), value_cache.dtype(),
            static_cast<float*>(output.data()), batches, heads, kv_heads, sequence,
            cache_batch_stride, cache_head_stride, width, repeats,
            context.native_stream(query.device()));
        return output;
#else
        throw std::runtime_error("microLLM was built without HIP operator support");
#endif
    }

    const auto query_values = query.to_vector();
    const auto key_values = key_cache.to_vector();
    const auto value_values = value_cache.to_vector();
    std::vector<float> output(static_cast<std::size_t>(batches * heads * width));
    std::vector<float> scores(static_cast<std::size_t>(sequence));
    for (std::int64_t batch = 0; batch < batches; ++batch) {
        for (std::int64_t head = 0; head < heads; ++head) {
            const auto kv_head = head / repeats;
            const auto query_base = (batch * heads + head) * width;
            const auto kv_base = (batch * kv_heads + kv_head) * sequence * width;
            float maximum = -std::numeric_limits<float>::infinity();
            for (std::int64_t position = 0; position < sequence; ++position) {
                float dot = 0.0F;
                for (std::int64_t column = 0; column < width; ++column) {
                    dot += query_values[static_cast<std::size_t>(query_base + column)] *
                           key_values[static_cast<std::size_t>(
                               kv_base + position * width + column)];
                }
                scores[static_cast<std::size_t>(position)] = dot * factor;
                maximum = std::max(maximum, dot * factor);
            }
            float denominator = 0.0F;
            for (auto& score : scores) {
                score = std::exp(score - maximum);
                denominator += score;
            }
            for (std::int64_t column = 0; column < width; ++column) {
                float total = 0.0F;
                for (std::int64_t position = 0; position < sequence; ++position) {
                    total += scores[static_cast<std::size_t>(position)] / denominator *
                             value_values[static_cast<std::size_t>(
                                 kv_base + position * width + column)];
                }
                output[static_cast<std::size_t>(query_base + column)] = total;
            }
        }
    }
    return Tensor::from_vector(output, {batches, heads, 1, width});
}

Tensor cached_gqa_attention_positions(
    const Tensor& query, const Tensor& key_cache, const Tensor& value_cache,
    const Tensor& positions, const Tensor& cache_rows, std::int64_t repeats,
    float factor, [[maybe_unused]] const OpContext& context) {
    require_float(query, "query");
    require_forward_float(key_cache, "key_cache");
    require_forward_float(value_cache, "value_cache");
    require_same_device(query, key_cache);
    require_same_device(query, value_cache);
    require_same_device(query, positions);
    require_same_device(query, cache_rows);
    require_same_shape(key_cache, value_cache);
    require_same_dtype(key_cache, value_cache);
    const auto active = query.shape().empty() ? 0 : query.shape()[0];
    if (query.dtype() != DType::Float32 ||
        (key_cache.dtype() != DType::Float32 &&
         key_cache.dtype() != DType::BFloat16) ||
        query.ndim() != 4 || key_cache.ndim() != 4 || active <= 0 ||
        query.shape()[2] != 1 || key_cache.shape()[0] <= 0 ||
        query.shape()[3] != key_cache.shape()[3] ||
        key_cache.shape()[2] <= 0 || repeats <= 0 ||
        query.shape()[1] != key_cache.shape()[1] * repeats ||
        positions.dtype() != DType::Int32 || cache_rows.dtype() != DType::Int32 ||
        positions.shape() != Shape({active}) ||
        cache_rows.shape() != Shape({active}) ||
        !std::isfinite(factor) || factor <= 0.0F ||
        key_cache.strides() != value_cache.strides()) {
        throw std::invalid_argument(
            "positioned cached GQA attention contract is invalid");
    }
    require_contiguous(query, "query");
    require_contiguous(positions, "positions");
    require_contiguous(cache_rows, "cache_rows");
    const auto cache_batches = key_cache.shape()[0];
    const auto heads = query.shape()[1];
    const auto kv_heads = key_cache.shape()[1];
    const auto sequence = key_cache.shape()[2];
    const auto width = query.shape()[3];
    const auto cache_head_stride = key_cache.stride(1);
    [[maybe_unused]] const auto cache_batch_stride = key_cache.stride(0);
    if (key_cache.stride(3) != 1 || key_cache.stride(2) != width ||
        cache_head_stride < sequence * width) {
        throw std::invalid_argument(
            "positioned cached attention requires a dense sequence prefix");
    }
    if (query.device().is_hip()) {
#if MICROLLM_HAS_HIP
        constexpr std::int64_t kMaximumFusedSequence = 4096;
        if (sequence <= kMaximumFusedSequence) {
            Tensor output({active, heads, 1, width}, DType::Float32,
                          query.device());
            hip::launch_cached_attention_fused_positions(
                static_cast<const float*>(query.data()), key_cache.data(),
                value_cache.data(),
                static_cast<const std::int32_t*>(positions.data()),
                static_cast<const std::int32_t*>(cache_rows.data()),
                key_cache.dtype(), static_cast<float*>(output.data()), active,
                cache_batches, heads, sequence, cache_batch_stride,
                cache_head_stride, width, repeats, factor,
                context.native_stream(query.device()));
            return output;
        }
        Tensor scores({active, heads, 1, sequence}, DType::Float32,
                      query.device());
        hip::launch_cached_attention_scores_positions(
            static_cast<const float*>(query.data()), key_cache.data(),
            static_cast<const std::int32_t*>(positions.data()),
            static_cast<const std::int32_t*>(cache_rows.data()),
            key_cache.dtype(), static_cast<float*>(scores.data()), active,
            cache_batches, heads, sequence, cache_batch_stride,
            cache_head_stride, width, repeats, factor,
            context.native_stream(query.device()));
        const auto probabilities = softmax(scores, -1, context);
        Tensor output({active, heads, 1, width}, DType::Float32,
                      query.device());
        hip::launch_cached_attention_context_positions(
            static_cast<const float*>(probabilities.data()), value_cache.data(),
            static_cast<const std::int32_t*>(positions.data()),
            static_cast<const std::int32_t*>(cache_rows.data()),
            value_cache.dtype(), static_cast<float*>(output.data()), active,
            cache_batches, heads, sequence, cache_batch_stride,
            cache_head_stride, width, repeats,
            context.native_stream(query.device()));
        return output;
#else
        throw std::runtime_error("microLLM was built without HIP operator support");
#endif
    }

    const auto query_values = query.to_vector();
    const auto key_values = key_cache.to_vector();
    const auto value_values = value_cache.to_vector();
    const auto position_values = positions.to_int32_vector();
    const auto row_values = cache_rows.to_int32_vector();
    std::vector<float> output(
        static_cast<std::size_t>(active * heads * width));
    std::vector<float> scores(static_cast<std::size_t>(sequence));
    for (std::int64_t batch = 0; batch < active; ++batch) {
        const auto cache_batch = row_values[static_cast<std::size_t>(batch)];
        const auto visible =
            static_cast<std::int64_t>(position_values[static_cast<std::size_t>(batch)]) + 1;
        if (cache_batch < 0 || cache_batch >= cache_batches || visible <= 0 ||
            visible > sequence) {
            throw std::out_of_range(
                "positioned cached attention row or prefix is invalid");
        }
        for (std::int64_t head = 0; head < heads; ++head) {
            const auto kv_head = head / repeats;
            const auto query_base = (batch * heads + head) * width;
            const auto kv_base =
                (cache_batch * kv_heads + kv_head) * sequence * width;
            float maximum = -std::numeric_limits<float>::infinity();
            for (std::int64_t position = 0; position < visible; ++position) {
                float dot = 0.0F;
                for (std::int64_t column = 0; column < width; ++column) {
                    dot += query_values[static_cast<std::size_t>(query_base + column)] *
                           key_values[static_cast<std::size_t>(
                               kv_base + position * width + column)];
                }
                scores[static_cast<std::size_t>(position)] = dot * factor;
                maximum = std::max(maximum, dot * factor);
            }
            float denominator = 0.0F;
            for (std::int64_t position = 0; position < visible; ++position) {
                auto& score = scores[static_cast<std::size_t>(position)];
                score = std::exp(score - maximum);
                denominator += score;
            }
            for (std::int64_t column = 0; column < width; ++column) {
                float total = 0.0F;
                for (std::int64_t position = 0; position < visible; ++position) {
                    total += scores[static_cast<std::size_t>(position)] /
                             denominator *
                             value_values[static_cast<std::size_t>(
                                 kv_base + position * width + column)];
                }
                output[static_cast<std::size_t>(query_base + column)] = total;
            }
        }
    }
    return Tensor::from_vector(output, {active, heads, 1, width});
}

void argmax_out_(const Tensor& input, Tensor& output,
                 [[maybe_unused]] const OpContext& context) {
    require_float(input, "input");
    if (input.numel() <= 0 || input.numel() > std::numeric_limits<std::int32_t>::max()) {
        throw std::invalid_argument("argmax requires 1..INT32_MAX float32 elements");
    }
    require_same_device(input, output);
    if (output.dtype() != DType::Int32 || output.shape() != Shape({1, 1}) ||
        !output.is_contiguous()) {
        throw std::invalid_argument("argmax output must be contiguous Int32 [1,1]");
    }
    if (input.device().is_hip()) {
        require_contiguous(input, "input");
#if MICROLLM_HAS_HIP
        constexpr std::int64_t kTwoStageThreshold = 32768;
        if (input.numel() >= kTwoStageThreshold) {
            const auto blocks = std::min<std::int64_t>(
                256, (input.numel() + 255) / 256);
            Tensor partials({blocks, 3}, DType::Float32, input.device());
            hip::launch_argmax_two_stage(
                static_cast<const float*>(input.data()),
                static_cast<float*>(partials.data()),
                static_cast<std::int32_t*>(output.data()), input.numel(), blocks,
                context.native_stream(input.device()));
        } else {
            hip::launch_argmax(static_cast<const float*>(input.data()),
                               static_cast<std::int32_t*>(output.data()), input.numel(),
                               context.native_stream(input.device()));
        }
        return;
#else
        throw std::runtime_error("microLLM was built without HIP operator support");
#endif
    }
    const auto values = input.to_vector();
    auto best = -std::numeric_limits<float>::infinity();
    std::int32_t index = 0;
    for (std::size_t candidate = 0; candidate < values.size(); ++candidate) {
        if (!std::isfinite(values[candidate])) {
            *static_cast<std::int32_t*>(output.data()) = -1;
            return;
        }
        if (values[candidate] > best) {
            best = values[candidate];
            index = static_cast<std::int32_t>(candidate);
        }
    }
    *static_cast<std::int32_t*>(output.data()) = index;
}

Tensor argmax(const Tensor& input, const OpContext& context) {
    Tensor output({1, 1}, DType::Int32, input.device());
    argmax_out_(input, output, context);
    return output;
}

void argmax_last_dim_out_(const Tensor& input, Tensor& output,
                          [[maybe_unused]] const OpContext& context) {
    require_float(input, "input");
    if (input.ndim() <= 0 || input.shape().back() <= 0 ||
        input.shape().back() > std::numeric_limits<std::int32_t>::max()) {
        throw std::invalid_argument(
            "argmax_last_dim requires a non-empty last dimension within int32");
    }
    const auto classes = input.shape().back();
    const auto rows = input.numel() / classes;
    auto output_shape = input.shape();
    output_shape.pop_back();
    require_same_device(input, output);
    if (output.dtype() != DType::Int32 || output.shape() != output_shape ||
        !output.is_contiguous()) {
        throw std::invalid_argument(
            "argmax_last_dim output shape, dtype, or layout is invalid");
    }
    if (input.device().is_hip()) {
        require_contiguous(input, "input");
#if MICROLLM_HAS_HIP
        hip::launch_argmax_last_dim(
            static_cast<const float*>(input.data()),
            static_cast<std::int32_t*>(output.data()), rows, classes,
            context.native_stream(input.device()));
        return;
#else
        throw std::runtime_error("microLLM was built without HIP operator support");
#endif
    }
    const auto values = input.to_vector();
    std::vector<std::int32_t> selected(static_cast<std::size_t>(rows), 0);
    for (std::int64_t row = 0; row < rows; ++row) {
        auto best = -std::numeric_limits<float>::infinity();
        for (std::int64_t column = 0; column < classes; ++column) {
            const auto value = values[static_cast<std::size_t>(row * classes + column)];
            if (!std::isfinite(value)) {
                selected[static_cast<std::size_t>(row)] = -1;
                break;
            }
            if (value > best) {
                best = value;
                selected[static_cast<std::size_t>(row)] =
                    static_cast<std::int32_t>(column);
            }
        }
    }
    std::copy(selected.begin(), selected.end(),
              static_cast<std::int32_t*>(output.data()));
}

Tensor argmax_last_dim(const Tensor& input, const OpContext& context) {
    auto output_shape = input.shape();
    if (!output_shape.empty()) output_shape.pop_back();
    Tensor output(output_shape, DType::Int32, input.device());
    argmax_last_dim_out_(input, output, context);
    return output;
}

Tensor silu_backward(const Tensor& input, const Tensor& gradient,
                     [[maybe_unused]] const OpContext& context) {
    require_float(input, "input");
    require_float(gradient, "gradient");
    require_same_shape(input, gradient);
    require_same_device(input, gradient);
    if (input.device().is_hip()) {
        require_contiguous(input, "input");
        require_contiguous(gradient, "gradient");
        Tensor input_gradient(input.shape(), DType::Float32, input.device());
#if MICROLLM_HAS_HIP
        hip::launch_silu_backward(static_cast<const float*>(input.data()),
                                  static_cast<const float*>(gradient.data()),
                                  static_cast<float*>(input_gradient.data()), input.numel(),
                                  context.native_stream(input.device()));
        return input_gradient;
#else
        throw std::runtime_error("microLLM was built without HIP operator support");
#endif
    }
    const auto values = input.to_vector();
    auto result = gradient.to_vector();
    for (std::size_t index = 0; index < values.size(); ++index) {
        const auto probability = sigmoid(values[index]);
        result[index] *= probability * (1.0F + values[index] * (1.0F - probability));
    }
    return from_values(std::move(result), input.shape());
}

TensorPair swiglu_backward(const Tensor& gate, const Tensor& up, const Tensor& gradient,
                           [[maybe_unused]] const OpContext& context) {
    require_float(gate, "gate");
    require_float(up, "up");
    require_float(gradient, "gradient");
    require_same_shape(gate, up);
    require_same_shape(gate, gradient);
    require_same_device(gate, up);
    require_same_device(gate, gradient);
    if (gate.device().is_hip()) {
        require_contiguous(gate, "gate");
        require_contiguous(up, "up");
        require_contiguous(gradient, "gradient");
        Tensor gate_gradient(gate.shape(), DType::Float32, gate.device());
        Tensor up_gradient(up.shape(), DType::Float32, up.device());
#if MICROLLM_HAS_HIP
        hip::launch_swiglu_backward(
            static_cast<const float*>(gate.data()), static_cast<const float*>(up.data()),
            static_cast<const float*>(gradient.data()),
            static_cast<float*>(gate_gradient.data()), static_cast<float*>(up_gradient.data()),
            gate.numel(), context.native_stream(gate.device()));
        return {std::move(gate_gradient), std::move(up_gradient)};
#else
        throw std::runtime_error("microLLM was built without HIP operator support");
#endif
    }
    const auto gate_values = gate.to_vector();
    const auto up_values = up.to_vector();
    const auto output_gradient = gradient.to_vector();
    std::vector<float> gate_gradient(output_gradient.size());
    std::vector<float> up_gradient(output_gradient.size());
    for (std::size_t index = 0; index < output_gradient.size(); ++index) {
        const auto probability = sigmoid(gate_values[index]);
        gate_gradient[index] = output_gradient[index] * up_values[index] * probability *
                               (1.0F + gate_values[index] * (1.0F - probability));
        up_gradient[index] = output_gradient[index] * gate_values[index] * probability;
    }
    return {from_values(std::move(gate_gradient), gate.shape()),
            from_values(std::move(up_gradient), up.shape())};
}

Tensor rope_backward(const Tensor& gradient, std::int64_t sequence_dim,
                     std::int64_t position_offset, float base,
                     [[maybe_unused]] const OpContext& context) {
    require_float(gradient, "gradient");
    const auto sequence = positive_dim(gradient, sequence_dim);
    if (gradient.ndim() < 2 || sequence == gradient.ndim() - 1 ||
        gradient.shape().back() % 2 != 0 || position_offset < 0 || base <= 0.0F) {
        throw std::invalid_argument("rope backward shape or configuration is invalid");
    }
    const auto head_width = gradient.shape().back();
    const auto sequence_stride =
        contiguous_strides(gradient.shape())[static_cast<std::size_t>(sequence)];
    if (gradient.device().is_hip()) {
        require_contiguous(gradient, "gradient");
        Tensor input_gradient(gradient.shape(), DType::Float32, gradient.device());
#if MICROLLM_HAS_HIP
        hip::launch_rope_backward(
            static_cast<const float*>(gradient.data()),
            static_cast<float*>(input_gradient.data()), gradient.numel(), head_width,
            gradient.shape()[static_cast<std::size_t>(sequence)], sequence_stride,
            position_offset, base, context.native_stream(gradient.device()));
        return input_gradient;
#else
        throw std::runtime_error("microLLM was built without HIP operator support");
#endif
    }
    const auto values = gradient.to_vector();
    auto result = values;
    for (std::int64_t linear = 0; linear < gradient.numel(); linear += head_width) {
        const auto position =
            (linear / sequence_stride) % gradient.shape()[static_cast<std::size_t>(sequence)];
        for (std::int64_t pair = 0; pair < head_width / 2; ++pair) {
            const auto angle = static_cast<float>(position + position_offset) *
                               std::pow(base, -2.0F * static_cast<float>(pair) /
                                                  static_cast<float>(head_width));
            const auto cosine = std::cos(angle);
            const auto sine = std::sin(angle);
            const auto even = static_cast<std::size_t>(linear + pair * 2);
            const auto odd = even + 1;
            result[even] = values[even] * cosine + values[odd] * sine;
            result[odd] = -values[even] * sine + values[odd] * cosine;
        }
    }
    return from_values(std::move(result), gradient.shape());
}

Tensor rope_split_half_backward(const Tensor& gradient,
                                std::int64_t sequence_dim,
                                std::int64_t position_offset, float base,
                                [[maybe_unused]] const OpContext& context) {
    require_float(gradient, "gradient");
    if (gradient.ndim() < 2) throw std::invalid_argument("rope backward requires rank two or greater");
    const auto sequence = positive_dim(gradient, sequence_dim);
    if (sequence == gradient.ndim() - 1 || gradient.shape().back() % 2 != 0 ||
        position_offset < 0 || base <= 0.0F) {
        throw std::invalid_argument("split-half rope backward configuration is invalid");
    }
    const auto head_width = gradient.shape().back();
    const auto sequence_stride =
        contiguous_strides(gradient.shape())[static_cast<std::size_t>(sequence)];
    if (gradient.device().is_hip()) {
        require_contiguous(gradient, "gradient");
        Tensor output(gradient.shape(), DType::Float32, gradient.device());
#if MICROLLM_HAS_HIP
        hip::launch_rope_split_half_backward(
            static_cast<const float*>(gradient.data()), static_cast<float*>(output.data()),
            gradient.numel(), head_width,
            gradient.shape()[static_cast<std::size_t>(sequence)], sequence_stride,
            position_offset, base, context.native_stream(gradient.device()));
        return output;
#else
        throw std::runtime_error("microLLM was built without HIP operator support");
#endif
    }
    const auto values = gradient.to_vector();
    auto output = values;
    const auto half = head_width / 2;
    for (std::int64_t linear = 0; linear < gradient.numel(); linear += head_width) {
        const auto position =
            (linear / sequence_stride) % gradient.shape()[static_cast<std::size_t>(sequence)];
        for (std::int64_t pair = 0; pair < half; ++pair) {
            const auto angle = static_cast<float>(position + position_offset) *
                               std::pow(base, -2.0F * static_cast<float>(pair) /
                                                  static_cast<float>(head_width));
            const auto cosine = std::cos(angle);
            const auto sine = std::sin(angle);
            const auto first = static_cast<std::size_t>(linear + pair);
            const auto second = static_cast<std::size_t>(linear + pair + half);
            output[first] = values[first] * cosine + values[second] * sine;
            output[second] = -values[first] * sine + values[second] * cosine;
        }
    }
    return from_values(std::move(output), gradient.shape());
}

Tensor cross_entropy_backward(const Tensor& logits, const Tensor& targets,
                              const Tensor& loss_gradient,
                              [[maybe_unused]] const OpContext& context) {
    require_float(logits, "logits");
    require_float(loss_gradient, "loss_gradient");
    if (loss_gradient.numel() != 1 || targets.dtype() != DType::Int32) {
        throw std::invalid_argument("cross entropy backward requires scalar gradient and int32 targets");
    }
    require_same_device(logits, targets);
    require_same_device(logits, loss_gradient);
    if (logits.ndim() < 1 || logits.shape().back() <= 0) {
        throw std::invalid_argument("cross entropy backward logits are invalid");
    }
    Shape expected(logits.shape().begin(), logits.shape().end() - 1);
    if (targets.shape() != expected) throw std::invalid_argument("target shape must match logits prefix");
    const auto classes = logits.shape().back();
    const auto rows = logits.numel() / classes;
    if (logits.device().is_hip()) {
        require_contiguous(logits, "logits");
        require_contiguous(targets, "targets");
        require_contiguous(loss_gradient, "loss_gradient");
        Tensor logits_gradient(logits.shape(), DType::Float32, logits.device());
        Tensor row_stats({rows, 2}, DType::Float32, logits.device());
        Tensor factor(Shape{}, DType::Float32, logits.device());
#if MICROLLM_HAS_HIP
        hip::launch_cross_entropy_backward(
            static_cast<const float*>(logits.data()),
            static_cast<const std::int32_t*>(targets.data()),
            static_cast<const float*>(loss_gradient.data()),
            static_cast<float*>(logits_gradient.data()),
            static_cast<float*>(row_stats.data()), static_cast<float*>(factor.data()),
            rows, classes,
            context.native_stream(logits.device()));
        return logits_gradient;
#else
        throw std::runtime_error("microLLM was built without HIP operator support");
#endif
    }
    const auto probabilities = softmax(logits).to_vector();
    const auto labels = targets.to_int32_vector();
    auto result = probabilities;
    const auto valid_rows = static_cast<std::int64_t>(std::count_if(
        labels.begin(), labels.end(), [](std::int32_t label) { return label != -100; }));
    if (valid_rows == 0) throw std::invalid_argument("cross_entropy has no non-ignored targets");
    for (std::int64_t row = 0; row < rows; ++row) {
        const auto label = labels[static_cast<std::size_t>(row)];
        if (label == -100) {
            std::fill_n(result.begin() + row * classes, classes, 0.0F);
        } else {
            if (label < 0 || label >= classes) throw std::out_of_range("target class out of range");
            result[static_cast<std::size_t>(row * classes + label)] -= 1.0F;
        }
    }
    const auto factor = loss_gradient.to_vector()[0] / static_cast<float>(valid_rows);
    for (auto& value : result) value *= factor;
    return from_values(std::move(result), logits.shape());
}

Tensor causal_softmax(const Tensor& scores, [[maybe_unused]] const OpContext& context) {
    require_float(scores, "scores");
    if (scores.ndim() < 2 || scores.shape()[scores.shape().size() - 2] != scores.shape().back() ||
        scores.shape().back() == 0) {
        throw std::invalid_argument("causal_softmax requires square non-empty final dimensions");
    }
    const auto sequence = scores.shape().back();
    const auto rows = scores.numel() / sequence;
    if (scores.device().is_hip()) {
        require_contiguous(scores, "scores");
        Tensor output(scores.shape(), DType::Float32, scores.device());
#if MICROLLM_HAS_HIP
        hip::launch_causal_softmax(static_cast<const float*>(scores.data()),
                                   static_cast<float*>(output.data()), rows, sequence,
                                   context.native_stream(scores.device()));
        return output;
#else
        throw std::runtime_error("microLLM was built without HIP operator support");
#endif
    }
    const auto values = scores.to_vector();
    auto result = values;
    for (std::int64_t row = 0; row < rows; ++row) {
        const auto visible = row % sequence;
        const auto base = row * sequence;
        float maximum = -std::numeric_limits<float>::infinity();
        for (std::int64_t column = 0; column <= visible; ++column) {
            maximum = std::max(maximum, values[static_cast<std::size_t>(base + column)]);
        }
        float denominator = 0.0F;
        for (std::int64_t column = 0; column <= visible; ++column) {
            const auto index = static_cast<std::size_t>(base + column);
            result[index] = std::exp(values[index] - maximum);
            denominator += result[index];
        }
        for (std::int64_t column = 0; column <= visible; ++column) {
            result[static_cast<std::size_t>(base + column)] /= denominator;
        }
        for (std::int64_t column = visible + 1; column < sequence; ++column) {
            result[static_cast<std::size_t>(base + column)] = 0.0F;
        }
    }
    return from_values(std::move(result), scores.shape());
}

Tensor causal_softmax_backward(const Tensor& output, const Tensor& gradient,
                               [[maybe_unused]] const OpContext& context) {
    require_float(output, "output");
    require_float(gradient, "gradient");
    require_same_shape(output, gradient);
    require_same_device(output, gradient);
    if (output.ndim() < 2 || output.shape()[output.shape().size() - 2] != output.shape().back() ||
        output.shape().back() == 0) {
        throw std::invalid_argument("causal softmax backward shape is invalid");
    }
    const auto sequence = output.shape().back();
    const auto rows = output.numel() / sequence;
    if (output.device().is_hip()) {
        require_contiguous(output, "output");
        require_contiguous(gradient, "gradient");
        Tensor input_gradient(output.shape(), DType::Float32, output.device());
#if MICROLLM_HAS_HIP
        hip::launch_causal_softmax_backward(
            static_cast<const float*>(output.data()),
            static_cast<const float*>(gradient.data()),
            static_cast<float*>(input_gradient.data()), rows, sequence,
            context.native_stream(output.device()));
        return input_gradient;
#else
        throw std::runtime_error("microLLM was built without HIP operator support");
#endif
    }
    const auto probabilities = output.to_vector();
    const auto output_gradient = gradient.to_vector();
    std::vector<float> input_gradient(output_gradient.size(), 0.0F);
    for (std::int64_t row = 0; row < rows; ++row) {
        const auto visible = row % sequence;
        const auto base = row * sequence;
        float dot = 0.0F;
        for (std::int64_t column = 0; column <= visible; ++column) {
            const auto index = static_cast<std::size_t>(base + column);
            dot += output_gradient[index] * probabilities[index];
        }
        for (std::int64_t column = 0; column <= visible; ++column) {
            const auto index = static_cast<std::size_t>(base + column);
            input_gradient[index] = probabilities[index] * (output_gradient[index] - dot);
        }
    }
    return from_values(std::move(input_gradient), output.shape());
}

namespace {

void validate_causal_gqa(const Tensor& query, const Tensor& key,
                         const Tensor& value, std::int64_t repeats,
                         float scale) {
    require_float(query, "query");
    require_float(key, "key");
    require_float(value, "value");
    require_same_device(query, key);
    require_same_device(query, value);
    if (query.ndim() != 4 || key.ndim() != 4 || value.shape() != key.shape() ||
        query.shape()[0] != key.shape()[0] ||
        query.shape()[2] != key.shape()[2] ||
        query.shape()[3] != key.shape()[3] ||
        query.shape()[1] != key.shape()[1] * repeats || repeats <= 0 ||
        query.shape()[2] <= 0 || query.shape()[3] <= 0 ||
        !std::isfinite(scale) || !(scale > 0.0F) ||
        !query.is_contiguous() || !key.is_contiguous() || !value.is_contiguous()) {
        throw std::invalid_argument(
            "causal GQA requires contiguous Q[B,H,T,D], K/V[B,KV,T,D] contracts");
    }
}

TensorPair causal_gqa_attention_saved_composed(
    const Tensor& query, const Tensor& key, const Tensor& value,
    std::int64_t repeats, float scale, const OpContext& context) {
    const auto expanded_key = repeats == 1 ? key : repeat_interleave(key, 1, repeats, context);
    const auto expanded_value = repeats == 1 ? value : repeat_interleave(value, 1, repeats, context);
    const auto scores = ops::scale(
        matmul(query, expanded_key.transpose(-2, -1).contiguous(), context),
        scale, context);
    auto probabilities = causal_softmax(scores, context);
    auto output = matmul(probabilities, expanded_value, context);
    return {std::move(output), std::move(probabilities)};
}

Tensor causal_gqa_attention_composed(const Tensor& query, const Tensor& key,
                                     const Tensor& value, std::int64_t repeats,
                                     float scale, const OpContext& context) {
    return causal_gqa_attention_saved_composed(
        query, key, value, repeats, scale, context).first;
}

TensorTriple causal_gqa_attention_backward_composed(
    const Tensor& query, const Tensor& key, const Tensor& value,
    const Tensor& output_gradient, std::int64_t repeats, float scale,
    const OpContext& context) {
    const auto expanded_key = repeats == 1 ? key : repeat_interleave(key, 1, repeats, context);
    const auto expanded_value = repeats == 1 ? value : repeat_interleave(value, 1, repeats, context);
    const auto scores = ops::scale(
        matmul(query, expanded_key.transpose(-2, -1).contiguous(), context),
        scale, context);
    const auto probabilities = causal_softmax(scores, context);
    const auto probability_gradient = matmul(
        output_gradient, expanded_value.transpose(-2, -1).contiguous(), context);
    const auto score_gradient = ops::scale(
        causal_softmax_backward(probabilities, probability_gradient, context),
        scale, context);
    auto query_gradient = matmul(score_gradient, expanded_key, context);
    auto key_gradient = matmul(
        score_gradient.transpose(-2, -1).contiguous(), query, context);
    auto value_gradient = matmul(
        probabilities.transpose(-2, -1).contiguous(), output_gradient, context);
    if (repeats != 1) {
        key_gradient = repeat_interleave_backward(
            key_gradient, key.shape(), 1, repeats, context);
        value_gradient = repeat_interleave_backward(
            value_gradient, value.shape(), 1, repeats, context);
    }
    return {std::move(query_gradient), std::move(key_gradient),
            std::move(value_gradient)};
}

}  // namespace

Tensor causal_gqa_attention(const Tensor& query, const Tensor& key,
                            const Tensor& value, std::int64_t repeats,
                            float scale, const OpContext& context) {
    validate_causal_gqa(query, key, value, repeats, scale);
    const auto batches = query.shape()[0];
    const auto heads = query.shape()[1];
    const auto kv_heads = key.shape()[1];
    const auto sequence = query.shape()[2];
    const auto width = query.shape()[3];
    if (query.device().is_hip()) {
        if (sequence > 4096 || width > 256) {
            return causal_gqa_attention_composed(
                query, key, value, repeats, scale, context);
        }
        if (sequence >= 256 && hipblaslt_available()) {
            return causal_gqa_attention_saved(
                query, key, value, repeats, scale, context).first;
        }
        Tensor output(query.shape(), DType::Float32, query.device());
#if MICROLLM_HAS_HIP
        hip::launch_causal_gqa_attention(
            static_cast<const float*>(query.data()),
            static_cast<const float*>(key.data()),
            static_cast<const float*>(value.data()),
            static_cast<float*>(output.data()), nullptr, batches, heads, kv_heads,
            sequence, width, repeats, scale,
            context.native_stream(query.device()));
        return output;
#else
        throw std::runtime_error("microLLM was built without HIP operator support");
#endif
    }
    const auto queries = query.to_vector();
    const auto keys = key.to_vector();
    const auto values = value.to_vector();
    std::vector<float> output(queries.size(), 0.0F);
    std::vector<float> probabilities(static_cast<std::size_t>(sequence));
    for (std::int64_t batch = 0; batch < batches; ++batch) {
        for (std::int64_t head = 0; head < heads; ++head) {
            const auto kv_head = head / repeats;
            const auto kv_base = (batch * kv_heads + kv_head) * sequence * width;
            for (std::int64_t position = 0; position < sequence; ++position) {
                const auto query_base =
                    ((batch * heads + head) * sequence + position) * width;
                auto maximum = -std::numeric_limits<float>::infinity();
                for (std::int64_t source = 0; source <= position; ++source) {
                    float dot = 0.0F;
                    for (std::int64_t column = 0; column < width; ++column) {
                        dot += queries[static_cast<std::size_t>(query_base + column)] *
                               keys[static_cast<std::size_t>(
                                   kv_base + source * width + column)];
                    }
                    probabilities[static_cast<std::size_t>(source)] = dot * scale;
                    maximum = std::max(maximum, dot * scale);
                }
                float denominator = 0.0F;
                for (std::int64_t source = 0; source <= position; ++source) {
                    auto& probability = probabilities[static_cast<std::size_t>(source)];
                    probability = std::exp(probability - maximum);
                    denominator += probability;
                }
                for (std::int64_t column = 0; column < width; ++column) {
                    float total = 0.0F;
                    for (std::int64_t source = 0; source <= position; ++source) {
                        total += probabilities[static_cast<std::size_t>(source)] /
                                 denominator * values[static_cast<std::size_t>(
                                     kv_base + source * width + column)];
                    }
                    output[static_cast<std::size_t>(query_base + column)] = total;
                }
            }
        }
    }
    return from_values(std::move(output), query.shape());
}

TensorPair causal_gqa_attention_saved(
    const Tensor& query, const Tensor& key, const Tensor& value,
    std::int64_t repeats, float scale, const OpContext& context) {
    validate_causal_gqa(query, key, value, repeats, scale);
    const auto batches = query.shape()[0];
    const auto heads = query.shape()[1];
    [[maybe_unused]] const auto kv_heads = key.shape()[1];
    const auto sequence = query.shape()[2];
    [[maybe_unused]] const auto width = query.shape()[3];
    if (!query.device().is_hip() || sequence > 4096 || width > 256) {
        return causal_gqa_attention_saved_composed(
            query, key, value, repeats, scale, context);
    }
    if (sequence >= 256 && hipblaslt_available()) {
        // Long training sequences amortize the library setup and avoid the
        // scalar fused kernel's repeated dot products.  Keep the readable
        // fused path below for short sequences and library-free builds.
        const auto expanded_key = repeats == 1
                                      ? key : repeat_interleave(key, 1, repeats, context);
        const auto expanded_value = repeats == 1
                                        ? value : repeat_interleave(value, 1, repeats, context);
        const auto scaled_query = ops::scale(query, scale, context);
        auto probabilities = matmul_with_implementation(
            scaled_query, expanded_key, MatmulImplementation::HipBLASLt,
            false, true, context);
#if MICROLLM_HAS_HIP
        // The QK scores are dead after softmax. The cooperative kernels finish
        // every score read before normalization writes, so input/output aliasing
        // removes one [B,H,T,T] allocation without changing reduction order.
        hip::launch_causal_softmax(
            static_cast<const float*>(probabilities.data()),
            static_cast<float*>(probabilities.data()),
            probabilities.numel() / sequence, sequence,
            context.native_stream(query.device()));
#else
        throw std::runtime_error("microLLM was built without HIP operator support");
#endif
        auto output = matmul_with_implementation(
            probabilities, expanded_value, MatmulImplementation::HipBLASLt,
            context);
        return {std::move(output), std::move(probabilities)};
    }
    Tensor output(query.shape(), DType::Float32, query.device());
    Tensor probabilities({batches, heads, sequence, sequence},
                         DType::Float32, query.device());
    fill_(probabilities, 0.0F, context);
#if MICROLLM_HAS_HIP
    hip::launch_causal_gqa_attention(
        static_cast<const float*>(query.data()),
        static_cast<const float*>(key.data()),
        static_cast<const float*>(value.data()),
        static_cast<float*>(output.data()),
        static_cast<float*>(probabilities.data()), batches, heads, kv_heads,
        sequence, width, repeats, scale,
        context.native_stream(query.device()));
    return {std::move(output), std::move(probabilities)};
#else
    throw std::runtime_error("microLLM was built without HIP operator support");
#endif
}

TensorTriple causal_gqa_attention_backward(
    const Tensor& query, const Tensor& key, const Tensor& value,
    const Tensor& output_gradient, std::int64_t repeats, float scale,
    const OpContext& context) {
    validate_causal_gqa(query, key, value, repeats, scale);
    require_float(output_gradient, "output_gradient");
    require_same_shape(query, output_gradient);
    require_same_device(query, output_gradient);
    require_contiguous(output_gradient, "output_gradient");
    const auto batches = query.shape()[0];
    const auto heads = query.shape()[1];
    const auto kv_heads = key.shape()[1];
    const auto sequence = query.shape()[2];
    const auto width = query.shape()[3];
    if (query.device().is_hip()) {
        if (sequence > 4096 || width > 256) {
            return causal_gqa_attention_backward_composed(
                query, key, value, output_gradient, repeats, scale, context);
        }
        Tensor query_gradient(query.shape(), DType::Float32, query.device());
#if MICROLLM_HAS_HIP
        if (sequence >= 256 && hipblaslt_available()) {
            Tensor probabilities({batches, heads, sequence, sequence},
                                 DType::Float32, query.device());
            Tensor scaled_score_gradients({batches, heads, sequence, sequence},
                                          DType::Float32, query.device());
            fill_(probabilities, 0.0F, context);
            fill_(scaled_score_gradients, 0.0F, context);
            hip::launch_causal_gqa_attention_backward_rows(
                static_cast<const float*>(query.data()),
                static_cast<const float*>(key.data()),
                static_cast<const float*>(value.data()),
                static_cast<const float*>(output_gradient.data()),
                static_cast<float*>(query_gradient.data()),
                static_cast<float*>(probabilities.data()),
                static_cast<float*>(scaled_score_gradients.data()),
                batches, heads, kv_heads, sequence, width, repeats, scale,
                context.native_stream(query.device()));
            auto expanded_key_gradient = matmul_with_implementation(
                scaled_score_gradients, query, MatmulImplementation::HipBLASLt,
                true, false, context);
            auto expanded_value_gradient = matmul_with_implementation(
                probabilities, output_gradient, MatmulImplementation::HipBLASLt,
                true, false, context);
            auto key_gradient = repeats == 1
                                    ? std::move(expanded_key_gradient)
                                    : repeat_interleave_backward(
                                          expanded_key_gradient, key.shape(), 1,
                                          repeats, context);
            auto value_gradient = repeats == 1
                                      ? std::move(expanded_value_gradient)
                                      : repeat_interleave_backward(
                                            expanded_value_gradient, value.shape(), 1,
                                            repeats, context);
            return {std::move(query_gradient), std::move(key_gradient),
                    std::move(value_gradient)};
        }
        Tensor key_gradient(key.shape(), DType::Float32, key.device());
        Tensor value_gradient(value.shape(), DType::Float32, value.device());
        fill_(key_gradient, 0.0F, context);
        fill_(value_gradient, 0.0F, context);
        hip::launch_causal_gqa_attention_backward(
            static_cast<const float*>(query.data()),
            static_cast<const float*>(key.data()),
            static_cast<const float*>(value.data()),
            static_cast<const float*>(output_gradient.data()),
            static_cast<float*>(query_gradient.data()),
            static_cast<float*>(key_gradient.data()),
            static_cast<float*>(value_gradient.data()), batches, heads, kv_heads,
            sequence, width, repeats, scale,
            context.native_stream(query.device()));
        return {std::move(query_gradient), std::move(key_gradient),
                std::move(value_gradient)};
#else
        throw std::runtime_error("microLLM was built without HIP operator support");
#endif
    }
    const auto queries = query.to_vector();
    const auto keys = key.to_vector();
    const auto values = value.to_vector();
    const auto gradients = output_gradient.to_vector();
    std::vector<float> query_gradient(queries.size(), 0.0F);
    std::vector<float> key_gradient(keys.size(), 0.0F);
    std::vector<float> value_gradient(values.size(), 0.0F);
    std::vector<float> probabilities(static_cast<std::size_t>(sequence));
    std::vector<float> probability_gradients(static_cast<std::size_t>(sequence));
    for (std::int64_t batch = 0; batch < batches; ++batch) {
        for (std::int64_t head = 0; head < heads; ++head) {
            const auto kv_head = head / repeats;
            const auto kv_base = (batch * kv_heads + kv_head) * sequence * width;
            for (std::int64_t position = 0; position < sequence; ++position) {
                const auto query_base =
                    ((batch * heads + head) * sequence + position) * width;
                auto maximum = -std::numeric_limits<float>::infinity();
                for (std::int64_t source = 0; source <= position; ++source) {
                    float score = 0.0F;
                    float probability_gradient = 0.0F;
                    for (std::int64_t column = 0; column < width; ++column) {
                        score += queries[static_cast<std::size_t>(query_base + column)] *
                                 keys[static_cast<std::size_t>(
                                     kv_base + source * width + column)];
                        probability_gradient +=
                            gradients[static_cast<std::size_t>(query_base + column)] *
                            values[static_cast<std::size_t>(
                                kv_base + source * width + column)];
                    }
                    probabilities[static_cast<std::size_t>(source)] = score * scale;
                    probability_gradients[static_cast<std::size_t>(source)] =
                        probability_gradient;
                    maximum = std::max(maximum, score * scale);
                }
                float denominator = 0.0F;
                for (std::int64_t source = 0; source <= position; ++source) {
                    auto& probability = probabilities[static_cast<std::size_t>(source)];
                    probability = std::exp(probability - maximum);
                    denominator += probability;
                }
                float weighted_gradient = 0.0F;
                for (std::int64_t source = 0; source <= position; ++source) {
                    auto& probability = probabilities[static_cast<std::size_t>(source)];
                    probability /= denominator;
                    weighted_gradient += probability *
                        probability_gradients[static_cast<std::size_t>(source)];
                }
                for (std::int64_t source = 0; source <= position; ++source) {
                    const auto probability = probabilities[static_cast<std::size_t>(source)];
                    const auto score_gradient = probability *
                        (probability_gradients[static_cast<std::size_t>(source)] -
                         weighted_gradient);
                    for (std::int64_t column = 0; column < width; ++column) {
                        const auto query_index = static_cast<std::size_t>(query_base + column);
                        const auto kv_index = static_cast<std::size_t>(
                            kv_base + source * width + column);
                        query_gradient[query_index] +=
                            score_gradient * scale * keys[kv_index];
                        key_gradient[kv_index] +=
                            score_gradient * scale * queries[query_index];
                        value_gradient[kv_index] +=
                            probability * gradients[query_index];
                    }
                }
            }
        }
    }
    return {from_values(std::move(query_gradient), query.shape()),
            from_values(std::move(key_gradient), key.shape()),
            from_values(std::move(value_gradient), value.shape())};
}

TensorTriple causal_gqa_attention_backward_saved(
    const Tensor& query, const Tensor& key, const Tensor& value,
    const Tensor& probabilities, const Tensor& output_gradient,
    std::int64_t repeats, float scale, const OpContext& context) {
    validate_causal_gqa(query, key, value, repeats, scale);
    require_float(probabilities, "probabilities");
    require_float(output_gradient, "output_gradient");
    require_same_shape(query, output_gradient);
    require_same_device(query, output_gradient);
    require_same_device(query, probabilities);
    require_contiguous(output_gradient, "output_gradient");
    require_contiguous(probabilities, "probabilities");
    const auto batches = query.shape()[0];
    const auto heads = query.shape()[1];
    [[maybe_unused]] const auto kv_heads = key.shape()[1];
    const auto sequence = query.shape()[2];
    [[maybe_unused]] const auto width = query.shape()[3];
    if (probabilities.shape() != Shape({batches, heads, sequence, sequence})) {
        throw std::invalid_argument("saved causal GQA probabilities have the wrong shape");
    }
    if (!query.device().is_hip() || !hipblaslt_available()) {
        return causal_gqa_attention_backward(
            query, key, value, output_gradient, repeats, scale, context);
    }
    if (sequence >= 256) {
        const auto expanded_key = repeats == 1
                                      ? key : repeat_interleave(key, 1, repeats, context);
        const auto expanded_value = repeats == 1
                                        ? value : repeat_interleave(value, 1, repeats, context);
        const auto probability_gradient = matmul_with_implementation(
            output_gradient, expanded_value, MatmulImplementation::HipBLASLt,
            false, true, context);
        auto scaled_score_gradients = ops::scale(
            causal_softmax_backward(probabilities, probability_gradient, context),
            scale, context);
        auto query_gradient = matmul_with_implementation(
            scaled_score_gradients, expanded_key,
            MatmulImplementation::HipBLASLt, context);
        auto expanded_key_gradient = matmul_with_implementation(
            scaled_score_gradients, query, MatmulImplementation::HipBLASLt,
            true, false, context);
        auto expanded_value_gradient = matmul_with_implementation(
            probabilities, output_gradient, MatmulImplementation::HipBLASLt,
            true, false, context);
        auto key_gradient = repeats == 1
                                ? std::move(expanded_key_gradient)
                                : repeat_interleave_backward(
                                      expanded_key_gradient, key.shape(), 1,
                                      repeats, context);
        auto value_gradient = repeats == 1
                                  ? std::move(expanded_value_gradient)
                                  : repeat_interleave_backward(
                                        expanded_value_gradient, value.shape(), 1,
                                        repeats, context);
        return {std::move(query_gradient), std::move(key_gradient),
                std::move(value_gradient)};
    }
    Tensor query_gradient(query.shape(), DType::Float32, query.device());
    Tensor scaled_score_gradients(probabilities.shape(), DType::Float32, query.device());
    fill_(scaled_score_gradients, 0.0F, context);
#if MICROLLM_HAS_HIP
    hip::launch_causal_gqa_attention_backward_saved_rows(
        static_cast<const float*>(key.data()),
        static_cast<const float*>(value.data()),
        static_cast<const float*>(output_gradient.data()),
        static_cast<const float*>(probabilities.data()),
        static_cast<float*>(query_gradient.data()),
        static_cast<float*>(scaled_score_gradients.data()),
        batches, heads, kv_heads, sequence, width, repeats, scale,
        context.native_stream(query.device()));
    auto expanded_key_gradient = matmul_with_implementation(
        scaled_score_gradients, query, MatmulImplementation::HipBLASLt,
        true, false, context);
    auto expanded_value_gradient = matmul_with_implementation(
        probabilities, output_gradient, MatmulImplementation::HipBLASLt,
        true, false, context);
    auto key_gradient = repeats == 1
                            ? std::move(expanded_key_gradient)
                            : repeat_interleave_backward(
                                  expanded_key_gradient, key.shape(), 1,
                                  repeats, context);
    auto value_gradient = repeats == 1
                              ? std::move(expanded_value_gradient)
                              : repeat_interleave_backward(
                                    expanded_value_gradient, value.shape(), 1,
                                    repeats, context);
    return {std::move(query_gradient), std::move(key_gradient),
            std::move(value_gradient)};
#else
    throw std::runtime_error("microLLM was built without HIP operator support");
#endif
}

Tensor repeat_interleave(const Tensor& input, std::int64_t dim, std::int64_t repeats,
                         [[maybe_unused]] const OpContext& context) {
    require_float(input, "input");
    dim = positive_dim(input, dim);
    if (repeats <= 0) throw std::invalid_argument("repeat_interleave repeats must be positive");
    auto output_shape = input.shape();
    if (output_shape[static_cast<std::size_t>(dim)] >
        std::numeric_limits<std::int64_t>::max() / repeats) {
        throw std::overflow_error("repeat_interleave shape overflows int64");
    }
    const auto input_width = output_shape[static_cast<std::size_t>(dim)];
    output_shape[static_cast<std::size_t>(dim)] *= repeats;
    std::int64_t inner = 1;
    for (std::size_t axis = static_cast<std::size_t>(dim) + 1; axis < output_shape.size(); ++axis) {
        inner *= output_shape[axis];
    }
    Tensor output(output_shape, DType::Float32, input.device());
    if (input.device().is_hip()) {
        require_contiguous(input, "input");
#if MICROLLM_HAS_HIP
        hip::launch_repeat_interleave(
            static_cast<const float*>(input.data()), static_cast<float*>(output.data()),
            output.numel(), input_width * repeats, inner, repeats,
            context.native_stream(input.device()));
        return output;
#else
        throw std::runtime_error("microLLM was built without HIP operator support");
#endif
    }
    const auto values = input.to_vector();
    std::vector<float> result(static_cast<std::size_t>(output.numel()));
    const auto repeated_width = input_width * repeats;
    for (std::int64_t index = 0; index < output.numel(); ++index) {
        const auto inner_index = index % inner;
        const auto repeated_coordinate = (index / inner) % repeated_width;
        const auto outer = index / (inner * repeated_width);
        const auto input_index = outer * input_width * inner +
                                 (repeated_coordinate / repeats) * inner + inner_index;
        result[static_cast<std::size_t>(index)] = values[static_cast<std::size_t>(input_index)];
    }
    return from_values(std::move(result), std::move(output_shape));
}

Tensor repeat_interleave_backward(const Tensor& gradient, const Shape& input_shape,
                                  std::int64_t dim, std::int64_t repeats,
                                  [[maybe_unused]] const OpContext& context) {
    require_float(gradient, "gradient");
    if (input_shape.empty()) throw std::invalid_argument("repeat backward input rank is empty");
    if (dim < 0) dim += static_cast<std::int64_t>(input_shape.size());
    if (dim < 0 || dim >= static_cast<std::int64_t>(input_shape.size()) || repeats <= 0) {
        throw std::invalid_argument("repeat backward dimension or repeats is invalid");
    }
    auto expected = input_shape;
    expected[static_cast<std::size_t>(dim)] *= repeats;
    if (gradient.shape() != expected) throw std::invalid_argument("repeat backward shape mismatch");
    const auto input_width = input_shape[static_cast<std::size_t>(dim)];
    std::int64_t inner = 1;
    for (std::size_t axis = static_cast<std::size_t>(dim) + 1; axis < input_shape.size(); ++axis) {
        inner *= input_shape[axis];
    }
    Tensor input_gradient(input_shape, DType::Float32, gradient.device());
    if (gradient.device().is_hip()) {
        require_contiguous(gradient, "gradient");
#if MICROLLM_HAS_HIP
        hip::launch_repeat_interleave_backward(
            static_cast<const float*>(gradient.data()),
            static_cast<float*>(input_gradient.data()), input_gradient.numel(), input_width,
            inner, repeats, context.native_stream(gradient.device()));
        return input_gradient;
#else
        throw std::runtime_error("microLLM was built without HIP operator support");
#endif
    }
    const auto values = gradient.to_vector();
    std::vector<float> result(static_cast<std::size_t>(input_gradient.numel()));
    const auto repeated_width = input_width * repeats;
    for (std::int64_t index = 0; index < input_gradient.numel(); ++index) {
        const auto inner_index = index % inner;
        const auto input_coordinate = (index / inner) % input_width;
        const auto outer = index / (inner * input_width);
        float total = 0.0F;
        for (std::int64_t repeat = 0; repeat < repeats; ++repeat) {
            const auto output_index = outer * repeated_width * inner +
                                      (input_coordinate * repeats + repeat) * inner + inner_index;
            total += values[static_cast<std::size_t>(output_index)];
        }
        result[static_cast<std::size_t>(index)] = total;
    }
    return from_values(std::move(result), input_shape);
}

}  // namespace microllm::ops
