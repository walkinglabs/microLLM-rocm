#include <microllm/ops/ops.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

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

}  // namespace

void adamw_update_(Tensor& parameter, const Tensor& gradient,
                   Tensor& first_moment, Tensor& second_moment,
                   float learning_rate, float beta1, float beta2,
                   float epsilon, float weight_decay,
                   float first_correction, float second_correction,
                   [[maybe_unused]] const OpContext& context) {
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
        hip::launch_adamw_update(
            static_cast<float*>(parameter.data()),
            static_cast<const float*>(gradient.data()),
            static_cast<float*>(first_moment.data()),
            static_cast<float*>(second_moment.data()), parameter.numel(), learning_rate,
            beta1, beta2, epsilon, weight_decay, first_correction,
            second_correction, context.native_stream(parameter.device()));
        return;
#else
        throw std::runtime_error("microLLM was built without HIP operator support");
#endif
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

ScaledTensor quantize_fp8(const Tensor& input, DType fp8_dtype, float scale,
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
    auto scale_tensor = Tensor::from_vector({scale}, {}, DType::Float32);
    if (input.device().is_hip()) scale_tensor = scale_tensor.to(input.device());
    return {std::move(output), std::move(scale_tensor), scale};
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
    if (!std::isfinite(input.scale_value) || input.scale_value <= 0.0F) {
        throw std::invalid_argument("FP8 dequantize scale must be finite and positive");
    }
    Tensor output(input.values.shape(), output_dtype, input.values.device());
    if (input.values.device().is_cpu()) {
        auto values = input.values.to_vector();
        for (auto& value : values) value *= input.scale_value;
        return Tensor::from_vector(values, input.values.shape(), output_dtype);
    }
#if MICROLLM_HAS_HIP
    hip::launch_dequantize_fp8(input.values.data(), input.values.dtype(), output.data(),
                               output_dtype, input.values.numel(), input.scale_value,
                               context.native_stream(input.values.device()));
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
#if MICROLLM_HAS_HIP
        hip::launch_cross_entropy(static_cast<const float*>(logits.data()),
                                  static_cast<const std::int32_t*>(targets.data()),
                                  static_cast<float*>(output.data()), rows, classes,
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
        fill_(weight_gradient, 0.0F, context);
#if MICROLLM_HAS_HIP
        hip::launch_rms_norm_backward(
            static_cast<const float*>(input.data()),
            static_cast<const float*>(weight.data()),
            static_cast<const float*>(gradient.data()),
            static_cast<float*>(input_gradient.data()),
            static_cast<float*>(weight_gradient.data()), rows, width, epsilon,
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
#if MICROLLM_HAS_HIP
        hip::launch_cross_entropy_backward(
            static_cast<const float*>(logits.data()),
            static_cast<const std::int32_t*>(targets.data()),
            static_cast<const float*>(loss_gradient.data()),
            static_cast<float*>(logits_gradient.data()), rows, classes,
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
