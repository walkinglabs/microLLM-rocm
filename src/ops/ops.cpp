#include <microllm/ops/ops.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace microllm::ops {
namespace {

void require_cpu_float(const Tensor& tensor, const char* name) {
    if (!tensor.defined()) throw std::invalid_argument(std::string(name) + " is undefined");
    if (!tensor.device().is_cpu()) {
        throw std::runtime_error(std::string(name) + " requires CPU until the HIP op milestone");
    }
    if (tensor.dtype() != DType::Float32) {
        throw std::invalid_argument(std::string(name) + " must be float32");
    }
}

void require_same_shape(const Tensor& left, const Tensor& right) {
    if (left.shape() != right.shape()) throw std::invalid_argument("tensor shapes must match");
}

Tensor from_values(std::vector<float> values, Shape shape) {
    return Tensor::from_vector(values, std::move(shape));
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

void fill_(Tensor& tensor, float value) {
    require_cpu_float(tensor, "tensor");
    tensor.fill(value);
}

Tensor add(const Tensor& left, const Tensor& right) {
    require_cpu_float(left, "left");
    require_cpu_float(right, "right");
    require_same_shape(left, right);
    auto left_values = left.to_vector();
    const auto right_values = right.to_vector();
    for (std::size_t index = 0; index < left_values.size(); ++index) {
        left_values[index] += right_values[index];
    }
    return from_values(std::move(left_values), left.shape());
}

Tensor multiply(const Tensor& left, const Tensor& right) {
    require_cpu_float(left, "left");
    require_cpu_float(right, "right");
    require_same_shape(left, right);
    auto left_values = left.to_vector();
    const auto right_values = right.to_vector();
    for (std::size_t index = 0; index < left_values.size(); ++index) {
        left_values[index] *= right_values[index];
    }
    return from_values(std::move(left_values), left.shape());
}

Tensor scale(const Tensor& input, float factor) {
    require_cpu_float(input, "input");
    auto values = input.to_vector();
    for (auto& value : values) value *= factor;
    return from_values(std::move(values), input.shape());
}

Tensor matmul(const Tensor& left, const Tensor& right) {
    require_cpu_float(left, "left");
    require_cpu_float(right, "right");
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
    return from_values(std::move(output), std::move(output_shape));
}

Tensor embedding(const Tensor& weight, const Tensor& indices) {
    require_cpu_float(weight, "weight");
    if (weight.ndim() != 2) throw std::invalid_argument("embedding weight must have rank two");
    if (!indices.device().is_cpu() || indices.dtype() != DType::Int32) {
        throw std::invalid_argument("embedding indices must be a CPU int32 tensor");
    }
    const auto vocabulary = weight.shape()[0];
    const auto width = weight.shape()[1];
    const auto weight_values = weight.to_vector();
    const auto index_values = indices.to_int32_vector();
    std::vector<float> output(index_values.size() * static_cast<std::size_t>(width));
    for (std::size_t token = 0; token < index_values.size(); ++token) {
        const auto index = index_values[token];
        if (index < 0 || index >= vocabulary) throw std::out_of_range("embedding index out of range");
        std::copy_n(weight_values.begin() + static_cast<std::ptrdiff_t>(index * width), width,
                    output.begin() + static_cast<std::ptrdiff_t>(token * static_cast<std::size_t>(width)));
    }
    auto output_shape = indices.shape();
    output_shape.push_back(width);
    return from_values(std::move(output), std::move(output_shape));
}

Tensor softmax(const Tensor& input, std::int64_t dim) {
    require_cpu_float(input, "input");
    const auto normalized = positive_dim(input, dim);
    if (normalized != input.ndim() - 1) {
        throw std::invalid_argument("the readable softmax currently supports the last dimension");
    }
    const auto width = input.shape().back();
    if (width == 0) throw std::invalid_argument("softmax dimension cannot be empty");
    auto output = input.to_vector();
    const auto rows = input.numel() / width;
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
    return from_values(std::move(output), input.shape());
}

Tensor rms_norm(const Tensor& input, const Tensor& weight, float epsilon) {
    require_cpu_float(input, "input");
    require_cpu_float(weight, "weight");
    if (epsilon <= 0.0F) throw std::invalid_argument("rms_norm epsilon must be positive");
    if (weight.ndim() != 1 || input.ndim() == 0 || weight.shape()[0] != input.shape().back()) {
        throw std::invalid_argument("rms_norm weight must match the last input dimension");
    }
    const auto width = input.shape().back();
    const auto rows = input.numel() / width;
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
    return from_values(std::move(output), input.shape());
}

Tensor silu(const Tensor& input) {
    require_cpu_float(input, "input");
    auto values = input.to_vector();
    for (auto& value : values) value *= sigmoid(value);
    return from_values(std::move(values), input.shape());
}

Tensor swiglu(const Tensor& gate, const Tensor& up) {
    require_cpu_float(gate, "gate");
    require_cpu_float(up, "up");
    require_same_shape(gate, up);
    auto gate_values = gate.to_vector();
    const auto up_values = up.to_vector();
    for (std::size_t index = 0; index < gate_values.size(); ++index) {
        gate_values[index] *= sigmoid(gate_values[index]) * up_values[index];
    }
    return from_values(std::move(gate_values), gate.shape());
}

Tensor rope(const Tensor& input, std::int64_t sequence_dim, std::int64_t position_offset,
            float base) {
    require_cpu_float(input, "input");
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
    const auto values = input.to_vector();
    auto output = values;
    const auto sequence_stride = contiguous_strides(input.shape())[static_cast<std::size_t>(sequence)];
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
    return from_values(std::move(output), input.shape());
}

Tensor cross_entropy(const Tensor& logits, const Tensor& targets) {
    require_cpu_float(logits, "logits");
    if (!targets.device().is_cpu() || targets.dtype() != DType::Int32) {
        throw std::invalid_argument("cross_entropy targets must be CPU int32");
    }
    if (logits.ndim() < 1 || logits.shape().back() <= 0) {
        throw std::invalid_argument("cross_entropy logits require a non-empty class dimension");
    }
    Shape expected(logits.shape().begin(), logits.shape().end() - 1);
    if (targets.shape() != expected) throw std::invalid_argument("target shape must match logits prefix");
    const auto classes = logits.shape().back();
    const auto rows = logits.numel() / classes;
    if (rows == 0) throw std::invalid_argument("cross_entropy requires at least one target");
    const auto values = logits.to_vector();
    const auto labels = targets.to_int32_vector();
    double total = 0.0;
    for (std::int64_t row = 0; row < rows; ++row) {
        const auto label = labels[static_cast<std::size_t>(row)];
        if (label < 0 || label >= classes) throw std::out_of_range("target class out of range");
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
    return from_values({static_cast<float>(total / static_cast<double>(rows))}, {});
}

}  // namespace microllm::ops
