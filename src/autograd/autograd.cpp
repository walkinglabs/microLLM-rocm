#include <microllm/autograd/autograd.h>

#include <algorithm>
#include <cmath>
#include <functional>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

#include <microllm/ops/ops.h>

namespace microllm::autograd {

struct Value::Node {
    Tensor data;
    Tensor gradient;
    bool requires_grad = false;
    std::vector<std::shared_ptr<Node>> parents;
    std::function<void(const Tensor&)> backward;
};

struct ValueAccess {
    static Value make(std::shared_ptr<Value::Node> node) { return Value(std::move(node)); }
};

namespace {

void require_value(const Value& value, const char* name) {
    if (!value.defined()) throw std::invalid_argument(std::string(name) + " is undefined");
    if (!value.data().device().is_cpu() || value.data().dtype() != DType::Float32) {
        throw std::invalid_argument(std::string(name) + " autograd requires CPU float32");
    }
}

void accumulate(const std::shared_ptr<Value::Node>& node, const Tensor& gradient) {
    if (!node->requires_grad) return;
    if (gradient.shape() != node->data.shape()) {
        throw std::invalid_argument("gradient shape does not match autograd value");
    }
    node->gradient = node->gradient.defined() ? ops::add(node->gradient, gradient) : gradient;
}

Value operation(Tensor data, std::vector<std::shared_ptr<Value::Node>> parents,
                std::function<void(const Tensor&)> backward) {
    auto node = std::make_shared<Value::Node>();
    node->data = std::move(data);
    node->requires_grad = std::any_of(parents.begin(), parents.end(),
                                      [](const auto& parent) { return parent->requires_grad; });
    if (node->requires_grad) {
        node->parents = std::move(parents);
        node->backward = std::move(backward);
    }
    return ValueAccess::make(std::move(node));
}

Tensor filled_like(const Tensor& input, float value) {
    Tensor output(input.shape());
    output.fill(value);
    return output;
}

float sigmoid(float value) {
    if (value >= 0.0F) return 1.0F / (1.0F + std::exp(-value));
    const auto exponential = std::exp(value);
    return exponential / (1.0F + exponential);
}

}  // namespace

Value::Value(Tensor data, bool requires_grad) : node_(std::make_shared<Node>()) {
    if (!data.defined()) throw std::invalid_argument("autograd Value data must be defined");
    if (requires_grad && (!data.device().is_cpu() || data.dtype() != DType::Float32)) {
        throw std::invalid_argument("the first autograd path supports CPU float32 only");
    }
    node_->data = std::move(data);
    node_->requires_grad = requires_grad;
}

Value::Value(std::shared_ptr<Node> node) : node_(std::move(node)) {}
bool Value::defined() const noexcept { return static_cast<bool>(node_); }
bool Value::requires_grad() const noexcept { return node_ && node_->requires_grad; }
const Tensor& Value::data() const {
    if (!node_) throw std::logic_error("undefined Value has no data");
    return node_->data;
}
Tensor& Value::mutable_data() {
    if (!node_) throw std::logic_error("undefined Value has no data");
    return node_->data;
}
bool Value::has_grad() const noexcept { return node_ && node_->gradient.defined(); }
const Tensor& Value::grad() const {
    if (!has_grad()) throw std::logic_error("Value does not have a gradient");
    return node_->gradient;
}
void Value::zero_grad() {
    if (node_) node_->gradient = Tensor();
}

void Value::backward() const {
    if (data().numel() != 1) {
        throw std::invalid_argument("implicit backward gradient requires a scalar output");
    }
    Tensor seed(data().shape());
    seed.fill(1.0F);
    backward(seed);
}

void Value::backward(const Tensor& gradient) const {
    if (!node_) throw std::logic_error("cannot backpropagate an undefined Value");
    if (!node_->requires_grad) throw std::logic_error("Value does not require gradients");
    if (gradient.shape() != node_->data.shape()) {
        throw std::invalid_argument("backward seed shape must match output");
    }

    std::vector<std::shared_ptr<Node>> topological;
    std::unordered_set<Node*> visited;
    std::function<void(const std::shared_ptr<Node>&)> visit = [&](const auto& current) {
        if (!visited.insert(current.get()).second) return;
        for (const auto& parent : current->parents) visit(parent);
        topological.push_back(current);
    };
    visit(node_);

    for (const auto& current : topological) {
        if (!current->parents.empty()) current->gradient = Tensor();
    }
    if (node_->parents.empty()) {
        accumulate(node_, gradient);
    } else {
        node_->gradient = gradient;
    }
    for (auto iterator = topological.rbegin(); iterator != topological.rend(); ++iterator) {
        const auto& current = *iterator;
        if (current->backward && current->gradient.defined()) current->backward(current->gradient);
    }
}

Value Value::detach() const { return Value(data(), false); }

Value add(const Value& left, const Value& right) {
    require_value(left, "left");
    require_value(right, "right");
    auto left_node = left.node_;
    auto right_node = right.node_;
    return operation(ops::add(left.data(), right.data()), {left_node, right_node},
                     [left_node, right_node](const Tensor& gradient) {
                         accumulate(left_node, gradient);
                         accumulate(right_node, gradient);
                     });
}

Value multiply(const Value& left, const Value& right) {
    require_value(left, "left");
    require_value(right, "right");
    auto left_node = left.node_;
    auto right_node = right.node_;
    return operation(ops::multiply(left.data(), right.data()), {left_node, right_node},
                     [left_node, right_node](const Tensor& gradient) {
                         accumulate(left_node, ops::multiply(gradient, right_node->data));
                         accumulate(right_node, ops::multiply(gradient, left_node->data));
                     });
}

Value scale(const Value& input, float factor) {
    require_value(input, "input");
    auto input_node = input.node_;
    return operation(ops::scale(input.data(), factor), {input_node},
                     [input_node, factor](const Tensor& gradient) {
                         accumulate(input_node, ops::scale(gradient, factor));
                     });
}

Value matmul(const Value& left, const Value& right) {
    require_value(left, "left");
    require_value(right, "right");
    auto left_node = left.node_;
    auto right_node = right.node_;
    const auto left_last = left.data().ndim() - 1;
    const auto right_last = right.data().ndim() - 1;
    return operation(ops::matmul(left.data(), right.data()), {left_node, right_node},
                     [left_node, right_node, left_last, right_last](const Tensor& gradient) {
                         const auto right_transposed =
                             right_node->data.transpose(right_last - 1, right_last);
                         const auto left_transposed =
                             left_node->data.transpose(left_last - 1, left_last);
                         accumulate(left_node, ops::matmul(gradient, right_transposed));
                         accumulate(right_node, ops::matmul(left_transposed, gradient));
                     });
}

Value sum(const Value& input) {
    require_value(input, "input");
    auto input_node = input.node_;
    double total = 0.0;
    for (const auto value : input.data().to_vector()) total += value;
    return operation(Tensor::from_vector({static_cast<float>(total)}, {}), {input_node},
                     [input_node](const Tensor& gradient) {
                         accumulate(input_node,
                                    filled_like(input_node->data, gradient.to_vector()[0]));
                     });
}

Value mean(const Value& input) {
    require_value(input, "input");
    if (input.data().numel() == 0) throw std::invalid_argument("mean requires non-empty input");
    return scale(sum(input), 1.0F / static_cast<float>(input.data().numel()));
}

Value reshape(const Value& input, Shape shape) {
    require_value(input, "input");
    auto input_node = input.node_;
    const auto original_shape = input.data().shape();
    return operation(input.data().reshape(std::move(shape)), {input_node},
                     [input_node, original_shape](const Tensor& gradient) {
                         accumulate(input_node, gradient.reshape(original_shape));
                     });
}

Value transpose(const Value& input, std::int64_t dim0, std::int64_t dim1) {
    require_value(input, "input");
    auto input_node = input.node_;
    return operation(input.data().transpose(dim0, dim1), {input_node},
                     [input_node, dim0, dim1](const Tensor& gradient) {
                         accumulate(input_node, gradient.transpose(dim0, dim1));
                     });
}

Value embedding(const Value& weight, const Tensor& indices) {
    require_value(weight, "weight");
    if (!indices.device().is_cpu() || indices.dtype() != DType::Int32) {
        throw std::invalid_argument("autograd embedding indices must be CPU int32");
    }
    auto weight_node = weight.node_;
    const auto saved_indices = indices.to_int32_vector();
    const auto vocabulary = weight.data().shape()[0];
    const auto width = weight.data().shape()[1];
    return operation(ops::embedding(weight.data(), indices), {weight_node},
                     [weight_node, saved_indices, vocabulary, width](const Tensor& gradient) {
                         std::vector<float> weight_gradient(
                             static_cast<std::size_t>(vocabulary * width), 0.0F);
                         const auto output_gradient = gradient.to_vector();
                         for (std::size_t token = 0; token < saved_indices.size(); ++token) {
                             const auto index = static_cast<std::int64_t>(saved_indices[token]);
                             for (std::int64_t column = 0; column < width; ++column) {
                                 weight_gradient[static_cast<std::size_t>(index * width + column)] +=
                                     output_gradient[token * static_cast<std::size_t>(width) +
                                                     static_cast<std::size_t>(column)];
                             }
                         }
                         accumulate(weight_node, Tensor::from_vector(
                                                     weight_gradient, weight_node->data.shape()));
                     });
}

Value softmax(const Value& input, std::int64_t dim) {
    require_value(input, "input");
    auto input_node = input.node_;
    const auto output = ops::softmax(input.data(), dim);
    const auto saved_output = output.to_vector();
    const auto width = input.data().shape().back();
    return operation(output, {input_node},
                     [input_node, saved_output, width](const Tensor& gradient) {
                         const auto output_gradient = gradient.to_vector();
                         std::vector<float> input_gradient(output_gradient.size());
                         const auto rows = static_cast<std::int64_t>(output_gradient.size()) / width;
                         for (std::int64_t row = 0; row < rows; ++row) {
                             float dot = 0.0F;
                             for (std::int64_t column = 0; column < width; ++column) {
                                 const auto index = static_cast<std::size_t>(row * width + column);
                                 dot += output_gradient[index] * saved_output[index];
                             }
                             for (std::int64_t column = 0; column < width; ++column) {
                                 const auto index = static_cast<std::size_t>(row * width + column);
                                 input_gradient[index] = saved_output[index] *
                                                         (output_gradient[index] - dot);
                             }
                         }
                         accumulate(input_node, Tensor::from_vector(
                                                    input_gradient, input_node->data.shape()));
                     });
}

Value rms_norm(const Value& input, const Value& weight, float epsilon) {
    require_value(input, "input");
    require_value(weight, "weight");
    auto input_node = input.node_;
    auto weight_node = weight.node_;
    const auto input_values = input.data().to_vector();
    const auto weight_values = weight.data().to_vector();
    const auto width = input.data().shape().back();
    return operation(ops::rms_norm(input.data(), weight.data(), epsilon),
                     {input_node, weight_node},
                     [input_node, weight_node, input_values, weight_values, width,
                      epsilon](const Tensor& gradient) {
                         const auto output_gradient = gradient.to_vector();
                         std::vector<float> input_gradient(input_values.size());
                         std::vector<float> weight_gradient(static_cast<std::size_t>(width), 0.0F);
                         const auto rows = static_cast<std::int64_t>(input_values.size()) / width;
                         for (std::int64_t row = 0; row < rows; ++row) {
                             float square_sum = 0.0F;
                             float weighted_dot = 0.0F;
                             for (std::int64_t column = 0; column < width; ++column) {
                                 const auto index = static_cast<std::size_t>(row * width + column);
                                 square_sum += input_values[index] * input_values[index];
                                 weighted_dot += output_gradient[index] *
                                                 weight_values[static_cast<std::size_t>(column)] *
                                                 input_values[index];
                             }
                             const auto inverse_rms =
                                 1.0F / std::sqrt(square_sum / static_cast<float>(width) + epsilon);
                             const auto correction = inverse_rms * inverse_rms * inverse_rms *
                                                     weighted_dot / static_cast<float>(width);
                             for (std::int64_t column = 0; column < width; ++column) {
                                 const auto index = static_cast<std::size_t>(row * width + column);
                                 input_gradient[index] =
                                     output_gradient[index] *
                                         weight_values[static_cast<std::size_t>(column)] * inverse_rms -
                                     input_values[index] * correction;
                                 weight_gradient[static_cast<std::size_t>(column)] +=
                                     output_gradient[index] * input_values[index] * inverse_rms;
                             }
                         }
                         accumulate(input_node, Tensor::from_vector(
                                                    input_gradient, input_node->data.shape()));
                         accumulate(weight_node, Tensor::from_vector(
                                                     weight_gradient, weight_node->data.shape()));
                     });
}

Value silu(const Value& input) {
    require_value(input, "input");
    auto input_node = input.node_;
    const auto values = input.data().to_vector();
    return operation(ops::silu(input.data()), {input_node},
                     [input_node, values](const Tensor& gradient) {
                         auto input_gradient = gradient.to_vector();
                         for (std::size_t index = 0; index < values.size(); ++index) {
                             const auto probability = sigmoid(values[index]);
                             input_gradient[index] *= probability *
                                                      (1.0F + values[index] *
                                                                  (1.0F - probability));
                         }
                         accumulate(input_node, Tensor::from_vector(
                                                    input_gradient, input_node->data.shape()));
                     });
}

Value swiglu(const Value& gate, const Value& up) {
    require_value(gate, "gate");
    require_value(up, "up");
    auto gate_node = gate.node_;
    auto up_node = up.node_;
    const auto gate_values = gate.data().to_vector();
    const auto up_values = up.data().to_vector();
    return operation(ops::swiglu(gate.data(), up.data()), {gate_node, up_node},
                     [gate_node, up_node, gate_values, up_values](const Tensor& gradient) {
                         const auto output_gradient = gradient.to_vector();
                         std::vector<float> gate_gradient(output_gradient.size());
                         std::vector<float> up_gradient(output_gradient.size());
                         for (std::size_t index = 0; index < output_gradient.size(); ++index) {
                             const auto probability = sigmoid(gate_values[index]);
                             const auto silu_value = gate_values[index] * probability;
                             gate_gradient[index] = output_gradient[index] * up_values[index] *
                                                    probability *
                                                    (1.0F + gate_values[index] *
                                                                (1.0F - probability));
                             up_gradient[index] = output_gradient[index] * silu_value;
                         }
                         accumulate(gate_node, Tensor::from_vector(
                                                   gate_gradient, gate_node->data.shape()));
                         accumulate(up_node, Tensor::from_vector(up_gradient,
                                                                 up_node->data.shape()));
                     });
}

Value rope(const Value& input, std::int64_t sequence_dim, std::int64_t position_offset,
           float base) {
    require_value(input, "input");
    auto input_node = input.node_;
    const auto shape = input.data().shape();
    if (sequence_dim < 0) sequence_dim += input.data().ndim();
    const auto head_width = shape.back();
    const auto sequence_stride = contiguous_strides(shape)[static_cast<std::size_t>(sequence_dim)];
    const auto sequence_size = shape[static_cast<std::size_t>(sequence_dim)];
    return operation(ops::rope(input.data(), sequence_dim, position_offset, base), {input_node},
                     [input_node, head_width, sequence_stride, sequence_size, position_offset,
                      base](const Tensor& gradient) {
                         const auto values = gradient.to_vector();
                         auto input_gradient = values;
                         for (std::int64_t linear = 0; linear < gradient.numel();
                              linear += head_width) {
                             const auto position = (linear / sequence_stride) % sequence_size;
                             for (std::int64_t pair = 0; pair < head_width / 2; ++pair) {
                                 const auto angle = static_cast<float>(position + position_offset) *
                                                    std::pow(base,
                                                             -2.0F * static_cast<float>(pair) /
                                                                 static_cast<float>(head_width));
                                 const auto cosine = std::cos(angle);
                                 const auto sine = std::sin(angle);
                                 const auto even = static_cast<std::size_t>(linear + pair * 2);
                                 const auto odd = even + 1;
                                 input_gradient[even] = values[even] * cosine + values[odd] * sine;
                                 input_gradient[odd] = -values[even] * sine + values[odd] * cosine;
                             }
                         }
                         accumulate(input_node, Tensor::from_vector(
                                                    input_gradient, input_node->data.shape()));
                     });
}

Value cross_entropy(const Value& logits, const Tensor& targets) {
    require_value(logits, "logits");
    if (!targets.device().is_cpu() || targets.dtype() != DType::Int32) {
        throw std::invalid_argument("autograd cross_entropy targets must be CPU int32");
    }
    auto logits_node = logits.node_;
    const auto probabilities = ops::softmax(logits.data()).to_vector();
    const auto labels = targets.to_int32_vector();
    const auto classes = logits.data().shape().back();
    const auto rows = logits.data().numel() / classes;
    return operation(ops::cross_entropy(logits.data(), targets), {logits_node},
                     [logits_node, probabilities, labels, classes,
                      rows](const Tensor& gradient) {
                         auto logits_gradient = probabilities;
                         for (std::int64_t row = 0; row < rows; ++row) {
                             logits_gradient[static_cast<std::size_t>(
                                 row * classes + labels[static_cast<std::size_t>(row)])] -= 1.0F;
                         }
                         const auto factor = gradient.to_vector()[0] / static_cast<float>(rows);
                         for (auto& value : logits_gradient) value *= factor;
                         accumulate(logits_node, Tensor::from_vector(
                                                     logits_gradient, logits_node->data.shape()));
                     });
}

}  // namespace microllm::autograd
