#include <microllm/autograd/autograd.h>

#include <algorithm>
#include <cmath>
#include <functional>
#include <limits>
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
    if (value.data().dtype() != DType::Float32) {
        throw std::invalid_argument(std::string(name) + " autograd requires float32");
    }
}

void accumulate(const std::shared_ptr<Value::Node>& node, const Tensor& gradient) {
    if (!node->requires_grad) return;
    if (gradient.shape() != node->data.shape()) {
        throw std::invalid_argument("gradient shape does not match autograd value");
    }
    auto prepared = gradient;
    if (prepared.device() != node->data.device()) prepared = prepared.to(node->data.device());
    if (!prepared.is_contiguous()) prepared = prepared.contiguous();
    node->gradient = node->gradient.defined() ? ops::add(node->gradient, prepared) : prepared;
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

}  // namespace

Value::Value(Tensor data, bool requires_grad) : node_(std::make_shared<Node>()) {
    if (!data.defined()) throw std::invalid_argument("autograd Value data must be defined");
    if (requires_grad && data.dtype() != DType::Float32) {
        throw std::invalid_argument("autograd parameters must be float32");
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
void Value::set_grad(Tensor gradient) {
    if (!node_ || !node_->requires_grad) {
        throw std::logic_error("cannot set gradient on a non-differentiable Value");
    }
    if (gradient.shape() != node_->data.shape() || gradient.dtype() != DType::Float32 ||
        gradient.device() != node_->data.device()) {
        throw std::invalid_argument("assigned gradient must match parameter shape/device/dtype");
    }
    node_->gradient = gradient.is_contiguous() ? std::move(gradient) : gradient.contiguous();
}
void Value::zero_grad() {
    if (node_) node_->gradient = Tensor();
}

void Value::backward() const {
    if (data().numel() != 1) {
        throw std::invalid_argument("implicit backward gradient requires a scalar output");
    }
    Tensor seed(data().shape(), DType::Float32, data().device());
    ops::fill_(seed, 1.0F);
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
    const auto left_forward = left.data().is_contiguous() ? left.data() : left.data().contiguous();
    const auto right_forward =
        right.data().is_contiguous() ? right.data() : right.data().contiguous();
    return operation(ops::matmul_with_implementation(
                         left_forward, right_forward, ops::MatmulImplementation::Auto),
                     {left_node, right_node},
                     [left_node, right_node, left_last, right_last](const Tensor& gradient) {
                         const auto right_transposed =
                             right_node->data.transpose(right_last - 1, right_last).contiguous();
                         const auto left_transposed =
                             left_node->data.transpose(left_last - 1, left_last).contiguous();
                         accumulate(left_node, ops::matmul_with_implementation(
                                                   gradient, right_transposed,
                                                   ops::MatmulImplementation::Auto));
                         accumulate(right_node, ops::matmul_with_implementation(
                                                    left_transposed, gradient,
                                                    ops::MatmulImplementation::Auto));
                     });
}

Value sum(const Value& input) {
    require_value(input, "input");
    auto input_node = input.node_;
    return operation(ops::reduce_sum(input.data()), {input_node},
                     [input_node](const Tensor& gradient) {
                         accumulate(input_node,
                                    ops::broadcast_scalar(gradient, input_node->data.shape()));
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
                         accumulate(input_node, gradient.contiguous().reshape(original_shape));
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
    if (indices.dtype() != DType::Int32 || indices.device() != weight.data().device()) {
        throw std::invalid_argument("autograd embedding indices must match weight device");
    }
    auto weight_node = weight.node_;
    const auto vocabulary = weight.data().shape()[0];
    return operation(ops::embedding(weight.data(), indices), {weight_node},
                     [weight_node, indices, vocabulary](const Tensor& gradient) {
                         accumulate(weight_node,
                                    ops::embedding_backward(gradient, indices, vocabulary));
                     });
}

Value softmax(const Value& input, std::int64_t dim) {
    require_value(input, "input");
    auto input_node = input.node_;
    const auto output = ops::softmax(input.data(), dim);
    return operation(output, {input_node},
                     [input_node, output](const Tensor& gradient) {
                         accumulate(input_node, ops::softmax_backward(output, gradient));
                     });
}

Value rms_norm(const Value& input, const Value& weight, float epsilon) {
    require_value(input, "input");
    require_value(weight, "weight");
    auto input_node = input.node_;
    auto weight_node = weight.node_;
    return operation(ops::rms_norm(input.data(), weight.data(), epsilon),
                     {input_node, weight_node},
                     [input_node, weight_node, epsilon](const Tensor& gradient) {
                         auto gradients = ops::rms_norm_backward(
                             input_node->data, weight_node->data, gradient, epsilon);
                         accumulate(input_node, gradients.first);
                         accumulate(weight_node, gradients.second);
                     });
}

Value silu(const Value& input) {
    require_value(input, "input");
    auto input_node = input.node_;
    return operation(ops::silu(input.data()), {input_node},
                     [input_node](const Tensor& gradient) {
                         accumulate(input_node, ops::silu_backward(input_node->data, gradient));
                     });
}

Value swiglu(const Value& gate, const Value& up) {
    require_value(gate, "gate");
    require_value(up, "up");
    auto gate_node = gate.node_;
    auto up_node = up.node_;
    return operation(ops::swiglu(gate.data(), up.data()), {gate_node, up_node},
                     [gate_node, up_node](const Tensor& gradient) {
                         auto gradients =
                             ops::swiglu_backward(gate_node->data, up_node->data, gradient);
                         accumulate(gate_node, gradients.first);
                         accumulate(up_node, gradients.second);
                     });
}

Value rope(const Value& input, std::int64_t sequence_dim, std::int64_t position_offset,
           float base) {
    require_value(input, "input");
    auto input_node = input.node_;
    const auto forward_input =
        input.data().is_contiguous() ? input.data() : input.data().contiguous();
    return operation(ops::rope(forward_input, sequence_dim, position_offset, base), {input_node},
                     [input_node, sequence_dim, position_offset,
                      base](const Tensor& gradient) {
                         accumulate(input_node, ops::rope_backward(
                                                    gradient, sequence_dim, position_offset, base));
                     });
}

Value cross_entropy(const Value& logits, const Tensor& targets) {
    require_value(logits, "logits");
    if (targets.dtype() != DType::Int32 || targets.device() != logits.data().device()) {
        throw std::invalid_argument("autograd cross_entropy targets must match logits device");
    }
    auto logits_node = logits.node_;
    return operation(ops::cross_entropy(logits.data(), targets), {logits_node},
                     [logits_node, targets](const Tensor& gradient) {
                         accumulate(logits_node, ops::cross_entropy_backward(
                                                     logits_node->data, targets, gradient));
                     });
}

Value contiguous(const Value& input) {
    require_value(input, "input");
    auto input_node = input.node_;
    return operation(input.data().contiguous(), {input_node},
                     [input_node](const Tensor& gradient) { accumulate(input_node, gradient); });
}

Value causal_softmax(const Value& scores) {
    require_value(scores, "scores");
    auto score_node = scores.node_;
    const auto output = ops::causal_softmax(scores.data());
    return operation(output, {score_node},
                     [score_node, output](const Tensor& gradient) {
                         accumulate(score_node,
                                    ops::causal_softmax_backward(output, gradient));
                     });
}

Value repeat_interleave(const Value& input, std::int64_t dim, std::int64_t repeats) {
    require_value(input, "input");
    if (dim < 0) dim += input.data().ndim();
    if (dim < 0 || dim >= input.data().ndim()) {
        throw std::out_of_range("repeat_interleave dimension is out of range");
    }
    if (repeats <= 0) throw std::invalid_argument("repeat_interleave repeats must be positive");
    const auto input_shape = input.data().shape();
    auto output_shape = input_shape;
    if (output_shape[static_cast<std::size_t>(dim)] >
        std::numeric_limits<std::int64_t>::max() / repeats) {
        throw std::overflow_error("repeat_interleave shape overflows int64");
    }
    output_shape[static_cast<std::size_t>(dim)] *= repeats;
    auto input_node = input.node_;
    return operation(ops::repeat_interleave(input.data(), dim, repeats), {input_node},
                     [input_node, input_shape, dim, repeats](const Tensor& gradient) {
                         accumulate(input_node, ops::repeat_interleave_backward(
                                                    gradient, input_shape, dim, repeats));
                     });
}

}  // namespace microllm::autograd
