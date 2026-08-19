#include <microllm/autograd/autograd.h>

#include <algorithm>
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

}  // namespace microllm::autograd
