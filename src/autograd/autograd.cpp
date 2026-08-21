#include <microllm/autograd/autograd.h>

#include <algorithm>
#include <cmath>
#include <functional>
#include <limits>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include <microllm/ops/ops.h>
#include <microllm/profiling/trace.h>

namespace microllm::autograd {

struct Value::Node {
    Tensor data;
    Tensor gradient;
    std::string operation = "leaf";
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

template <typename Forward>
Tensor profiled_tensor(const char* name, Device device, Forward&& forward) {
    profiling::TraceTimer timer(profiling::TraceKind::Operator, name, device);
    auto output = forward();
    timer.finish(output);
    return output;
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

Value operation(const char* operation_name, Tensor data,
                std::vector<std::shared_ptr<Value::Node>> parents,
                std::function<void(const Tensor&)> backward) {
    auto node = std::make_shared<Value::Node>();
    node->data = std::move(data);
    node->operation = operation_name;
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
    auto output = profiled_tensor("add", left.data().device(),
                                  [&] { return ops::add(left.data(), right.data()); });
    return operation("add", std::move(output), {left_node, right_node},
                     [left_node, right_node](const Tensor& gradient) {
                         accumulate(left_node, gradient);
                         accumulate(right_node, gradient);
                     });
}

Value add_bias(const Value& input, const Value& bias) {
    require_value(input, "input");
    require_value(bias, "bias");
    auto input_node = input.node_;
    auto bias_node = bias.node_;
    auto output = profiled_tensor("add_bias", input.data().device(),
                                  [&] { return ops::add_bias(input.data(), bias.data()); });
    return operation("add_bias", std::move(output), {input_node, bias_node},
                     [input_node, bias_node](const Tensor& gradient) {
                         accumulate(input_node, gradient);
                         accumulate(bias_node, ops::bias_gradient(gradient));
                     });
}

Value multiply(const Value& left, const Value& right) {
    require_value(left, "left");
    require_value(right, "right");
    auto left_node = left.node_;
    auto right_node = right.node_;
    auto output = profiled_tensor("multiply", left.data().device(),
                                  [&] { return ops::multiply(left.data(), right.data()); });
    return operation("multiply", std::move(output),
                     {left_node, right_node},
                     [left_node, right_node](const Tensor& gradient) {
                         accumulate(left_node, ops::multiply(gradient, right_node->data));
                         accumulate(right_node, ops::multiply(gradient, left_node->data));
                     });
}

Value scale(const Value& input, float factor) {
    require_value(input, "input");
    auto input_node = input.node_;
    auto output = profiled_tensor("scale", input.data().device(),
                                  [&] { return ops::scale(input.data(), factor); });
    return operation("scale", std::move(output), {input_node},
                     [input_node, factor](const Tensor& gradient) {
                         accumulate(input_node, ops::scale(gradient, factor));
                     });
}

Value matmul(const Value& left, const Value& right) {
    require_value(left, "left");
    require_value(right, "right");
    if (left.data().ndim() == 2 && right.data().ndim() == 2 &&
        left.data().is_contiguous() && right.data().is_contiguous()) {
        return matmul(left, right, false, false);
    }
    auto left_node = left.node_;
    auto right_node = right.node_;
    const auto left_last = left.data().ndim() - 1;
    const auto right_last = right.data().ndim() - 1;
    const auto left_forward = left.data().is_contiguous() ? left.data() : left.data().contiguous();
    const auto right_forward =
        right.data().is_contiguous() ? right.data() : right.data().contiguous();
    auto output = profiled_tensor("matmul", left.data().device(), [&] {
        return ops::matmul_with_implementation(left_forward, right_forward,
                                               ops::MatmulImplementation::Auto);
    });
    return operation("matmul", std::move(output),
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

Value matmul(const Value& left, const Value& right,
             bool transpose_left, bool transpose_right) {
    require_value(left, "left");
    require_value(right, "right");
    auto left_node = left.node_;
    auto right_node = right.node_;
    auto output = profiled_tensor("matmul", left.data().device(), [&] {
        return ops::matmul_with_implementation(
            left.data(), right.data(), ops::MatmulImplementation::Auto,
            transpose_left, transpose_right);
    });
    return operation(
        "matmul", std::move(output), {left_node, right_node},
        [left_node, right_node, transpose_left, transpose_right](const Tensor& gradient) {
            Tensor left_gradient;
            if (!transpose_left) {
                left_gradient = ops::matmul_with_implementation(
                    gradient, right_node->data, ops::MatmulImplementation::Auto,
                    false, !transpose_right);
            } else {
                left_gradient = ops::matmul_with_implementation(
                    right_node->data, gradient, ops::MatmulImplementation::Auto,
                    transpose_right, true);
            }

            Tensor right_gradient;
            if (!transpose_right) {
                right_gradient = ops::matmul_with_implementation(
                    left_node->data, gradient, ops::MatmulImplementation::Auto,
                    !transpose_left, false);
            } else {
                right_gradient = ops::matmul_with_implementation(
                    gradient, left_node->data, ops::MatmulImplementation::Auto,
                    true, transpose_left);
            }
            accumulate(left_node, std::move(left_gradient));
            accumulate(right_node, std::move(right_gradient));
        });
}

Value fp8_matmul(const Value& left, const Value& right,
                 float left_scale, float right_scale, DType fp8_dtype) {
    require_value(left, "left");
    require_value(right, "right");
    auto left_node = left.node_;
    auto right_node = right.node_;
    const auto left_last = left.data().ndim() - 1;
    const auto right_last = right.data().ndim() - 1;
    const auto left_forward = left.data().is_contiguous() ? left.data() : left.data().contiguous();
    const auto right_forward = right.data().is_contiguous() ? right.data() : right.data().contiguous();
    auto output = profiled_tensor("fp8_matmul", left.data().device(), [&] {
        const auto quantized_left = ops::quantize_fp8(left_forward, fp8_dtype, left_scale);
        const auto quantized_right = ops::quantize_fp8(right_forward, fp8_dtype, right_scale);
        return ops::fp8_matmul(quantized_left, quantized_right, DType::Float32);
    });
    return operation("fp8_matmul", std::move(output), {left_node, right_node},
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

Value bf16_matmul(const Value& left, const Value& right) {
    require_value(left, "left");
    require_value(right, "right");
    if (left.data().ndim() != 2 || right.data().ndim() != 2) {
        throw std::invalid_argument("BF16 autograd matmul currently requires 2D tensors");
    }
    if (left.data().dtype() != DType::Float32 ||
        right.data().dtype() != DType::Float32) {
        throw std::invalid_argument(
            "BF16 autograd matmul requires FP32 master tensors");
    }
    const auto right_forward = right.data().is_contiguous()
                                   ? right.data() : right.data().contiguous();
    const auto mirror = ops::cast(right_forward, DType::BFloat16);
    return bf16_matmul(left, right, mirror);
}

Value bf16_matmul(const Value& left, const Value& right_master,
                  const Tensor& right_bf16_mirror) {
    require_value(left, "left");
    require_value(right_master, "right_master");
    if (left.data().ndim() != 2 || right_master.data().ndim() != 2 ||
        right_bf16_mirror.ndim() != 2 ||
        right_bf16_mirror.dtype() != DType::BFloat16 ||
        right_bf16_mirror.shape() != right_master.data().shape() ||
        right_bf16_mirror.device() != right_master.data().device() ||
        !right_bf16_mirror.is_contiguous()) {
        throw std::invalid_argument(
            "BF16 cached autograd matmul requires a matching contiguous mirror");
    }
    auto left_node = left.node_;
    auto right_node = right_master.node_;
    const auto left_forward = left.data().is_contiguous()
                                  ? left.data() : left.data().contiguous();
    auto output = profiled_tensor("bf16_matmul_cached", left.data().device(), [&] {
        return ops::bf16_matmul(left_forward, right_bf16_mirror);
    });
    return operation("bf16_matmul", std::move(output), {left_node, right_node},
                     [left_node, right_node](const Tensor& gradient) {
                         accumulate(left_node, ops::matmul_with_implementation(
                                                   gradient, right_node->data,
                                                   ops::MatmulImplementation::Auto,
                                                   false, true));
                         accumulate(right_node, ops::matmul_with_implementation(
                                                    left_node->data, gradient,
                                                    ops::MatmulImplementation::Auto,
                                                    true, false));
                     });
}

Value sum(const Value& input) {
    require_value(input, "input");
    auto input_node = input.node_;
    auto output = profiled_tensor("sum", input.data().device(),
                                  [&] { return ops::reduce_sum(input.data()); });
    return operation("sum", std::move(output), {input_node},
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
    auto output = profiled_tensor("reshape", input.data().device(),
                                  [&] { return input.data().reshape(std::move(shape)); });
    return operation("reshape", std::move(output), {input_node},
                     [input_node, original_shape](const Tensor& gradient) {
                         accumulate(input_node, gradient.contiguous().reshape(original_shape));
                     });
}

Value transpose(const Value& input, std::int64_t dim0, std::int64_t dim1) {
    require_value(input, "input");
    auto input_node = input.node_;
    auto output = profiled_tensor("transpose", input.data().device(),
                                  [&] { return input.data().transpose(dim0, dim1); });
    return operation("transpose", std::move(output), {input_node},
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
    auto output = profiled_tensor("embedding", weight.data().device(),
                                  [&] { return ops::embedding(weight.data(), indices); });
    return operation("embedding", std::move(output), {weight_node},
                     [weight_node, indices, vocabulary](const Tensor& gradient) {
                         accumulate(weight_node,
                                    ops::embedding_backward(gradient, indices, vocabulary));
                     });
}

Value softmax(const Value& input, std::int64_t dim) {
    require_value(input, "input");
    auto input_node = input.node_;
    const auto output = profiled_tensor("softmax", input.data().device(),
                                        [&] { return ops::softmax(input.data(), dim); });
    return operation("softmax", output, {input_node},
                     [input_node, output](const Tensor& gradient) {
                         accumulate(input_node, ops::softmax_backward(output, gradient));
                     });
}

Value rms_norm(const Value& input, const Value& weight, float epsilon) {
    require_value(input, "input");
    require_value(weight, "weight");
    auto input_node = input.node_;
    auto weight_node = weight.node_;
    auto output = profiled_tensor("rms_norm", input.data().device(), [&] {
        return ops::rms_norm(input.data(), weight.data(), epsilon);
    });
    return operation("rms_norm", std::move(output),
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
    auto output = profiled_tensor("silu", input.data().device(),
                                  [&] { return ops::silu(input.data()); });
    return operation("silu", std::move(output), {input_node},
                     [input_node](const Tensor& gradient) {
                         accumulate(input_node, ops::silu_backward(input_node->data, gradient));
                     });
}

Value swiglu(const Value& gate, const Value& up) {
    require_value(gate, "gate");
    require_value(up, "up");
    auto gate_node = gate.node_;
    auto up_node = up.node_;
    auto output = profiled_tensor("swiglu", gate.data().device(),
                                  [&] { return ops::swiglu(gate.data(), up.data()); });
    return operation("swiglu", std::move(output), {gate_node, up_node},
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
    auto output = profiled_tensor("rope", input.data().device(), [&] {
        return ops::rope(forward_input, sequence_dim, position_offset, base);
    });
    return operation("rope", std::move(output),
                     {input_node},
                     [input_node, sequence_dim, position_offset,
                      base](const Tensor& gradient) {
                         accumulate(input_node, ops::rope_backward(
                                                    gradient, sequence_dim, position_offset, base));
                     });
}

Value rope_split_half(const Value& input, std::int64_t sequence_dim,
                      std::int64_t position_offset, float base) {
    require_value(input, "input");
    auto input_node = input.node_;
    const auto forward_input =
        input.data().is_contiguous() ? input.data() : input.data().contiguous();
    auto output = profiled_tensor("rope_split_half", input.data().device(), [&] {
        return ops::rope_split_half(forward_input, sequence_dim, position_offset, base);
    });
    return operation("rope_split_half", std::move(output), {input_node},
                     [input_node, sequence_dim, position_offset,
                      base](const Tensor& gradient) {
                         accumulate(input_node, ops::rope_split_half_backward(
                                                    gradient, sequence_dim,
                                                    position_offset, base));
                     });
}

Value rope_split_half_bias(const Value& input, const Value& bias,
                           std::int64_t position_offset, float base) {
    require_value(input, "input");
    require_value(bias, "bias");
    auto input_node = input.node_;
    auto bias_node = bias.node_;
    const auto forward_input =
        input.data().is_contiguous() ? input.data() : input.data().contiguous();
    auto output = profiled_tensor("rope_split_half_bias", input.data().device(), [&] {
        return ops::rope_split_half_bias(forward_input, bias.data(), position_offset, base);
    });
    return operation("rope_split_half_bias", std::move(output), {input_node, bias_node},
                     [input_node, bias_node, position_offset, base](const Tensor& gradient) {
                         auto pre_rope = ops::rope_split_half_backward(
                             gradient, 2, position_offset, base);
                         accumulate(input_node, pre_rope);
                         const auto& shape = pre_rope.shape();
                         auto flat = pre_rope.transpose(1, 2).contiguous().reshape(
                             {shape[0] * shape[2], shape[1] * shape[3]});
                         accumulate(bias_node, ops::bias_gradient(flat));
                     });
}

Value cross_entropy(const Value& logits, const Tensor& targets) {
    require_value(logits, "logits");
    if (targets.dtype() != DType::Int32 || targets.device() != logits.data().device()) {
        throw std::invalid_argument("autograd cross_entropy targets must match logits device");
    }
    auto logits_node = logits.node_;
    auto output = profiled_tensor("cross_entropy", logits.data().device(),
                                  [&] { return ops::cross_entropy(logits.data(), targets); });
    return operation("cross_entropy", std::move(output),
                     {logits_node},
                     [logits_node, targets](const Tensor& gradient) {
                         accumulate(logits_node, ops::cross_entropy_backward(
                                                     logits_node->data, targets, gradient));
                     });
}

Value contiguous(const Value& input) {
    require_value(input, "input");
    auto input_node = input.node_;
    auto output = profiled_tensor("contiguous", input.data().device(),
                                  [&] { return input.data().contiguous(); });
    return operation("contiguous", std::move(output), {input_node},
                     [input_node](const Tensor& gradient) { accumulate(input_node, gradient); });
}

Value causal_softmax(const Value& scores) {
    require_value(scores, "scores");
    auto score_node = scores.node_;
    const auto output = profiled_tensor("causal_softmax", scores.data().device(),
                                        [&] { return ops::causal_softmax(scores.data()); });
    return operation("causal_softmax", output, {score_node},
                     [score_node, output](const Tensor& gradient) {
                         accumulate(score_node,
                                    ops::causal_softmax_backward(output, gradient));
                     });
}

Value causal_gqa_attention(const Value& query, const Value& key,
                           const Value& value, std::int64_t repeats,
                           float scale) {
    require_value(query, "query");
    require_value(key, "key");
    require_value(value, "value");
    auto query_node = query.node_;
    auto key_node = key.node_;
    auto value_node = value.node_;
    const auto query_forward = query.data().is_contiguous()
                                   ? query.data() : query.data().contiguous();
    const auto key_forward = key.data().is_contiguous()
                                 ? key.data() : key.data().contiguous();
    const auto value_forward = value.data().is_contiguous()
                                   ? value.data() : value.data().contiguous();
    auto output = profiled_tensor("causal_gqa_attention", query.data().device(), [&] {
        return ops::causal_gqa_attention(
            query_forward, key_forward, value_forward, repeats, scale);
    });
    return operation(
        "causal_gqa_attention", std::move(output),
        {query_node, key_node, value_node},
        [query_node, key_node, value_node, query_forward, key_forward,
         value_forward, repeats, scale](const Tensor& gradient) {
            const auto prepared_gradient = gradient.is_contiguous()
                                               ? gradient : gradient.contiguous();
            auto gradients = ops::causal_gqa_attention_backward(
                query_forward, key_forward, value_forward,
                prepared_gradient, repeats, scale);
            accumulate(query_node, std::move(gradients.first));
            accumulate(key_node, std::move(gradients.second));
            accumulate(value_node, std::move(gradients.third));
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
    const auto forward_input =
        input.data().is_contiguous() ? input.data() : input.data().contiguous();
    auto output = profiled_tensor("repeat_interleave", input.data().device(), [&] {
        return ops::repeat_interleave(forward_input, dim, repeats);
    });
    return operation("repeat_interleave", std::move(output),
                     {input_node},
                     [input_node, input_shape, dim, repeats](const Tensor& gradient) {
                         accumulate(input_node, ops::repeat_interleave_backward(
                                                    gradient, input_shape, dim, repeats));
                     });
}

GraphSnapshot inspect_graph(const Value& root) {
    if (!root.node_) throw std::invalid_argument("cannot inspect an undefined graph root");
    std::vector<std::shared_ptr<Value::Node>> topological;
    std::unordered_set<Value::Node*> visited;
    std::function<void(const std::shared_ptr<Value::Node>&)> visit = [&](const auto& current) {
        if (!visited.insert(current.get()).second) return;
        for (const auto& parent : current->parents) visit(parent);
        topological.push_back(current);
    };
    visit(root.node_);

    std::unordered_map<Value::Node*, std::size_t> ids;
    for (std::size_t index = 0; index < topological.size(); ++index) {
        ids.emplace(topological[index].get(), index);
    }
    GraphSnapshot snapshot;
    snapshot.nodes.reserve(topological.size());
    for (std::size_t index = 0; index < topological.size(); ++index) {
        const auto& node = topological[index];
        GraphNodeInfo info;
        info.id = index;
        info.operation = node->operation;
        info.shape = node->data.shape();
        info.requires_grad = node->requires_grad;
        info.parents.reserve(node->parents.size());
        for (const auto& parent : node->parents) {
            info.parents.push_back(ids.at(parent.get()));
            ++snapshot.edge_count;
        }
        snapshot.nodes.push_back(std::move(info));
    }
    snapshot.root_id = ids.at(root.node_.get());
    return snapshot;
}

}  // namespace microllm::autograd
