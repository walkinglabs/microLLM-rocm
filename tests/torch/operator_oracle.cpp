#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <microllm/autograd/autograd.h>
#include <microllm/model/model.h>
#include <microllm/ops/ops.h>
#include <microllm/training/optimizer.h>

namespace {

using microllm::DType;
using microllm::Shape;
using microllm::Tensor;
using microllm::autograd::Value;

void emit(const std::string& name, const Tensor& tensor) {
    std::cout << "{\"name\":\"" << name << "\",\"shape\":[";
    for (std::size_t index = 0; index < tensor.shape().size(); ++index) {
        if (index != 0) std::cout << ',';
        std::cout << tensor.shape()[index];
    }
    std::cout << "],\"values\":[" << std::setprecision(9);
    const auto values = tensor.to_vector();
    for (std::size_t index = 0; index < values.size(); ++index) {
        if (index != 0) std::cout << ',';
        std::cout << values[index];
    }
    std::cout << "]}\n";
}

void emit_bool(const std::string& name, bool value) {
    std::cout << "{\"name\":\"" << name << "\",\"bool\":"
              << (value ? "true" : "false") << "}\n";
}

template <typename Function>
bool rejected(Function&& function) {
    try {
        function();
    } catch (const std::exception&) {
        return true;
    }
    return false;
}

Tensor f32(std::vector<float> values, Shape shape) {
    return Tensor::from_vector(values, std::move(shape));
}

void emit_forward_cases() {
    using namespace microllm::ops;
    Tensor filled({2, 3});
    fill_(filled, -1.25F);
    emit("fill", filled);

    const auto left = f32({1, -2, 3, 4, 0.5F, -0.25F}, {2, 3});
    const auto right = f32({4, 5, -6, 2, 1.5F, 0.25F}, {2, 3});
    emit("add", add(left, right));
    emit("multiply", multiply(left, right));
    emit("scale", scale(left, -0.25F));

    const auto matrix_left = f32({1, 2, 3, 4, 5, 6}, {2, 3});
    const auto matrix_right = f32({1, 2, 3, 4, 5, 6}, {3, 2});
    emit("matmul_2d", matmul(matrix_left, matrix_right));
    emit("matmul_readable",
         matmul_with_implementation(matrix_left, matrix_right,
                                    MatmulImplementation::Readable));
    emit("matmul_3d",
         matmul(f32({1, 2, 3, 4, 5, 6, 1, 0, 0, 1, 1, 1}, {2, 2, 3}),
                f32({1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 6}, {2, 3, 2})));

    const auto embedding_weight = f32({0, 1, 2, 3, 4, 5, 6, 7}, {4, 2});
    const auto indices = Tensor::from_int32_vector({2, 0, 2}, {3});
    emit("embedding", embedding(embedding_weight, indices));

    const auto nonlinear = f32({1000, 1000, 999, -2, 0, 2}, {2, 3});
    emit("softmax", softmax(nonlinear));
    emit("rms_norm", rms_norm(nonlinear, f32({1, 0.5F, 2}, {3})));
    emit("silu", silu(left));
    emit("swiglu", swiglu(left, right));

    const auto rope_input = f32({1, 0, 0, 1, 1, 0, 0, 1}, {1, 2, 1, 4});
    emit("rope", rope(rope_input));
    const auto logits = f32({2, 1, 0, 100, -100, 0}, {2, 3});
    const auto targets = Tensor::from_int32_vector({0, -100}, {2});
    emit("cross_entropy", cross_entropy(logits, targets));
    emit("reduce_sum", reduce_sum(left));
    emit("broadcast_scalar", broadcast_scalar(f32({2.5F}, {}), {2, 3}));

    const auto scores = f32({1, 2, 3, 4, 5, 6, 7, 8, 9}, {1, 3, 3});
    emit("causal_softmax", causal_softmax(scores));
    emit("repeat_interleave", repeat_interleave(f32({1, 2, 3, 4}, {2, 2}), 0, 2));
}

void emit_graph_gradient_cases() {
    using namespace microllm::autograd;
    Value a(f32({1, 2, 3, 4}, {2, 2}), true);
    Value b(f32({5, 6, 7, 8}, {2, 2}), true);
    const auto basic_output = add(multiply(a, b), scale(a, 2.0F));
    emit("graph_basic_output", basic_output.data());
    const auto basic_loss = mean(basic_output);
    emit("graph_basic_loss", basic_loss.data());
    basic_loss.backward();
    emit("graph_basic_a_grad", a.grad());
    emit("graph_basic_b_grad", b.grad());

    Value mat_left(f32({1, 2, 3, 4, 5, 6}, {2, 3}), true);
    Value mat_right(f32({1, 2, 3, 4, 5, 6}, {3, 2}), true);
    const Value mat_seed(f32({1, 2, 3, 4}, {2, 2}));
    sum(multiply(matmul(mat_left, mat_right), mat_seed)).backward();
    emit("graph_matmul_left_grad", mat_left.grad());
    emit("graph_matmul_right_grad", mat_right.grad());

    Value embed_weight(f32({0, 1, 2, 3, 4, 5, 6, 7}, {4, 2}), true);
    const auto index = Tensor::from_int32_vector({2, 0, 2}, {3});
    const Value embed_seed(f32({1, 2, 3, 4, 5, 6}, {3, 2}));
    sum(multiply(embedding(embed_weight, index), embed_seed)).backward();
    emit("graph_embedding_weight_grad", embed_weight.grad());

    Value softmax_input(f32({2, 1, 0, -1, 0, 1}, {2, 3}), true);
    const Value softmax_seed(f32({1, 2, 3, 4, 5, 6}, {2, 3}));
    sum(multiply(softmax(softmax_input), softmax_seed)).backward();
    emit("graph_softmax_input_grad", softmax_input.grad());

    Value norm_input(f32({1, 2, 3, 4, 5, 6}, {2, 3}), true);
    Value norm_weight(f32({1, 0.5F, 2}, {3}), true);
    const Value norm_seed(f32({1, -1, 2, -2, 3, -3}, {2, 3}));
    sum(multiply(rms_norm(norm_input, norm_weight), norm_seed)).backward();
    emit("graph_rms_input_grad", norm_input.grad());
    emit("graph_rms_weight_grad", norm_weight.grad());

    Value silu_input(f32({-2, -1, 0, 1, 2, 3}, {2, 3}), true);
    const Value activation_seed(f32({1, 2, 3, -1, -2, -3}, {2, 3}));
    sum(multiply(silu(silu_input), activation_seed)).backward();
    emit("graph_silu_input_grad", silu_input.grad());

    Value gate(f32({-2, -1, 0, 1, 2, 3}, {2, 3}), true);
    Value up(f32({1, 2, 3, 4, 5, 6}, {2, 3}), true);
    sum(multiply(swiglu(gate, up), activation_seed)).backward();
    emit("graph_swiglu_gate_grad", gate.grad());
    emit("graph_swiglu_up_grad", up.grad());

    Value rope_input(f32({1, 0, 0, 1, 1, 0, 0, 1}, {1, 2, 1, 4}), true);
    const Value rope_seed(f32({1, 2, 3, 4, -1, -2, -3, -4}, {1, 2, 1, 4}));
    sum(multiply(rope(rope_input), rope_seed)).backward();
    emit("graph_rope_input_grad", rope_input.grad());

    Value logits(f32({2, 1, 0, 100, -100, 0}, {2, 3}), true);
    cross_entropy(logits, Tensor::from_int32_vector({0, -100}, {2})).backward(
        f32({0.75F}, {}));
    emit("graph_cross_entropy_logits_grad", logits.grad());

    Value causal_input(f32({1, 2, 3, 4, 5, 6, 7, 8, 9}, {1, 3, 3}), true);
    const Value causal_seed(f32({1, 2, 3, -1, 0, 1, 2, -2, 0.5F}, {1, 3, 3}));
    sum(multiply(causal_softmax(causal_input), causal_seed)).backward();
    emit("graph_causal_softmax_input_grad", causal_input.grad());

    Value repeat_input(f32({1, 2, 3, 4}, {2, 2}), true);
    const Value repeat_seed(f32({1, 2, 3, 4, 5, 6, 7, 8}, {4, 2}));
    sum(multiply(repeat_interleave(repeat_input, 0, 2), repeat_seed)).backward();
    emit("graph_repeat_input_grad", repeat_input.grad());

    Value view_input(f32({0, 1, 2, 3, 4, 5}, {2, 3}), true);
    const auto packed = contiguous(transpose(view_input, 0, 1));
    const Value view_seed(f32({1, 2, 3, 4, 5, 6}, {3, 2}));
    sum(multiply(packed, view_seed)).backward();
    emit("graph_view_input_grad", view_input.grad());

    Value reshape_input(f32({0, 1, 2, 3, 4, 5}, {2, 3}), true);
    const Value reshape_seed(f32({1, 2, 3, 4, 5, 6}, {3, 2}));
    sum(multiply(reshape(reshape_input, {3, 2}), reshape_seed)).backward();
    emit("graph_reshape_input_grad", reshape_input.grad());
}

void emit_invalid_shape_cases() {
    using namespace microllm::ops;
    const auto matrix = f32({1, 2, 3, 4}, {2, 2});
    const auto vector = f32({1, 2, 3}, {3});
    emit_bool("invalid_add_shape", rejected([&] { (void)add(matrix, vector); }));
    emit_bool("invalid_multiply_shape", rejected([&] { (void)multiply(matrix, vector); }));
    emit_bool("invalid_scale_dtype", rejected([&] {
                  (void)scale(Tensor::from_int32_vector({1, 2}, {2}), 2.0F);
              }));
    emit_bool("invalid_matmul_inner", rejected([&] { (void)matmul(matrix, f32({1, 2}, {2, 1, 1})); }));
    emit_bool("invalid_embedding_weight", rejected([&] {
                  (void)embedding(vector, Tensor::from_int32_vector({0}, {1}));
              }));
    emit_bool("invalid_softmax_dim", rejected([&] { (void)softmax(matrix, 0); }));
    emit_bool("invalid_rms_weight", rejected([&] { (void)rms_norm(matrix, vector); }));
    emit_bool("invalid_silu_dtype", rejected([&] {
                  (void)silu(Tensor::from_int32_vector({1, 2}, {2}));
              }));
    emit_bool("invalid_swiglu_shape", rejected([&] { (void)swiglu(matrix, vector); }));
    emit_bool("invalid_rope_width", rejected([&] { (void)rope(f32({1, 2, 3}, {1, 1, 3})); }));
    emit_bool("invalid_cross_entropy_shape", rejected([&] {
                  (void)cross_entropy(matrix, Tensor::from_int32_vector({0}, {1}));
              }));
    emit_bool("invalid_reduce_dtype", rejected([&] {
                  (void)reduce_sum(Tensor::from_int32_vector({1, 2}, {2}));
              }));
    emit_bool("invalid_broadcast_source", rejected([&] { (void)broadcast_scalar(vector, {2}); }));
    emit_bool("invalid_causal_shape", rejected([&] { (void)causal_softmax(f32({1, 2, 3, 4, 5, 6}, {2, 3})); }));
    emit_bool("invalid_repeat_count", rejected([&] { (void)repeat_interleave(matrix, 0, 0); }));

    emit_bool("invalid_embedding_backward_shape", rejected([&] {
                  (void)embedding_backward(matrix, Tensor::from_int32_vector({0}, {1}), 3);
              }));
    emit_bool("invalid_softmax_backward_shape", rejected([&] {
                  (void)softmax_backward(matrix, vector);
              }));
    emit_bool("invalid_rms_backward_shape", rejected([&] {
                  (void)rms_norm_backward(matrix, f32({1, 2}, {2}), vector);
              }));
    emit_bool("invalid_silu_backward_shape", rejected([&] { (void)silu_backward(matrix, vector); }));
    emit_bool("invalid_swiglu_backward_shape", rejected([&] {
                  (void)swiglu_backward(matrix, matrix, vector);
              }));
    emit_bool("invalid_rope_backward_width", rejected([&] {
                  (void)rope_backward(f32({1, 2, 3}, {1, 1, 3}));
              }));
    emit_bool("invalid_cross_entropy_backward_seed", rejected([&] {
                  (void)cross_entropy_backward(matrix, Tensor::from_int32_vector({0, 1}, {2}),
                                               vector);
              }));
    emit_bool("invalid_causal_backward_shape", rejected([&] {
                  (void)causal_softmax_backward(matrix, vector);
              }));
    emit_bool("invalid_repeat_backward_shape", rejected([&] {
                  (void)repeat_interleave_backward(matrix, {2, 2}, 0, 2);
              }));
}

void emit_model_graph_case() {
    const microllm::model::ModelConfig config{.vocabulary_size = 8,
                                               .dimension = 8,
                                               .layers = 1,
                                               .heads = 2,
                                               .kv_heads = 1,
                                               .ffn_dimension = 16,
                                               .max_sequence_length = 4,
                                               .rope_base = 10000.0F,
                                               .tie_embeddings = false};
    microllm::model::TransformerModel model(config, 211);
    for (const auto& [name, parameter] : model.named_parameters()) {
        emit("model_param:" + name, parameter->data());
    }
    const auto tokens = Tensor::from_int32_vector({0, 1, 2, 3}, {1, 4});
    const auto targets = Tensor::from_int32_vector({1, 2, 3, 0}, {1, 4});
    const auto logits = model.forward(tokens);
    emit("model_logits", logits.data());
    const auto loss = microllm::autograd::cross_entropy(logits, targets);
    emit("model_loss", loss.data());
    const auto graph = microllm::autograd::inspect_graph(loss);
    emit_bool("model_graph_has_topology", graph.nodes.size() > 20 && graph.edge_count > 20 &&
                                              graph.root_id + 1 == graph.nodes.size());
    loss.backward();
    for (const auto& [name, parameter] : model.named_parameters()) {
        emit("model_grad:" + name, parameter->grad());
    }
}

void emit_optimizer_cases() {
    Value sgd_parameter(f32({1.0F, -2.0F}, {2}), true);
    sgd_parameter.set_grad(f32({0.5F, -0.25F}, {2}));
    microllm::training::SGD sgd({&sgd_parameter}, 0.1F, 0.01F);
    sgd.step();
    emit("optimizer_sgd_parameter_step1", sgd_parameter.data());

    Value adam_parameter(f32({1.0F, -2.0F}, {2}), true);
    microllm::training::AdamW adam(
        {&adam_parameter}, {.learning_rate = 0.01F,
                            .beta1 = 0.9F,
                            .beta2 = 0.99F,
                            .epsilon = 1.0e-8F,
                            .weight_decay = 0.1F});
    adam_parameter.set_grad(f32({0.5F, -0.25F}, {2}));
    adam.step();
    emit("optimizer_adamw_parameter_step1", adam_parameter.data());
    auto state = adam.state();
    emit("optimizer_adamw_first_moment_step1", state.first_moments[0]);
    emit("optimizer_adamw_second_moment_step1", state.second_moments[0]);

    adam_parameter.set_grad(f32({-1.0F, 2.0F}, {2}));
    adam.step();
    emit("optimizer_adamw_parameter_step2", adam_parameter.data());
    state = adam.state();
    emit("optimizer_adamw_first_moment_step2", state.first_moments[0]);
    emit("optimizer_adamw_second_moment_step2", state.second_moments[0]);
}

}  // namespace

int main() {
    emit_forward_cases();
    emit_graph_gradient_cases();
    emit_invalid_shape_cases();
    emit_model_graph_case();
    emit_optimizer_cases();
    return 0;
}
