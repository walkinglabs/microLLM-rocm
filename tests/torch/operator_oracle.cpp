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
    emit("cast_bf16", cast(left, DType::BFloat16));
    emit("add_bias", add_bias(left, f32({0.5F, -1.0F, 2.0F}, {3})));
    const auto residual_norm = add_rms_norm(
        left, right, f32({1, 0.5F, 2}, {3}));
    emit("add_rms_norm_sum", residual_norm.first);
    emit("add_rms_norm_normalized", residual_norm.second);

    const auto matrix_left = f32({1, 2, 3, 4, 5, 6}, {2, 3});
    const auto matrix_right = f32({1, 2, 3, 4, 5, 6}, {3, 2});
    emit("matmul_2d", matmul(matrix_left, matrix_right));
    emit("bf16_mixed_matmul",
         bf16_matmul(matrix_left, cast(matrix_right, DType::BFloat16)));
    emit("bf16_output_matmul", bf16_matmul_output(
        cast(matrix_left, DType::BFloat16), cast(matrix_right, DType::BFloat16),
        DType::BFloat16));
    const auto ffn_gate = Tensor::from_vector(
        {0.5F, -1.0F, 0.25F, 0.75F, 1.5F, -0.5F,
         -0.25F, 0.5F, 1.0F, -1.25F, 0.125F, 0.875F},
        {3, 4}, DType::BFloat16);
    const auto ffn_up = Tensor::from_vector(
        {1.0F, 0.5F, -0.75F, 0.25F, -0.5F, 1.25F,
         0.625F, -1.0F, 0.75F, -0.25F, 1.5F, 0.5F},
        {3, 4}, DType::BFloat16);
    const auto ffn_down = Tensor::from_vector(
        {0.25F, -0.5F, 1.0F, 0.75F, -1.25F, 0.5F, 0.125F, -0.875F},
        {4, 2}, DType::BFloat16);
    emit("bf16_ffn", bf16_ffn(matrix_left, ffn_gate, ffn_up, ffn_down));
    emit("matmul_readable",
         matmul_with_implementation(matrix_left, matrix_right,
                                    MatmulImplementation::Readable));
    const auto wide_right = f32(
        {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}, {3, 4});
    const auto transposed_left = f32({1, 4, 2, 5, 3, 6}, {3, 2});
    const auto transposed_right = f32(
        {1, 5, 9, 2, 6, 10, 3, 7, 11, 4, 8, 12}, {4, 3});
    emit("matmul_nn", matmul_with_implementation(
                          matrix_left, wide_right, MatmulImplementation::Readable,
                          false, false));
    emit("matmul_nt", matmul_with_implementation(
                          matrix_left, transposed_right, MatmulImplementation::Readable,
                          false, true));
    emit("matmul_tn", matmul_with_implementation(
                          transposed_left, wide_right, MatmulImplementation::Readable,
                          true, false));
    emit("matmul_tt", matmul_with_implementation(
                          transposed_left, transposed_right, MatmulImplementation::Readable,
                          true, true));
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
    emit("rope_split_half", rope_split_half(rope_input));
    const auto logits = f32({2, 1, 0, 100, -100, 0}, {2, 3});
    const auto targets = Tensor::from_int32_vector({0, -100}, {2});
    emit("cross_entropy", cross_entropy(logits, targets));
    std::vector<float> large_logits_values(3 * 257);
    for (std::size_t index = 0; index < large_logits_values.size(); ++index) {
        large_logits_values[index] =
            static_cast<float>(static_cast<int>(index % 251) - 125) * 0.03125F;
    }
    const auto large_logits = f32(std::move(large_logits_values), {3, 257});
    const auto large_targets = Tensor::from_int32_vector({17, 241, -100}, {3});
    emit("cross_entropy_large", cross_entropy(large_logits, large_targets));
    emit("reduce_sum", reduce_sum(left));
    emit("broadcast_scalar", broadcast_scalar(f32({2.5F}, {}), {2, 3}));

    const auto scores = f32({1, 2, 3, 4, 5, 6, 7, 8, 9}, {1, 3, 3});
    emit("causal_softmax", causal_softmax(scores));
    emit("repeat_interleave", repeat_interleave(f32({1, 2, 3, 4}, {2, 2}), 0, 2));
}

void emit_low_precision_forward_cases() {
    using namespace microllm::ops;
    for (const auto& [prefix, dtype] :
         {std::pair{"fp16", DType::Float16},
          std::pair{"bf16", DType::BFloat16}}) {
        const auto name = [&](const char* operation) {
            return std::string(prefix) + "_" + operation;
        };
        const auto left = Tensor::from_vector(
            {1, -2, 3, 4, 0.5F, -0.25F}, {2, 3}, dtype);
        const auto right = Tensor::from_vector(
            {4, 5, -6, 2, 1.5F, 0.25F}, {2, 3}, dtype);
        emit(name("add"), add(left, right));
        emit(name("multiply"), multiply(left, right));
        emit(name("scale"), scale(left, -0.25F));
        emit(name("silu"), silu(left));
        emit(name("swiglu"), swiglu(left, right));

        const auto mat_left = Tensor::from_vector({1, 2, 3, 4, 5, 6}, {2, 3}, dtype);
        const auto mat_right = Tensor::from_vector({1, 2, 3, 4, 5, 6}, {3, 2}, dtype);
        emit(name("matmul"), matmul(mat_left, mat_right));
        const auto embedding_weight = Tensor::from_vector(
            {0, 1, 2, 3, 4, 5, 6, 7}, {4, 2}, dtype);
        emit(name("embedding"), embedding(
            embedding_weight, Tensor::from_int32_vector({2, 0, 2}, {3})));
        emit(name("softmax"), softmax(left));
        emit(name("rms_norm"), rms_norm(
            left, Tensor::from_vector({1, 0.5F, 2}, {3}, dtype)));
        const auto rope_input = Tensor::from_vector(
            {1, 0, 0, 1, 1, 0, 0, 1}, {1, 2, 1, 4}, dtype);
        emit(name("rope"), rope(rope_input));
        emit(name("cross_entropy"), cross_entropy(
            Tensor::from_vector({2, 1, 0, 0, 1, 2}, {2, 3}, dtype),
            Tensor::from_int32_vector({0, 2}, {2})));
    }
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

    Value bias_input(f32({1, 2, 3, 4, 5, 6}, {2, 3}), true);
    Value bias(f32({0.5F, -1.0F, 2.0F}, {3}), true);
    const Value bias_seed(f32({1, 2, 3, -1, -2, -3}, {2, 3}));
    sum(multiply(add_bias(bias_input, bias), bias_seed)).backward();
    emit("graph_add_bias_input_grad", bias_input.grad());
    emit("graph_add_bias_bias_grad", bias.grad());

    Value mat_left(f32({1, 2, 3, 4, 5, 6}, {2, 3}), true);
    Value mat_right(f32({1, 2, 3, 4, 5, 6}, {3, 2}), true);
    const Value mat_seed(f32({1, 2, 3, 4}, {2, 2}));
    sum(multiply(matmul(mat_left, mat_right), mat_seed)).backward();
    emit("graph_matmul_left_grad", mat_left.grad());
    emit("graph_matmul_right_grad", mat_right.grad());

    Value tied_hidden(f32({1, 2, 3, 4, 5, 6}, {2, 3}), true);
    Value tied_weight(f32({1, 0, -1, 2, 1, 0, -2, 0.5F, 1, 3, -1, 2}, {4, 3}), true);
    const Value tied_seed(f32({1, -1, 0.5F, 2, -2, 1, 3, -0.5F}, {2, 4}));
    const auto tied_output = matmul(tied_hidden, tied_weight, false, true);
    emit("graph_tied_matmul_output", tied_output.data());
    sum(multiply(tied_output, tied_seed)).backward();
    emit("graph_tied_matmul_hidden_grad", tied_hidden.grad());
    emit("graph_tied_matmul_weight_grad", tied_weight.grad());

    Value fp8_left(f32({1, -2, 3, 4, 0.5F, -0.25F}, {2, 3}), true);
    Value fp8_right(f32({1, 2, 3, 4, 5, 6}, {3, 2}), true);
    const Value fp8_seed(f32({1, -1, 0.5F, 2}, {2, 2}));
    const auto fp8_output = fp8_matmul(fp8_left, fp8_right, 0.025F, 0.05F);
    emit("graph_fp8_matmul_output", fp8_output.data());
    sum(multiply(fp8_output, fp8_seed)).backward();
    emit("graph_fp8_matmul_left_grad", fp8_left.grad());
    emit("graph_fp8_matmul_right_grad", fp8_right.grad());

    Value bf16_left(f32({1, -2, 3, 4, 0.5F, -0.25F}, {2, 3}), true);
    Value bf16_right(f32({1, 2, 3, 4, 5, 6}, {3, 2}), true);
    const Value bf16_seed(f32({1, -1, 0.5F, 2}, {2, 2}));
    const auto bf16_output = bf16_matmul(bf16_left, bf16_right);
    emit("graph_bf16_matmul_output", bf16_output.data());
    sum(multiply(bf16_output, bf16_seed)).backward();
    emit("graph_bf16_matmul_left_grad", bf16_left.grad());
    emit("graph_bf16_matmul_right_grad", bf16_right.grad());

    Value fused_rope_input(f32(
        {1, 2, 3, 4, 5, 6, 7, 8,
         -1, -2, -3, -4, -5, -6, -7, -8}, {1, 2, 2, 4}), true);
    Value fused_rope_bias(f32(
        {0.1F, 0.2F, 0.3F, 0.4F, -0.1F, -0.2F, -0.3F, -0.4F}, {8}), true);
    const Value fused_rope_seed(f32(
        {1, -1, 2, -2, 3, -3, 4, -4,
         -1, 1, -2, 2, -3, 3, -4, 4}, {1, 2, 2, 4}));
    const auto fused_rope_output = rope_split_half_bias(fused_rope_input, fused_rope_bias);
    emit("graph_rope_split_half_bias_output", fused_rope_output.data());
    sum(multiply(fused_rope_output, fused_rope_seed)).backward();
    emit("graph_rope_split_half_bias_input_grad", fused_rope_input.grad());
    emit("graph_rope_split_half_bias_bias_grad", fused_rope_bias.grad());

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

    Value split_rope_input(f32({1, 2, 3, 4, 5, 6, 7, 8}, {1, 2, 1, 4}), true);
    sum(multiply(rope_split_half(split_rope_input), rope_seed)).backward();
    emit("graph_rope_split_half_input_grad", split_rope_input.grad());

    Value logits(f32({2, 1, 0, 100, -100, 0}, {2, 3}), true);
    cross_entropy(logits, Tensor::from_int32_vector({0, -100}, {2})).backward(
        f32({0.75F}, {}));
    emit("graph_cross_entropy_logits_grad", logits.grad());

    std::vector<float> large_ce_values(3 * 257);
    for (std::size_t element = 0; element < large_ce_values.size(); ++element) {
        large_ce_values[element] =
            static_cast<float>(static_cast<int>(element % 251) - 125) * 0.03125F;
    }
    Value large_ce_logits(f32(std::move(large_ce_values), {3, 257}), true);
    cross_entropy(large_ce_logits, Tensor::from_int32_vector({17, 241, -100}, {3}))
        .backward(f32({0.75F}, {}));
    emit("graph_cross_entropy_large_logits_grad", large_ce_logits.grad());

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
    emit_bool("invalid_add_bias_shape", rejected([&] { (void)add_bias(matrix, vector); }));
    emit_bool("invalid_bias_gradient_rank", rejected([&] {
                  (void)bias_gradient(f32({1}, {}));
              }));
    emit_bool("invalid_scale_dtype", rejected([&] {
                  (void)scale(Tensor::from_int32_vector({1, 2}, {2}), 2.0F);
              }));
    emit_bool("invalid_cast_dtype", rejected([&] {
                  (void)cast(Tensor::from_int32_vector({1, 2}, {2}), DType::BFloat16);
              }));
    emit_bool("invalid_matmul_inner", rejected([&] { (void)matmul(matrix, f32({1, 2}, {2, 1, 1})); }));
    emit_bool("invalid_bf16_matmul_dtype", rejected([&] {
                  (void)bf16_matmul(matrix, matrix);
              }));
    emit_bool("invalid_bf16_ffn_shape", rejected([&] {
                  const auto bf16 = matrix.cast(DType::BFloat16);
                  (void)bf16_ffn(matrix, bf16, bf16,
                                 Tensor({3, 2}, DType::BFloat16));
              }));
    emit_bool("invalid_embedding_weight", rejected([&] {
                  (void)embedding(vector, Tensor::from_int32_vector({0}, {1}));
              }));
    emit_bool("invalid_softmax_dim", rejected([&] { (void)softmax(matrix, 0); }));
    emit_bool("invalid_rms_weight", rejected([&] { (void)rms_norm(matrix, vector); }));
    emit_bool("invalid_add_rms_norm_shape", rejected([&] {
                  (void)add_rms_norm(matrix, matrix, vector);
              }));
    emit_bool("invalid_silu_dtype", rejected([&] {
                  (void)silu(Tensor::from_int32_vector({1, 2}, {2}));
              }));
    emit_bool("invalid_swiglu_shape", rejected([&] { (void)swiglu(matrix, vector); }));
    emit_bool("invalid_rope_width", rejected([&] { (void)rope(f32({1, 2, 3}, {1, 1, 3})); }));
    emit_bool("invalid_rope_split_half_width", rejected([&] {
                  (void)rope_split_half(f32({1, 2, 3}, {1, 1, 3}));
              }));
    emit_bool("invalid_rope_split_half_bias_shape", rejected([&] {
                  (void)rope_split_half_bias(f32({1, 2, 3, 4}, {1, 1, 1, 4}), vector);
              }));
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
    emit_bool("invalid_rope_split_half_backward_width", rejected([&] {
                  (void)rope_split_half_backward(f32({1, 2, 3}, {1, 1, 3}));
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
    const auto preparation = model.prepare_bf16_ffn_inference();
    if (preparation.converted_tensors != 3U) {
        throw std::logic_error("tiny model BF16 FFN preparation count changed");
    }
    emit("model_bf16_ffn_logits", model.forward_inference(tokens));
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
    emit_low_precision_forward_cases();
    emit_graph_gradient_cases();
    emit_invalid_shape_cases();
    emit_model_graph_case();
    emit_optimizer_cases();
    return 0;
}
