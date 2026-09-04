#include <iomanip>
#include <iostream>
#include <limits>
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

// emit() only accepts floating tensors; router indices are small enough
// that the exact int->float conversion loses nothing.
Tensor int32_as_float(const Tensor& indices) {
    const auto values = indices.to_int32_vector();
    return Tensor::from_vector(std::vector<float>(values.begin(), values.end()),
                               indices.shape());
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
    auto in_place_scale = f32({1, -2, 3, 4, 0.5F, -0.25F}, {2, 3});
    scale_in_place_(in_place_scale, -0.25F);
    emit("scale_in_place", in_place_scale);
    emit("cast_bf16", cast(left, DType::BFloat16));
    emit("add_bias", add_bias(left, f32({0.5F, -1.0F, 2.0F}, {3})));
    emit("add_bias_bf16", add_bias_bf16(
        left.cast(DType::BFloat16), f32({0.5F, -1.0F, 2.0F}, {3})));
    const auto residual_norm = add_rms_norm(
        left, right, f32({1, 0.5F, 2}, {3}));
    emit("add_rms_norm_sum", residual_norm.first);
    emit("add_rms_norm_normalized", residual_norm.second);

    const auto matrix_left = f32({1, 2, 3, 4, 5, 6}, {2, 3});
    const auto matrix_right = f32({1, 2, 3, 4, 5, 6}, {3, 2});
    emit("matmul_2d", matmul(matrix_left, matrix_right));
    Tensor caller_weight_gradient({3, 2});
    matmul_weight_gradient_out_(
        caller_weight_gradient, matrix_left,
        f32({1, -1, 0.5F, 2}, {2, 2}),
        MatmulImplementation::Readable);
    emit("matmul_weight_gradient_out", caller_weight_gradient);
    emit("bf16_mixed_matmul",
         bf16_matmul(matrix_left, cast(matrix_right, DType::BFloat16)));
    emit("bf16_output_matmul", bf16_matmul_output(
        cast(matrix_left, DType::BFloat16), cast(matrix_right, DType::BFloat16),
        DType::BFloat16));
    emit("bf16_weight_gradient", bf16_weight_gradient(
        matrix_left, f32({1, -1, 0.5F, 2}, {2, 2})));
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
    const auto gate_up = bf16_gate_up_projection(
        matrix_left, ffn_gate, ffn_up);
    emit("bf16_gate_up_gate", gate_up.first);
    emit("bf16_gate_up_up", gate_up.second);
    const auto qkv = bf16_qkv_projection(
        matrix_left, cast(matrix_right, DType::BFloat16),
        Tensor::from_vector({0.5F, -1.0F, 0.25F}, {3, 1}, DType::BFloat16),
        Tensor::from_vector({-0.5F, 1.25F, 0.75F}, {3, 1}, DType::BFloat16));
    emit("bf16_qkv_query", qkv.first);
    emit("bf16_qkv_key", qkv.second);
    emit("bf16_qkv_value", qkv.third);
    emit("matmul_readable",
         matmul_with_implementation(matrix_left, matrix_right,
                                    MatmulImplementation::Readable));
    emit("matmul_scaled", matmul_scaled_with_implementation(
        matrix_left, matrix_right, -0.25F,
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

    const auto moe_router_logits = f32(
        {2, 1, 0, -1, -1, 0, 2, 1, 0.1F, 0.4F, 0.3F, 0.2F}, {3, 4});
    const auto moe_router_result = moe_router_top_k(moe_router_logits, 2, true);
    emit("moe_router_top_k_indices", int32_as_float(moe_router_result.first));
    emit("moe_router_top_k_weights", moe_router_result.second);

    const auto moe_ffn_input = f32({1, 0, 0, 1}, {2, 2});
    const auto moe_ffn_indices = Tensor::from_int32_vector({1, 0}, {2, 1});
    const auto moe_gate_weight = f32({1, 0, 0, 1, 0, 1, 1, 0}, {2, 2, 2});
    const auto moe_up_weight = f32({1, 1, 1, 1, 2, 0, 0, 2}, {2, 2, 2});
    const auto moe_down_weight = f32({1, 0, 0, 1, 0, 1, 1, 0}, {2, 2, 2});
    emit("moe_expert_ffn", moe_expert_ffn(
        moe_ffn_input, moe_ffn_indices, moe_gate_weight, moe_up_weight, moe_down_weight));

    const auto moe_combine_output = f32({1, 2, 3, 4, 5, 6, 7, 8}, {2, 2, 2});
    const auto moe_combine_indices = Tensor::from_int32_vector({1, 0}, {2, 1});
    const auto moe_combine_weights = f32({0.5F, 2.0F}, {2, 1});
    emit("moe_combine", moe_combine(
        moe_combine_output, moe_combine_indices, moe_combine_weights));

    const auto rope_input = f32({1, 0, 0, 1, 1, 0, 0, 1}, {1, 2, 1, 4});
    emit("rope", rope(rope_input));
    emit("rope_split_half", rope_split_half(rope_input));
    emit("rope_split_half_bias_bthd_bf16",
         rope_split_half_bias_bthd_bf16(
             rope_input.cast(DType::BFloat16),
             f32({0.5F, -0.25F, 1, -1}, {4})));
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
    const auto attention_query = f32(
        {0.5F, -1, 1.5F, 0.25F, -0.5F, 1,
         0.75F, -0.25F, 1, 0.5F, -1, 0.25F}, {1, 2, 3, 2});
    const auto attention_key = f32(
        {0.5F, 1, -0.5F, 0.25F, 1.5F, -1}, {1, 1, 3, 2});
    const auto attention_value = f32({1, 2, 3, 4, 5, 6}, {1, 1, 3, 2});
    emit("causal_gqa_attention", causal_gqa_attention(
        attention_query, attention_key, attention_value, 2, 0.5F));
    emit("cached_gqa_attention_scores", cached_gqa_attention_scores(
        f32({1, 0, 0, 1}, {1, 2, 1, 2}),
        f32({3, 4, 1, 0, -1, 2}, {1, 1, 3, 2}), 2, 0.5F));
    const auto cached_scores = cached_gqa_attention_scores(
        f32({1, 0, 0, 1}, {1, 2, 1, 2}),
        f32({3, 4, 1, 0, -1, 2}, {1, 1, 3, 2}), 2, 0.5F);
    emit("cached_gqa_attention_context", cached_gqa_attention_context(
        softmax(cached_scores, -1),
        f32({1, 2, 3, 4, 5, 6}, {1, 1, 3, 2}), 2));
    emit("cached_gqa_attention_split_sequence",
         cached_gqa_attention_split_sequence(
             f32({1, 0, 0, 1}, {1, 2, 1, 2}),
             f32({3, 4, 1, 0, -1, 2}, {1, 1, 3, 2}),
             f32({1, 2, 3, 4, 5, 6}, {1, 1, 3, 2}),
             2, 0.5F, 2));
    emit("cached_gqa_attention_materialized_scores",
         cached_gqa_attention_materialized_scores(
             f32({1, 0, 0, 1}, {1, 2, 1, 2}),
             f32({3, 4, 1, 0, -1, 2}, {1, 1, 3, 2}),
             f32({1, 2, 3, 4, 5, 6}, {1, 1, 3, 2}),
             2, 0.5F));
    emit("cached_gqa_attention_materialized_scores_64",
         cached_gqa_attention_materialized_scores(
             f32({1, 0, 0, 1}, {1, 2, 1, 2}),
             f32({3, 4, 1, 0, -1, 2}, {1, 1, 3, 2}),
             f32({1, 2, 3, 4, 5, 6}, {1, 1, 3, 2}),
             2, 0.5F, 64));
    emit("cached_gqa_attention_materialized_scores_128",
         cached_gqa_attention_materialized_scores(
             f32({1, 0, 0, 1}, {1, 2, 1, 2}),
             f32({3, 4, 1, 0, -1, 2}, {1, 1, 3, 2}),
             f32({1, 2, 3, 4, 5, 6}, {1, 1, 3, 2}),
             2, 0.5F, 128));
    emit("cached_gqa_attention_split_pv_exact_softmax",
         cached_gqa_attention_split_pv_exact_softmax(
             f32({1, 0, 0, 1}, {1, 2, 1, 2}),
             f32({3, 4, 1, 0, -1, 2}, {1, 1, 3, 2}),
             f32({1, 2, 3, 4, 5, 6}, {1, 1, 3, 2}),
             2, 0.5F, 2));
    emit("cached_gqa_attention_gqa_value_reuse",
         cached_gqa_attention_gqa_value_reuse(
             f32({1, 0, 0, 1}, {1, 2, 1, 2}),
             f32({3, 4, 1, 0, -1, 2}, {1, 1, 3, 2}),
             f32({1, 2, 3, 4, 5, 6}, {1, 1, 3, 2}),
             2, 0.5F, 8));
    emit("online_causal_gqa_attention_bthd",
         online_causal_gqa_attention_bthd(
             attention_query.cast(DType::BFloat16),
             attention_key.cast(DType::BFloat16),
             attention_value.transpose(1, 2).contiguous().cast(
                 DType::BFloat16),
             2, 0.5F));
    emit("attention_probability_value_bthd",
         attention_probability_value_bthd(
             f32({1, 0, 0.25F, 0.75F,
                  1, 0, 0.5F, 0.5F}, {1, 2, 2, 2}),
             f32({1, 2, 10, 20, 3, 4, 30, 40}, {1, 2, 2, 2})));
    emit("attention_probability_value_gqa_bthd",
         attention_probability_value_gqa_bthd(
             f32({1, 0, 0.25F, 0.75F, 1, 0, 0.5F, 0.5F,
                  1, 0, 0.75F, 0.25F, 1, 0, 0.1F, 0.9F}, {1, 4, 2, 2}),
             f32({1, 2, 10, 20, 3, 4, 30, 40}, {1, 2, 2, 2}), 2));
    emit("attention_probability_gradient_gqa_bthd",
         attention_probability_gradient_gqa_bthd(
             f32({1, -1, 2, -2, 3, -3, 4, -4,
                  5, -5, 6, -6, 7, -7, 8, -8}, {1, 2, 4, 2}),
             f32({1, 2, 10, 20, 3, 4, 30, 40}, {1, 2, 2, 2}), 2));
    emit("repeat_interleave", repeat_interleave(f32({1, 2, 3, 4}, {2, 2}), 0, 2));
    const auto paired_repeat = repeat_gqa_kv_bthd(
        f32({1, 2, 3, 4, 10, 20, 30, 40}, {1, 2, 2, 2}),
        f32({5, 6, 50, 60, 7, 8, 70, 80}, {1, 2, 2, 2}), 2);
    emit("repeat_gqa_kv_bthd_key", paired_repeat.first);
    emit("repeat_gqa_kv_bthd_value", paired_repeat.second);
    const auto paired_repeat_backward = repeat_gqa_kv_bthd_backward(
        f32({1, 2, 3, 4, 5, 6, 7, 8,
             9, 10, 11, 12, 13, 14, 15, 16}, {1, 4, 2, 2}),
        f32({16, 15, 14, 13, 12, 11, 10, 9,
             8, 7, 6, 5, 4, 3, 2, 1}, {1, 2, 4, 2}), 2);
    emit("repeat_gqa_kv_bthd_backward_key", paired_repeat_backward.first);
    emit("repeat_gqa_kv_bthd_backward_value", paired_repeat_backward.second);
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

    Value shared_input(f32({1, -2, 3, 0.5F, 2, -1}, {2, 3}), true);
    Value shared_gate(f32({1, 0.5F, -1, 2, 0.25F, -0.75F}, {3, 2}), true);
    Value shared_up(f32({-0.5F, 1.5F, 2, -1, 0.75F, 0.25F}, {3, 2}), true);
    const auto shared_gate_mirror = shared_gate.data().cast(DType::BFloat16);
    const auto shared_up_mirror = shared_up.data().cast(DType::BFloat16);
    const auto shared_pair = bf16_gate_up_projection(
        shared_input, shared_gate, shared_gate_mirror,
        shared_up, shared_up_mirror);
    emit("graph_bf16_gate_up_gate", shared_pair.first.data());
    emit("graph_bf16_gate_up_up", shared_pair.second.data());
    const Value shared_gate_seed(f32({1, -2, 0.5F, 3}, {2, 2}));
    const Value shared_up_seed(f32({-1, 0.25F, 2, -0.5F}, {2, 2}));
    add(sum(multiply(shared_pair.first, shared_gate_seed)),
        sum(multiply(shared_pair.second, shared_up_seed))).backward();
    emit("graph_bf16_gate_up_input_grad", shared_input.grad());
    emit("graph_bf16_gate_up_gate_grad", shared_gate.grad());
    emit("graph_bf16_gate_up_up_grad", shared_up.grad());

    Value shared_query(f32({1, 0.5F, -1, 2, 0.25F, -0.75F}, {3, 2}), true);
    Value shared_key(f32({1, -0.5F, 2}, {3, 1}), true);
    Value shared_value(f32({-1, 0.25F, 0.5F}, {3, 1}), true);
    Value shared_qkv_input(f32({1, -2, 3, 0.5F, 2, -1}, {2, 3}), true);
    const auto shared_qkv = bf16_qkv_projection(
        shared_qkv_input, shared_query,
        shared_query.data().cast(DType::BFloat16), shared_key,
        shared_key.data().cast(DType::BFloat16), shared_value,
        shared_value.data().cast(DType::BFloat16));
    emit("graph_bf16_qkv_query", shared_qkv.first.data());
    emit("graph_bf16_qkv_key", shared_qkv.second.data());
    emit("graph_bf16_qkv_value", shared_qkv.third.data());
    const Value shared_key_seed(f32({1.5F, -2}, {2, 1}));
    const Value shared_value_seed(f32({-0.25F, 3}, {2, 1}));
    add(add(sum(multiply(shared_qkv.first, shared_gate_seed)),
            sum(multiply(shared_qkv.second, shared_key_seed))),
        sum(multiply(shared_qkv.third, shared_value_seed))).backward();
    emit("graph_bf16_qkv_input_grad", shared_qkv_input.grad());
    emit("graph_bf16_qkv_query_grad", shared_query.grad());
    emit("graph_bf16_qkv_key_grad", shared_key.grad());
    emit("graph_bf16_qkv_value_grad", shared_value.grad());

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

    Value layout_rope_input(f32(
        {1, 2, 3, 4, 5, 6, 7, 8,
         -1, -2, -3, -4, -5, -6, -7, -8}, {1, 2, 2, 4}), true);
    Value layout_rope_bias(f32(
        {0.1F, 0.2F, 0.3F, 0.4F, -0.1F, -0.2F, -0.3F, -0.4F}, {8}), true);
    const Value layout_rope_seed(f32(
        {1, -1, 2, -2, 3, -3, 4, -4,
         -1, 1, -2, 2, -3, 3, -4, 4}, {1, 2, 2, 4}));
    const auto layout_rope_output =
        rope_split_half_bias_bthd(layout_rope_input, layout_rope_bias);
    emit("graph_rope_split_half_bias_bthd_output", layout_rope_output.data());
    sum(multiply(layout_rope_output, layout_rope_seed)).backward();
    emit("graph_rope_split_half_bias_bthd_input_grad", layout_rope_input.grad());
    emit("graph_rope_split_half_bias_bthd_bias_grad", layout_rope_bias.grad());

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

    Value fused_norm_left(f32({1, 2, 3, -1, -2, -3}, {2, 3}), true);
    Value fused_norm_right(f32({0.5F, -0.5F, 1, 2, 1, 0}, {2, 3}), true);
    Value fused_norm_weight(f32({1, 0.5F, 2}, {3}), true);
    const Value fused_norm_sum_seed(f32(
        {1, -1, 2, -2, 3, -3}, {2, 3}));
    const Value fused_norm_seed(f32(
        {0.5F, 2, -1, 1.5F, -0.5F, 3}, {2, 3}));
    const auto fused_norm = add_rms_norm(
        fused_norm_left, fused_norm_right, fused_norm_weight);
    emit("graph_add_rms_norm_sum", fused_norm.first.data());
    emit("graph_add_rms_norm_normalized", fused_norm.second.data());
    add(sum(multiply(fused_norm.first, fused_norm_sum_seed)),
        sum(multiply(fused_norm.second, fused_norm_seed))).backward();
    emit("graph_add_rms_norm_left_grad", fused_norm_left.grad());
    emit("graph_add_rms_norm_right_grad", fused_norm_right.grad());
    emit("graph_add_rms_norm_weight_grad", fused_norm_weight.grad());

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

    Value attention_query(f32(
        {0.5F, -1, 1.5F, 0.25F, -0.5F, 1,
         0.75F, -0.25F, 1, 0.5F, -1, 0.25F}, {1, 2, 3, 2}), true);
    Value attention_key(f32(
        {0.5F, 1, -0.5F, 0.25F, 1.5F, -1}, {1, 1, 3, 2}), true);
    Value attention_value(f32({1, 2, 3, 4, 5, 6}, {1, 1, 3, 2}), true);
    const Value attention_seed(f32(
        {1, -1, 0.5F, 2, -0.5F, 1.5F,
         2, 1, -1, 0.25F, 0.75F, -2}, {1, 2, 3, 2}));
    const auto attention_output = causal_gqa_attention(
        attention_query, attention_key, attention_value, 2, 0.5F);
    emit("graph_causal_gqa_output", attention_output.data());
    sum(multiply(attention_output, attention_seed)).backward();
    emit("graph_causal_gqa_query_grad", attention_query.grad());
    emit("graph_causal_gqa_key_grad", attention_key.grad());
    emit("graph_causal_gqa_value_grad", attention_value.grad());

    const auto bthd_probabilities = f32(
        {1, 0, 0, 0.25F, 0.75F, 0, 0.1F, 0.2F, 0.7F,
         1, 0, 0, 0.5F, 0.5F, 0, 0.2F, 0.3F, 0.5F}, {1, 2, 3, 3});
    const auto bthd_expanded_value = f32(
        {1, 2, 10, 20, 3, 4, 30, 40, 5, 6, 50, 60}, {1, 3, 2, 2});
    const auto bthd_seed = f32(
        {1, -1, 2, 1, 0.5F, 2, -0.5F, 0.25F,
         -0.5F, 1.5F, 0.75F, -2}, {1, 3, 2, 2});
    emit("graph_causal_gqa_bthd_probability_grad",
         microllm::ops::attention_probability_gradient_bthd(
             bthd_seed, bthd_expanded_value));
    emit("graph_causal_gqa_bthd_expanded_value_grad",
         microllm::ops::attention_value_gradient_bthd(
             bthd_probabilities, bthd_seed));

    Value bthd_query(f32(
        {0.5F, -1, 1.5F, 0.25F, -0.5F, 1,
         0.75F, -0.25F, 1, 0.5F, -1, 0.25F}, {1, 2, 3, 2}), true);
    Value bthd_key(f32(
        {0.5F, 1, -0.5F, 0.25F, 1.5F, -1}, {1, 1, 3, 2}), true);
    Value bthd_value(f32({1, 2, 3, 4, 5, 6}, {1, 3, 1, 2}), true);
    const Value bthd_graph_seed(bthd_seed);
    const auto bthd_output = causal_gqa_attention_bthd(
        bthd_query, bthd_key, bthd_value, 2, 0.5F);
    emit("graph_causal_gqa_bthd_output", bthd_output.data());
    sum(multiply(bthd_output, bthd_graph_seed)).backward();
    emit("graph_causal_gqa_bthd_query_grad", bthd_query.grad());
    emit("graph_causal_gqa_bthd_key_grad", bthd_key.grad());
    emit("graph_causal_gqa_bthd_value_grad", bthd_value.grad());

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
    emit_bool("invalid_add_bias_bf16_shape", rejected([&] {
                  (void)add_bias_bf16(
                      matrix.cast(DType::BFloat16), f32({1}, {1}));
              }));
    emit_bool("invalid_bias_gradient_rank", rejected([&] {
                  (void)bias_gradient(f32({1}, {}));
              }));
    emit_bool("invalid_scale_dtype", rejected([&] {
                  (void)scale(Tensor::from_int32_vector({1, 2}, {2}), 2.0F);
              }));
    emit_bool("invalid_scale_in_place_factor", rejected([&] {
                  auto value = matrix;
                  scale_in_place_(value, std::numeric_limits<float>::infinity());
              }));
    emit_bool("invalid_cast_dtype", rejected([&] {
                  (void)cast(Tensor::from_int32_vector({1, 2}, {2}), DType::BFloat16);
              }));
    emit_bool("invalid_matmul_inner", rejected([&] { (void)matmul(matrix, f32({1, 2}, {2, 1, 1})); }));
    emit_bool("invalid_matmul_scaled_factor", rejected([&] {
                  (void)matmul_scaled_with_implementation(
                      matrix, matrix,
                      std::numeric_limits<float>::infinity(),
                      MatmulImplementation::Readable);
              }));
    emit_bool("invalid_bf16_matmul_dtype", rejected([&] {
                  (void)bf16_matmul(matrix, matrix);
              }));
    emit_bool("invalid_bf16_weight_gradient_shape", rejected([&] {
                  (void)bf16_weight_gradient(matrix, f32({1, 2, 3}, {3, 1}));
              }));
    emit_bool("invalid_bf16_ffn_shape", rejected([&] {
                  const auto bf16 = matrix.cast(DType::BFloat16);
                  (void)bf16_ffn(matrix, bf16, bf16,
                                 Tensor({3, 2}, DType::BFloat16));
              }));
    emit_bool("invalid_bf16_qkv_shape", rejected([&] {
                  const auto bf16 = matrix.cast(DType::BFloat16);
                  (void)bf16_qkv_projection(matrix, bf16, bf16,
                                             Tensor({3, 1}, DType::BFloat16));
              }));
    emit_bool("invalid_bf16_gate_up_shape", rejected([&] {
                  const auto bf16 = matrix.cast(DType::BFloat16);
                  (void)bf16_gate_up_projection(
                      matrix, bf16,
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
    emit_bool("invalid_moe_router_top_k_shape", rejected([&] {
                  (void)moe_router_top_k(f32({1, 2, 3}, {3}), 2, true);
              }));
    emit_bool("invalid_moe_expert_ffn_shape", rejected([&] {
                  (void)moe_expert_ffn(
                      f32({1, 0, 0, 1}, {2, 2}), Tensor::from_int32_vector({1, 0}, {2, 1}),
                      f32({1, 2, 3, 4}, {2, 2}),
                      f32({1, 1, 1, 1, 2, 0, 0, 2}, {2, 2, 2}),
                      f32({1, 0, 0, 1, 0, 1, 1, 0}, {2, 2, 2}));
              }));
    emit_bool("invalid_moe_combine_shape", rejected([&] {
                  (void)moe_combine(
                      f32({1, 2, 3, 4, 5, 6, 7, 8}, {2, 2, 2}),
                      Tensor::from_int32_vector({1, 0}, {2, 1}),
                      f32({0.5F, 2.0F, 1.0F}, {3}));
              }));
    emit_bool("invalid_rope_width", rejected([&] { (void)rope(f32({1, 2, 3}, {1, 1, 3})); }));
    emit_bool("invalid_rope_split_half_width", rejected([&] {
                  (void)rope_split_half(f32({1, 2, 3}, {1, 1, 3}));
              }));
    emit_bool("invalid_rope_split_half_bias_shape", rejected([&] {
                  (void)rope_split_half_bias(f32({1, 2, 3, 4}, {1, 1, 1, 4}), vector);
              }));
    emit_bool("invalid_rope_split_half_bias_bthd_shape", rejected([&] {
                  (void)rope_split_half_bias_bthd(
                      f32({1, 2, 3, 4}, {1, 1, 1, 4}), vector);
              }));
    emit_bool("invalid_rope_split_half_bias_bthd_bf16_shape", rejected([&] {
                  (void)rope_split_half_bias_bthd_bf16(
                      Tensor({1, 2, 1, 3}, DType::BFloat16),
                      f32({1, 2, 3}, {3}));
              }));
    emit_bool("invalid_cross_entropy_shape", rejected([&] {
                  (void)cross_entropy(matrix, Tensor::from_int32_vector({0}, {1}));
              }));
    emit_bool("invalid_reduce_dtype", rejected([&] {
                  (void)reduce_sum(Tensor::from_int32_vector({1, 2}, {2}));
              }));
    emit_bool("invalid_broadcast_source", rejected([&] { (void)broadcast_scalar(vector, {2}); }));
    emit_bool("invalid_causal_shape", rejected([&] { (void)causal_softmax(f32({1, 2, 3, 4, 5, 6}, {2, 3})); }));
    emit_bool("invalid_causal_gqa_shape", rejected([&] {
                  const auto query = Tensor({1, 2, 3, 2});
                  const auto key = Tensor({1, 1, 3, 2});
                  (void)causal_gqa_attention(query, key, key, 3, 0.5F);
              }));
    emit_bool("invalid_cached_gqa_scores_shape", rejected([&] {
                  (void)cached_gqa_attention_scores(
                      Tensor({1, 2, 1, 2}), Tensor({1, 1, 3, 2}),
                      3, 0.5F);
              }));
    emit_bool("invalid_cached_gqa_context_shape", rejected([&] {
                  (void)cached_gqa_attention_context(
                      Tensor({1, 2, 1, 3}), Tensor({1, 1, 3, 2}), 3);
              }));
    emit_bool("invalid_cached_gqa_split_count", rejected([&] {
                  (void)cached_gqa_attention_split_sequence(
                      Tensor({1, 2, 1, 2}), Tensor({1, 1, 3, 2}),
                      Tensor({1, 1, 3, 2}), 2, 0.5F, 4);
              }));
    emit_bool("invalid_cached_gqa_materialized_shape", rejected([&] {
                  (void)cached_gqa_attention_materialized_scores(
                      Tensor({1, 2, 1, 2}), Tensor({1, 1, 3, 2}),
                      Tensor({1, 1, 4, 2}), 2, 0.5F);
              }));
    emit_bool("invalid_cached_gqa_materialized_threads", rejected([&] {
                  (void)cached_gqa_attention_materialized_scores(
                      Tensor({1, 2, 1, 2}), Tensor({1, 1, 3, 2}),
                      Tensor({1, 1, 3, 2}), 2, 0.5F, 32);
              }));
    emit_bool("invalid_cached_gqa_split_pv_count", rejected([&] {
                  (void)cached_gqa_attention_split_pv_exact_softmax(
                      Tensor({1, 2, 1, 2}), Tensor({1, 1, 3, 2}),
                      Tensor({1, 1, 3, 2}), 2, 0.5F, 4);
              }));
    emit_bool("invalid_cached_gqa_value_reuse_tile", rejected([&] {
                  (void)cached_gqa_attention_gqa_value_reuse(
                      Tensor({1, 2, 1, 2}), Tensor({1, 1, 3, 2}),
                      Tensor({1, 1, 3, 2}), 2, 0.5F, 7);
              }));
    emit_bool("invalid_attention_probability_value_bthd_shape", rejected([&] {
                  (void)attention_probability_value_bthd(
                      Tensor({1, 2, 3, 3}), Tensor({1, 3, 1, 2}));
              }));
    emit_bool("invalid_attention_probability_value_gqa_bthd_shape", rejected([&] {
                  (void)attention_probability_value_gqa_bthd(
                      Tensor({1, 4, 2, 2}), Tensor({1, 2, 1, 2}), 2);
              }));
    emit_bool("invalid_attention_probability_gradient_bthd_shape", rejected([&] {
                  (void)attention_probability_gradient_bthd(
                      Tensor({1, 3, 2, 2}), Tensor({1, 3, 1, 2}));
              }));
    emit_bool("invalid_attention_probability_gradient_gqa_bthd_shape", rejected([&] {
                  (void)attention_probability_gradient_gqa_bthd(
                      Tensor({1, 2, 4, 2}), Tensor({1, 2, 1, 2}), 2);
              }));
    emit_bool("invalid_attention_value_gradient_bthd_shape", rejected([&] {
                  (void)attention_value_gradient_bthd(
                      Tensor({1, 2, 3, 3}), Tensor({1, 3, 1, 2}));
              }));
    emit_bool("invalid_causal_gqa_bthd_shape", rejected([&] {
                  (void)causal_gqa_attention_bthd(
                      Tensor({1, 2, 3, 2}), Tensor({1, 1, 3, 2}),
                      Tensor({1, 3, 2, 2}), 2, 0.5F);
              }));
    emit_bool("invalid_online_causal_gqa_bthd_shape", rejected([&] {
                  (void)online_causal_gqa_attention_bthd(
                      Tensor({1, 2, 3, 2}, DType::BFloat16),
                      Tensor({1, 1, 3, 2}, DType::BFloat16),
                      Tensor({1, 3, 2, 2}, DType::BFloat16),
                      2, 0.5F);
              }));
    emit_bool("invalid_repeat_count", rejected([&] { (void)repeat_interleave(matrix, 0, 0); }));
    emit_bool("invalid_repeat_gqa_kv_bthd_shape", rejected([&] {
                  (void)repeat_gqa_kv_bthd(
                      Tensor({1, 2, 3, 2}), Tensor({1, 3, 1, 2}), 2);
              }));

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
    emit_bool("invalid_rope_split_half_bias_bthd_backward_width", rejected([&] {
                  (void)rope_split_half_bias_bthd_backward(
                      f32({1, 1, 1, 3}, {1, 1, 1, 3}));
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
    emit_bool("invalid_repeat_gqa_kv_bthd_backward_shape", rejected([&] {
                  (void)repeat_gqa_kv_bthd_backward(
                      Tensor({1, 4, 2, 2}), Tensor({1, 2, 3, 2}), 2);
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
    const auto attention_preparation = model.prepare_bf16_attention_inference();
    if (attention_preparation.converted_tensors != 4U) {
        throw std::logic_error("tiny model BF16 Attention preparation count changed");
    }
    emit("model_bf16_attention_ffn_logits", model.forward_inference(tokens));

    auto bf16_training_config = config;
    bf16_training_config.linear_precision =
        microllm::model::LinearPrecision::BFloat16;
    microllm::model::TransformerModel bf16_training_model(bf16_training_config, 211);
    for (const auto& [name, parameter] : bf16_training_model.named_parameters()) {
        emit("model_bf16_train_param:" + name, parameter->data());
    }
    const auto bf16_training_logits = bf16_training_model.forward(tokens);
    emit("model_bf16_train_logits", bf16_training_logits.data());
    const auto bf16_training_loss = microllm::autograd::cross_entropy(
        bf16_training_logits, targets);
    emit("model_bf16_train_loss", bf16_training_loss.data());
    bf16_training_loss.backward();
    for (const auto& [name, parameter] : bf16_training_model.named_parameters()) {
        emit("model_bf16_train_grad:" + name, parameter->grad());
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

    adam_parameter.set_grad(f32({0.25F, -0.5F}, {2}));
    adam.step();
    emit("optimizer_adamw_parameter_step3", adam_parameter.data());
    state = adam.state();
    emit("optimizer_adamw_first_moment_step3", state.first_moments[0]);
    emit("optimizer_adamw_second_moment_step3", state.second_moments[0]);

    Tensor bf16_moment_parameter = f32({1.0F, -2.0F}, {2});
    Tensor bf16_first({2}, DType::BFloat16);
    Tensor bf16_second({2}, DType::BFloat16);
    Tensor bf16_mirror({2}, DType::BFloat16);
    microllm::ops::fill_(bf16_first, 0.0F);
    microllm::ops::fill_(bf16_second, 0.0F);
    for (const auto& [step, gradient] :
         std::vector<std::pair<int, Tensor>>{
             {1, f32({0.5F, -0.25F}, {2})},
             {2, f32({-1.0F, 2.0F}, {2})}}) {
        microllm::ops::adamw_update_bf16_moments_(
            bf16_moment_parameter, gradient, bf16_first, bf16_second,
            &bf16_mirror, 0.01F, 0.9F, 0.99F, 1.0e-8F, 0.1F,
            1.0F - std::pow(0.9F, static_cast<float>(step)),
            1.0F - std::pow(0.99F, static_cast<float>(step)));
    }
    emit("optimizer_bf16_moment_parameter_step2", bf16_moment_parameter);
    emit("optimizer_bf16_moment_first_step2", bf16_first);
    emit("optimizer_bf16_moment_second_step2", bf16_second);
    emit("optimizer_bf16_moment_mirror_step2", bf16_mirror);
    for (int step = 3; step <= 32; ++step) {
        const auto first_gradient =
            static_cast<float>(step % 7 - 3) * 0.125F;
        const auto second_gradient =
            static_cast<float>(step % 5 - 2) * -0.25F;
        const auto gradient = f32({first_gradient, second_gradient}, {2});
        microllm::ops::adamw_update_bf16_moments_(
            bf16_moment_parameter, gradient, bf16_first, bf16_second,
            &bf16_mirror, 0.01F, 0.9F, 0.99F, 1.0e-8F, 0.1F,
            1.0F - std::pow(0.9F, static_cast<float>(step)),
            1.0F - std::pow(0.99F, static_cast<float>(step)));
    }
    emit("optimizer_bf16_moment_parameter_step32", bf16_moment_parameter);
    emit("optimizer_bf16_moment_first_step32", bf16_first);
    emit("optimizer_bf16_moment_second_step32", bf16_second);
    emit("optimizer_bf16_moment_mirror_step32", bf16_mirror);
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
