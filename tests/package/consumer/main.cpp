#include <iostream>
#include <stdexcept>
#include <vector>

#include <microllm/base/device.h>
#include <microllm/model/config.h>
#include <microllm/model/model.h>
#include <microllm/ops/tuning.h>
#include <microllm/autograd/diagnostics.h>
#include <microllm/runtime/diagnostics.h>

int main() {
    const auto device = microllm::Device::cpu();
    microllm::runtime::HipGraphExecutable empty_graph;
    bool rejected_cpu_deferred_release = false;
    bool rejected_cpu_scoped_stream = false;
    bool rejected_cpu_stream_ordered = false;
    bool rejected_cpu_activation_arena = false;
    try {
        const microllm::runtime::Stream cpu_stream(device);
        microllm::runtime::DeferredHipDeallocationScope scope(cpu_stream);
    } catch (const std::invalid_argument&) {
        rejected_cpu_deferred_release = true;
    }
    try {
        const microllm::runtime::Stream cpu_stream(device);
        microllm::runtime::ScopedDeferredHipStream scope(cpu_stream);
    } catch (const std::invalid_argument&) {
        rejected_cpu_scoped_stream = true;
    }
    try {
        const microllm::runtime::Stream cpu_stream(device);
        microllm::runtime::StreamOrderedHipBuffer buffer(cpu_stream, 16);
    } catch (const std::invalid_argument&) {
        rejected_cpu_stream_ordered = true;
    }
    try {
        const microllm::runtime::Stream cpu_stream(device);
        microllm::runtime::HipActivationArena arena(cpu_stream, 1024);
    } catch (const std::invalid_argument&) {
        rejected_cpu_activation_arena = true;
    }
    const auto config = microllm::model::ModelConfig::model_s();
    auto prewarm_config = config;
    prewarm_config.vocabulary_size = 8;
    prewarm_config.dimension = 8;
    prewarm_config.layers = 1;
    prewarm_config.heads = 2;
    prewarm_config.kv_heads = 1;
    prewarm_config.ffn_dimension = 16;
    prewarm_config.max_sequence_length = 8;
    microllm::model::TransformerModel prewarm_model(prewarm_config, 1);
    bool rejected_cpu_grouped_prewarm = false;
    try {
        (void)prewarm_model.prewarm_bf16_grouped_qkv(1);
    } catch (const std::logic_error&) {
        rejected_cpu_grouped_prewarm = true;
    }
    const auto bias_input = microllm::Tensor::from_vector(
        {1.0F, 2.0F, 3.0F, 4.0F}, {2, 2});
    std::vector<float> external_values{0.0F, 0.0F};
    const auto external_storage = microllm::Storage::from_external(
        external_values.data(), external_values.size() * sizeof(float), device);
    const auto external_gate = microllm::Tensor::from_storage(
        external_storage, {2}, {1}, 0, microllm::DType::Float32);
    const auto external_up = microllm::Tensor::from_vector({2.0F, 2.0F}, {2});
    microllm::Tensor external_swiglu({2});
    microllm::ops::swiglu_out_(external_swiglu, external_gate, external_up);
    const auto bf16_input = microllm::Tensor::from_vector(
        {0.0F, 0.0F}, {1, 2});
    const auto bf16_gate_weight = microllm::Tensor::from_vector(
        {0.0F, 0.0F, 0.0F, 0.0F}, {2, 2},
        microllm::DType::BFloat16);
    const auto bf16_up_weight = microllm::Tensor::from_vector(
        {0.0F, 0.0F, 0.0F, 0.0F}, {2, 2},
        microllm::DType::BFloat16);
    const auto bf16_down_weight = microllm::Tensor::from_vector(
        {0.0F, 0.0F, 0.0F, 0.0F}, {2, 2},
        microllm::DType::BFloat16);
    microllm::ops::Bf16FfnWorkspace bf16_workspace{
        .input_bf16 = microllm::Tensor({1, 2}, microllm::DType::BFloat16),
        .gate = microllm::Tensor({1, 2}, microllm::DType::BFloat16),
        .up = microllm::Tensor({1, 2}, microllm::DType::BFloat16),
        .activated = microllm::Tensor({1, 2}, microllm::DType::BFloat16),
        .output_fallback_bf16 =
            microllm::Tensor({1, 2}, microllm::DType::BFloat16)};
    microllm::Tensor bf16_output({1, 2});
    microllm::ops::bf16_ffn_out_(
        bf16_output, bf16_workspace, bf16_input, bf16_gate_weight,
        bf16_up_weight, bf16_down_weight);
    const auto bias_result = microllm::ops::bias_gradient_with_implementation(
        bias_input, microllm::ops::BiasGradientImplementation::ScalarColumns);
    auto embedding_gradient = microllm::Tensor::from_vector(
        {1.0F, 2.0F, 3.0F, 4.0F}, {2, 2});
    const auto token_gradient = microllm::Tensor::from_vector(
        {0.5F, -0.5F}, {1, 2});
    const auto token_index = microllm::Tensor::from_int32_vector({1}, {1});
    microllm::ops::embedding_backward_add_(
        embedding_gradient, token_gradient, token_index);
    const auto layout_input = microllm::Tensor::from_vector(
        {1.0F, 2.0F, 3.0F, 4.0F}, {1, 1, 1, 4});
    const auto layout_bias = microllm::Tensor::from_vector(
        {0.5F, 0.5F, 0.5F, 0.5F}, {4});
    const auto layout_output = microllm::ops::rope_split_half_bias_bthd(
        layout_input, layout_bias);
    const auto layout_gradient =
        microllm::ops::rope_split_half_bias_bthd_backward(
            microllm::Tensor::from_vector(
                {1.0F, 1.0F, 1.0F, 1.0F}, {1, 1, 1, 4}));
    const auto layout_context =
        microllm::ops::attention_probability_value_bthd(
            microllm::Tensor::from_vector({1.0F}, {1, 1, 1, 1}),
            microllm::Tensor::from_vector({2.0F, 3.0F}, {1, 1, 1, 2}));
    const auto gqa_layout_context = microllm::ops::causal_gqa_attention_bthd(
        microllm::Tensor::from_vector({1.0F, 0.0F}, {1, 1, 1, 2}),
        microllm::Tensor::from_vector({1.0F, 0.0F}, {1, 1, 1, 2}),
        microllm::Tensor::from_vector({2.0F, 3.0F}, {1, 1, 1, 2}),
        1, 1.0F);
    const auto scaled_product = microllm::ops::matmul_scaled_with_implementation(
        microllm::Tensor::from_vector({1.0F, 2.0F}, {1, 2}),
        microllm::Tensor::from_vector({3.0F, 4.0F}, {2, 1}),
        0.5F, microllm::ops::MatmulImplementation::Readable);
    auto inplace_sum = microllm::Tensor::from_vector({1.0F, 2.0F}, {2});
    microllm::ops::add_in_place_(
        inplace_sum, microllm::Tensor::from_vector({3.0F, 4.0F}, {2}));
    microllm::Tensor caller_matmul({1, 1});
    microllm::ops::matmul_out_(
        caller_matmul,
        microllm::Tensor::from_vector({1.0F, 2.0F}, {1, 2}),
        microllm::Tensor::from_vector({3.0F, 4.0F}, {2, 1}),
        microllm::ops::MatmulImplementation::Readable);
    const auto paired_repeat = microllm::ops::repeat_gqa_kv_bthd(
        microllm::Tensor::from_vector({1.0F, 2.0F}, {1, 1, 1, 2}),
        microllm::Tensor::from_vector({3.0F, 4.0F}, {1, 1, 1, 2}), 1);
    const auto fused_bf16_repeat =
        microllm::ops::repeat_interleave_bf16_to_float(
            microllm::Tensor::from_vector(
                {1.0F, 2.0F}, {1, 1, 1, 2},
                microllm::DType::BFloat16),
            2, 2);
    const auto broadcast_context =
        microllm::ops::attention_probability_value_gqa_bthd(
            microllm::Tensor::from_vector({1.0F}, {1, 1, 1, 1}),
            microllm::Tensor::from_vector({2.0F, 3.0F}, {1, 1, 1, 2}), 1);
    const auto broadcast_probability_gradient =
        microllm::ops::attention_probability_gradient_gqa_bthd(
            microllm::Tensor::from_vector({1.0F, 1.0F}, {1, 1, 1, 2}),
            microllm::Tensor::from_vector({2.0F, 3.0F}, {1, 1, 1, 2}), 1);
    microllm::autograd::enable_gradient_accumulation_diagnostics(false);
    microllm::autograd::enable_unique_gradient_inplace_add(true);
    microllm::autograd::enable_unique_gradient_inplace_add(false);
    microllm::runtime::enable_strided_copy_diagnostics(false);
    microllm::autograd::enable_attention_rope_layout_fusion(false);
    microllm::autograd::enable_attention_rope_layout_fusion(true);
    microllm::autograd::enable_attention_context_layout_fusion(false);
    microllm::autograd::enable_attention_context_layout_fusion(true);
    microllm::ops::clear_attention_layout_plan_cache();
    microllm::ops::enable_inference_bthd_attention(true);
    microllm::ops::enable_inference_bthd_attention(false);
    microllm::ops::enable_inference_bthd_bf16_qk(true);
    microllm::ops::enable_inference_bthd_bf16_qk(false);
    const auto attention_plan_stats =
        microllm::ops::attention_layout_plan_cache_stats();
    const auto fp32_solution_key =
        microllm::ops::make_fp32_matmul_solution_key(
            {1, 2, 4, 8}, {1, 2, 4, 8}, device, false, true);
    microllm::ops::clear_fp32_matmul_solution_registry();
    const auto fp32_solution_stats =
        microllm::ops::fp32_matmul_solution_stats();
    const auto grouped_qkv_key = microllm::ops::make_bf16_grouped_qkv_key(
        512, 896, 896, 128, 128, device);
    microllm::ops::clear_bf16_grouped_qkv_registry();
    const auto grouped_qkv_stats = microllm::ops::bf16_grouped_qkv_stats();
    const auto grouped_gate_up_key =
        microllm::ops::make_bf16_grouped_gate_up_key(
            512, 896, 4864, device);
    microllm::ops::clear_bf16_grouped_gate_up_registry();
    const auto grouped_gate_up_stats =
        microllm::ops::bf16_grouped_gate_up_stats();
    bool rejected_cpu_tuning = false;
    bool rejected_cpu_adamw_tuning = false;
    bool rejected_cpu_softmax_candidate = false;
    try {
        const microllm::Tensor left({2, 2});
        const microllm::Tensor right({2, 2});
        (void)microllm::ops::autotune_matmul(left, right);
    } catch (const std::invalid_argument&) {
        rejected_cpu_tuning = true;
    }
    try {
        const microllm::Tensor parameter({4});
        const microllm::Tensor gradient({4});
        const microllm::Tensor first({4});
        const microllm::Tensor second({4});
        (void)microllm::ops::autotune_adamw(
            parameter, gradient, first, second);
    } catch (const std::invalid_argument&) {
        rejected_cpu_adamw_tuning = true;
    }
    try {
        (void)microllm::ops::causal_softmax_with_implementation(
            microllm::Tensor({1, 256, 256}),
            microllm::ops::CausalSoftmaxImplementation::Rows128);
    } catch (const std::invalid_argument&) {
        rejected_cpu_softmax_candidate = true;
    }
    if (!device.is_cpu() || empty_graph.defined() || !rejected_cpu_deferred_release ||
        !rejected_cpu_scoped_stream ||
        !rejected_cpu_stream_ordered ||
        !rejected_cpu_activation_arena ||
        microllm::runtime::stream_ordered_allocator_supported(device) ||
        config.parameter_count() == 0 ||
        external_swiglu.to_vector() != std::vector<float>({0.0F, 0.0F}) ||
        bf16_output.to_vector() != std::vector<float>({0.0F, 0.0F}) ||
        bias_result.to_vector() != std::vector<float>({4.0F, 6.0F}) ||
        embedding_gradient.to_vector() !=
            std::vector<float>({1.0F, 2.0F, 3.5F, 3.5F}) ||
        layout_output.to_vector() !=
            std::vector<float>({1.5F, 2.5F, 3.5F, 4.5F}) ||
        layout_gradient.to_vector() !=
            std::vector<float>({1.0F, 1.0F, 1.0F, 1.0F}) ||
        layout_context.to_vector() != std::vector<float>({2.0F, 3.0F}) ||
        gqa_layout_context.to_vector() != std::vector<float>({2.0F, 3.0F}) ||
        scaled_product.to_vector() != std::vector<float>({5.5F}) ||
        inplace_sum.to_vector() != std::vector<float>({4.0F, 6.0F}) ||
        caller_matmul.to_vector() != std::vector<float>({11.0F}) ||
        paired_repeat.first.to_vector() != std::vector<float>({1.0F, 2.0F}) ||
        paired_repeat.second.to_vector() != std::vector<float>({3.0F, 4.0F}) ||
        fused_bf16_repeat.to_vector() !=
            std::vector<float>({1.0F, 2.0F, 1.0F, 2.0F}) ||
        broadcast_context.to_vector() != std::vector<float>({2.0F, 3.0F}) ||
        broadcast_probability_gradient.to_vector() != std::vector<float>({5.0F}) ||
        !microllm::autograd::attention_context_layout_fusion_enabled() ||
        attention_plan_stats.entries != 0 || attention_plan_stats.hits != 0 ||
        attention_plan_stats.misses != 0 ||
        fp32_solution_key.batches != 2 ||
        fp32_solution_key.output_columns != 4 ||
        fp32_solution_stats.registered_entries != 0 ||
        grouped_qkv_key.rows != 512 || grouped_qkv_key.key_columns != 128 ||
        grouped_qkv_stats.registered_entries != 0 ||
        grouped_gate_up_key.rows != 512 ||
        grouped_gate_up_key.columns != 4864 ||
        grouped_gate_up_stats.registered_entries != 0 ||
        microllm::ops::inference_bthd_attention_enabled() ||
        microllm::ops::inference_bthd_bf16_qk_enabled() ||
        !microllm::autograd::attention_rope_layout_fusion_enabled() ||
        microllm::autograd::unique_gradient_inplace_add_enabled() ||
        !rejected_cpu_tuning || !rejected_cpu_adamw_tuning ||
        !rejected_cpu_softmax_candidate ||
        !rejected_cpu_grouped_prewarm) return 1;
    std::cout << "microLLM package consumer: pass\n";
    return 0;
}
