#include <iostream>
#include <stdexcept>
#include <vector>

#include <microllm/base/device.h>
#include <microllm/model/config.h>
#include <microllm/ops/tuning.h>
#include <microllm/autograd/diagnostics.h>
#include <microllm/runtime/diagnostics.h>

int main() {
    const auto device = microllm::Device::cpu();
    const auto config = microllm::model::ModelConfig::model_s();
    const auto bias_input = microllm::Tensor::from_vector(
        {1.0F, 2.0F, 3.0F, 4.0F}, {2, 2});
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
    const auto paired_repeat = microllm::ops::repeat_gqa_kv_bthd(
        microllm::Tensor::from_vector({1.0F, 2.0F}, {1, 1, 1, 2}),
        microllm::Tensor::from_vector({3.0F, 4.0F}, {1, 1, 1, 2}), 1);
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
    const auto attention_plan_stats =
        microllm::ops::attention_layout_plan_cache_stats();
    bool rejected_cpu_tuning = false;
    bool rejected_cpu_adamw_tuning = false;
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
    if (!device.is_cpu() || config.parameter_count() == 0 ||
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
        paired_repeat.first.to_vector() != std::vector<float>({1.0F, 2.0F}) ||
        paired_repeat.second.to_vector() != std::vector<float>({3.0F, 4.0F}) ||
        broadcast_context.to_vector() != std::vector<float>({2.0F, 3.0F}) ||
        broadcast_probability_gradient.to_vector() != std::vector<float>({5.0F}) ||
        !microllm::autograd::attention_context_layout_fusion_enabled() ||
        attention_plan_stats.entries != 0 || attention_plan_stats.hits != 0 ||
        attention_plan_stats.misses != 0 ||
        !microllm::autograd::attention_rope_layout_fusion_enabled() ||
        microllm::autograd::unique_gradient_inplace_add_enabled() ||
        !rejected_cpu_tuning || !rejected_cpu_adamw_tuning) return 1;
    std::cout << "microLLM package consumer: pass\n";
    return 0;
}
