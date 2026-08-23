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
    microllm::autograd::enable_gradient_accumulation_diagnostics(false);
    microllm::runtime::enable_strided_copy_diagnostics(false);
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
        !rejected_cpu_tuning || !rejected_cpu_adamw_tuning) return 1;
    std::cout << "microLLM package consumer: pass\n";
    return 0;
}
