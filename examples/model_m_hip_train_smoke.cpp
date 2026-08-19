#include <cmath>
#include <cstddef>
#include <iomanip>
#include <iostream>
#include <stdexcept>

#include <microllm/model/model.h>
#include <microllm/runtime/memory.h>
#include <microllm/runtime/runtime.h>
#include <microllm/training/trainer.h>

int main() {
    try {
        if (!microllm::runtime::hip_compiled() || microllm::runtime::hip_device_count() == 0) {
            std::cout << "SKIP: no compiled and visible HIP device\n";
            return 0;
        }
        const auto config = microllm::model::ModelConfig::model_m();
        microllm::runtime::reset_allocation_peak(microllm::Device::hip());
        microllm::model::TransformerModel model(config, 20260819);
        model.to(microllm::Device::hip());
        microllm::training::AdamW optimizer(
            model.parameters(), {.learning_rate = 1.0e-4F,
                                 .beta1 = 0.9F,
                                 .beta2 = 0.999F,
                                 .epsilon = 1.0e-8F,
                                 .weight_decay = 0.01F});
        const auto parameters = model.named_parameters();
        const auto probe_index = static_cast<std::size_t>(config.dimension);
        const auto before = parameters.front().second->data().to_vector()[probe_index];
        const microllm::io::TokenBatch batch{
            microllm::Tensor::from_int32_vector({1}, {1, 1}),
            microllm::Tensor::from_int32_vector({2}, {1, 1})};
        const auto metrics = microllm::training::train_step(model, optimizer, batch, 1);
        const auto after = parameters.front().second->data().to_vector()[probe_index];
        const auto memory = microllm::runtime::allocation_stats(microllm::Device::hip());
        const auto info = microllm::runtime::device_info(microllm::Device::hip());
        std::cout << std::setprecision(9);
        std::cout << "gpu=" << info.name << '\n';
        std::cout << "arch=" << info.architecture << '\n';
        std::cout << "parameters=" << model.parameter_count() << '\n';
        std::cout << "fp32_weight_bytes=" << config.weight_bytes(4) << '\n';
        std::cout << "loss=" << metrics.loss << '\n';
        std::cout << "gradient_l2_norm=" << metrics.gradient_l2_norm << '\n';
        std::cout << "probe_parameter_delta=" << (after - before) << '\n';
        std::cout << "peak_engine_hip_bytes=" << memory.peak_bytes << '\n';
        if (model.parameter_count() != 31'334'912U || !std::isfinite(metrics.loss) ||
            !std::isfinite(metrics.gradient_l2_norm) || !(metrics.gradient_l2_norm > 0.0F) ||
            before == after) {
            throw std::runtime_error("Model-M HIP training-step contract failed");
        }
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "model_m_hip_train_smoke: " << error.what() << '\n';
        return 1;
    }
}
