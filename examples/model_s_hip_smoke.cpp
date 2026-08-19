#include <algorithm>
#include <cmath>
#include <iostream>
#include <stdexcept>

#include <microllm/inference/kv_cache.h>
#include <microllm/model/model.h>
#include <microllm/runtime/runtime.h>

int main() {
    try {
        if (!microllm::runtime::hip_compiled() || microllm::runtime::hip_device_count() == 0) {
            std::cout << "SKIP: no compiled and visible HIP device\n";
            return 0;
        }
        const auto config = microllm::model::ModelConfig::model_s();
        const auto token = microllm::Tensor::from_int32_vector({1}, {1, 1});
        microllm::model::TransformerModel cpu_model(config, 20260819);
        microllm::inference::KVCache cpu_cache(config.layers, config.max_sequence_length);
        const auto reference = cpu_model.forward_cached(token, cpu_cache).to_vector();

        microllm::model::TransformerModel hip_model(config, 20260819);
        hip_model.to(microllm::Device::hip());
        microllm::inference::KVCache hip_cache(config.layers, config.max_sequence_length);
        const auto actual = hip_model.forward_cached(token, hip_cache).to_vector();
        float maximum_error = 0.0F;
        for (std::size_t index = 0; index < actual.size(); ++index) {
            maximum_error = std::max(maximum_error, std::abs(actual[index] - reference[index]));
        }
        const auto info = microllm::runtime::device_info(microllm::Device::hip());
        std::cout << "gpu=" << info.name << '\n';
        std::cout << "arch=" << info.architecture << '\n';
        std::cout << "parameters=" << hip_model.parameter_count() << '\n';
        std::cout << "logits=" << actual.size() << '\n';
        std::cout << "maximum_absolute_error=" << maximum_error << '\n';
        if (actual.size() != reference.size() || !std::isfinite(maximum_error) ||
            maximum_error > 2.0e-4F) {
            throw std::runtime_error("Model-S CPU/HIP logits exceed tolerance");
        }
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "model_s_hip_smoke: " << error.what() << '\n';
        return 1;
    }
}
