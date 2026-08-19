#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>

#include <microllm/model/model.h>

int main() {
    try {
        const auto config = microllm::model::ModelConfig::model_s();
        microllm::model::TransformerModel model(config, 20260819);
        const auto logits =
            model.forward(microllm::Tensor::from_int32_vector({1}, {1, 1})).data().to_vector();
        float minimum = std::numeric_limits<float>::infinity();
        float maximum = -std::numeric_limits<float>::infinity();
        for (const auto value : logits) {
            if (!std::isfinite(value)) throw std::runtime_error("Model-S produced non-finite logits");
            minimum = std::min(minimum, value);
            maximum = std::max(maximum, value);
        }
        std::cout << "parameters=" << model.parameter_count() << '\n';
        std::cout << "fp32_weight_bytes=" << config.weight_bytes(4) << '\n';
        std::cout << "logits=" << logits.size() << '\n';
        std::cout << "logit_min=" << minimum << '\n';
        std::cout << "logit_max=" << maximum << '\n';
        if (model.parameter_count() != 15'586'176U || logits.size() != 8192U) {
            throw std::runtime_error("Model-S smoke contract mismatch");
        }
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "model_s_smoke: " << error.what() << '\n';
        return 1;
    }
}
