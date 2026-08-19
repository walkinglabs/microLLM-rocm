#include <cmath>
#include <cstdint>
#include <iostream>
#include <iomanip>
#include <stdexcept>

#include <microllm/model/model.h>
#include <microllm/training/trainer.h>

int main() {
    try {
        microllm::model::TransformerModel model(
            microllm::model::ModelConfig::model_s(), 20260819);
        microllm::training::AdamW optimizer(
            model.parameters(), {.learning_rate = 1.0e-4F,
                                 .beta1 = 0.9F,
                                 .beta2 = 0.999F,
                                 .epsilon = 1.0e-8F,
                                 .weight_decay = 0.01F});
        const auto named = model.named_parameters();
        const auto probe_index = static_cast<std::size_t>(model.config().dimension);
        const auto before = named.front().second->data().to_vector()[probe_index];
        const microllm::io::TokenBatch batch{
            microllm::Tensor::from_int32_vector({1}, {1, 1}),
            microllm::Tensor::from_int32_vector({2}, {1, 1})};
        float first_loss = 0.0F;
        microllm::training::StepMetrics metrics;
        for (std::uint64_t step = 1; step <= 3; ++step) {
            metrics = microllm::training::train_step(model, optimizer, batch, step);
            if (step == 1) first_loss = metrics.loss;
            std::cout << "step=" << step << " loss=" << metrics.loss
                      << " gradient_l2_norm=" << metrics.gradient_l2_norm << '\n';
        }
        const auto after = named.front().second->data().to_vector()[probe_index];
        std::cout << "parameters=" << model.parameter_count() << '\n';
        std::cout << "first_loss=" << first_loss << '\n';
        std::cout << "final_loss=" << metrics.loss << '\n';
        std::cout << std::setprecision(9);
        std::cout << "probe_parameter_before=" << before << '\n';
        std::cout << "probe_parameter_after=" << after << '\n';
        std::cout << "probe_parameter_delta=" << (after - before) << '\n';
        if (!std::isfinite(metrics.loss) || !std::isfinite(metrics.gradient_l2_norm) ||
            !(metrics.gradient_l2_norm > 0.0F) || before == after ||
            !(metrics.loss < first_loss)) {
            throw std::runtime_error("Model-S training trajectory did not update finite state");
        }
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "model_s_train_smoke: " << error.what() << '\n';
        return 1;
    }
}
