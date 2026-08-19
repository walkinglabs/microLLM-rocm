#include <microllm/training/trainer.h>

#include <cmath>
#include <stdexcept>

namespace microllm::training {

StepMetrics train_step(model::TransformerModel& model, AdamW& optimizer,
                       const io::TokenBatch& batch, std::uint64_t step) {
    if (batch.inputs.shape() != batch.targets.shape()) {
        throw std::invalid_argument("training input and target shapes must match");
    }
    optimizer.zero_grad();
    const auto loss = model.loss(batch.inputs, batch.targets);
    const auto loss_value = loss.data().to_vector()[0];
    if (!std::isfinite(loss_value)) throw std::runtime_error("training loss is not finite");
    loss.backward();

    double squared_norm = 0.0;
    for (const auto* parameter : model.parameters()) {
        if (!parameter->has_grad()) continue;
        for (const auto gradient : parameter->grad().to_vector()) {
            squared_norm += static_cast<double>(gradient) * gradient;
        }
    }
    optimizer.step();
    return {.step = step,
            .loss = loss_value,
            .gradient_l2_norm = static_cast<float>(std::sqrt(squared_norm))};
}

}  // namespace microllm::training
