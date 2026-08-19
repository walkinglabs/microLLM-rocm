#pragma once

#include <cstdint>

#include <microllm/io/token_dataset.h>
#include <microllm/model/model.h>
#include <microllm/training/optimizer.h>

namespace microllm::training {

struct StepMetrics {
    std::uint64_t step = 0;
    float loss = 0.0F;
    float gradient_l2_norm = 0.0F;
};

[[nodiscard]] StepMetrics train_step(model::TransformerModel& model, AdamW& optimizer,
                                     const io::TokenBatch& batch, std::uint64_t step);

}  // namespace microllm::training
