#include <cmath>
#include <vector>

#include <gtest/gtest.h>
#include <microllm/io/token_dataset.h>
#include <microllm/model/model.h>
#include <microllm/training/trainer.h>

namespace microllm::training {
namespace {

model::ModelConfig tiny_config() {
    return {.vocabulary_size = 6,
            .dimension = 8,
            .layers = 1,
            .heads = 2,
            .kv_heads = 1,
            .ffn_dimension = 16,
            .max_sequence_length = 6,
            .rope_base = 10000.0F,
            .tie_embeddings = false};
}

}  // namespace

TEST(TrainerTest, MultipleStepsLowerLossWithFiniteMetrics) {
    io::TokenDataset dataset({0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3}, 3);
    model::TransformerModel model(tiny_config(), 31);
    AdamW optimizer(model.parameters(), {.learning_rate = 0.02F,
                                         .beta1 = 0.9F,
                                         .beta2 = 0.99F,
                                         .epsilon = 1.0e-8F,
                                         .weight_decay = 0.0F});
    float first = 0.0F;
    float last = 0.0F;
    for (std::uint64_t step = 1; step <= 20; ++step) {
        const auto metrics = train_step(model, optimizer, dataset.next_batch(2), step);
        if (step == 1) first = metrics.loss;
        last = metrics.loss;
        EXPECT_TRUE(std::isfinite(metrics.loss));
        EXPECT_TRUE(std::isfinite(metrics.gradient_l2_norm));
        EXPECT_GE(metrics.gradient_l2_norm, 0.0F);
    }
    EXPECT_LT(last, first);
}

TEST(TrainerTest, Bf16LinearPolicyLowersLossWithFp32MasterAdamw) {
    io::TokenDataset dataset({0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3}, 3);
    auto config = tiny_config();
    config.linear_precision = model::LinearPrecision::BFloat16;
    model::TransformerModel transformer(config, 32);
    AdamW optimizer(transformer.parameters(), {.learning_rate = 0.02F,
                                                .beta1 = 0.9F,
                                                .beta2 = 0.99F,
                                                .epsilon = 1.0e-8F,
                                                .weight_decay = 0.0F});
    float first = 0.0F;
    float last = 0.0F;
    for (std::uint64_t step = 1; step <= 20; ++step) {
        const auto metrics = train_step(transformer, optimizer, dataset.next_batch(2), step);
        if (step == 1) first = metrics.loss;
        last = metrics.loss;
        EXPECT_TRUE(std::isfinite(metrics.loss));
        EXPECT_TRUE(std::isfinite(metrics.gradient_l2_norm));
    }
    EXPECT_LT(last, first);
    for (const auto& [name, parameter] : transformer.named_parameters()) {
        EXPECT_EQ(parameter->data().dtype(), DType::Float32) << name;
    }
}

TEST(TrainerTest, RejectsMismatchedBatchShapes) {
    model::TransformerModel model(tiny_config(), 37);
    AdamW optimizer(model.parameters());
    const io::TokenBatch bad{Tensor::from_int32_vector({1, 2}, {1, 2}),
                             Tensor::from_int32_vector({2}, {1, 1})};
    EXPECT_THROW((void)train_step(model, optimizer, bad, 1), std::invalid_argument);
}

}  // namespace microllm::training
