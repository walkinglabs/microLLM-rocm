#include <algorithm>
#include <cmath>

#include <gtest/gtest.h>
#include <microllm/multi_gpu/data_parallel.h>
#include <microllm/profiling/trace.h>
#include <microllm/runtime/runtime.h>

namespace microllm::multi_gpu {
namespace {

model::ModelConfig config() {
    return {.vocabulary_size = 8,
            .dimension = 8,
            .layers = 1,
            .heads = 2,
            .kv_heads = 1,
            .ffn_dimension = 16,
            .max_sequence_length = 4,
            .rope_base = 10000.0F,
            .tie_embeddings = false};
}

std::vector<io::TokenBatch> local_batches() {
    return {
        {Tensor::from_int32_vector({0, 1, 2, 3}, {1, 4}),
         Tensor::from_int32_vector({1, 2, 3, 0}, {1, 4})},
        {Tensor::from_int32_vector({3, 2, 1, 0}, {1, 4}),
         Tensor::from_int32_vector({2, 1, 0, 3}, {1, 4})},
    };
}

io::TokenBatch global_batch() {
    return {Tensor::from_int32_vector({0, 1, 2, 3, 3, 2, 1, 0}, {2, 4}),
            Tensor::from_int32_vector({1, 2, 3, 0, 2, 1, 0, 3}, {2, 4})};
}

float difference(model::TransformerModel& left, model::TransformerModel& right) {
    const auto left_parameters = left.parameters();
    const auto right_parameters = right.parameters();
    float maximum = 0.0F;
    for (std::size_t parameter = 0; parameter < left_parameters.size(); ++parameter) {
        const auto lhs = left_parameters[parameter]->data().to_vector();
        const auto rhs = right_parameters[parameter]->data().to_vector();
        for (std::size_t index = 0; index < lhs.size(); ++index) {
            maximum = std::max(maximum, std::abs(lhs[index] - rhs[index]));
        }
    }
    return maximum;
}

}  // namespace

TEST(DataParallelTrainerTest, MultiStepTwoRankTrainingMatchesSingleGlobalBatch) {
    if (runtime::hip_device_count() < 2) GTEST_SKIP() << "two visible HIP devices required";
    const training::AdamWConfig optimizer{.learning_rate = 0.005F,
                                           .beta1 = 0.9F,
                                           .beta2 = 0.99F,
                                           .epsilon = 1.0e-8F,
                                           .weight_decay = 0.0F};
    model::TransformerModel reference(config(), 503);
    training::AdamW reference_optimizer(reference.parameters(), optimizer);
    DataParallelTrainer trainer(
        config(), 503,
        {.device_indices = {0, 1},
         .maximum_bucket_bytes = 4096,
         .optimizer = optimizer});

    for (std::uint64_t step = 1; step <= 3; ++step) {
        reference_optimizer.zero_grad();
        reference.loss(global_batch().inputs, global_batch().targets).backward();
        reference_optimizer.step();
        const auto metrics = trainer.step(local_batches(), step);
        EXPECT_EQ(metrics.step, step);
        EXPECT_EQ(metrics.rank_losses.size(), 2U);
        EXPECT_GT(metrics.buckets.bucket_count, 0U);
        EXPECT_GT(metrics.buckets.parameter_count, 0U);
        EXPECT_TRUE(metrics.parameter_check_performed);
        EXPECT_EQ(metrics.maximum_parameter_difference, 0.0F);
        EXPECT_GE(metrics.forward_backward_ms, 0.0);
        EXPECT_GE(metrics.communication_ms, 0.0);
        EXPECT_GE(metrics.optimizer_ms, 0.0);
        EXPECT_GE(metrics.verification_ms, 0.0);
        EXPECT_GE(metrics.total_ms, metrics.communication_ms);
    }
    EXPECT_LE(difference(reference, trainer.model(0)), 2.0e-5F);
    EXPECT_EQ(difference(trainer.model(0), trainer.model(1)), 0.0F);
}

TEST(DataParallelTrainerTest, ParameterVerificationIntervalIsExplicit) {
    if (runtime::hip_device_count() < 2) GTEST_SKIP() << "two visible HIP devices required";
    {
        DataParallelTrainer trainer(
            config(), 527,
            {.device_indices = {0, 1},
             .maximum_bucket_bytes = 4096,
             .parameter_check_interval = 2,
             .optimizer = {}});
        const auto first = trainer.step(local_batches(), 1);
        EXPECT_FALSE(first.parameter_check_performed);
        EXPECT_EQ(first.verification_ms, 0.0);
        const auto second = trainer.step(local_batches(), 2);
        EXPECT_TRUE(second.parameter_check_performed);
        EXPECT_GE(second.verification_ms, 0.0);
        EXPECT_EQ(second.maximum_parameter_difference, 0.0F);
    }
    {
        DataParallelTrainer disabled(
            config(), 529,
            {.device_indices = {0, 1},
             .maximum_bucket_bytes = 4096,
             .parameter_check_interval = 0,
             .optimizer = {}});
        const auto skipped = disabled.step(local_batches(), 1);
        EXPECT_FALSE(skipped.parameter_check_performed);
        EXPECT_EQ(skipped.verification_ms, 0.0);
    }
}

TEST(DataParallelTrainerTest, RejectsUnequalLocalBatchWeighting) {
    if (runtime::hip_device_count() < 2) GTEST_SKIP() << "two visible HIP devices required";
    DataParallelTrainer trainer(
        config(), 509,
        {.device_indices = {0, 1}, .maximum_bucket_bytes = 4096, .optimizer = {}});
    auto batches = local_batches();
    batches[1] = {Tensor::from_int32_vector({0, 1}, {1, 2}),
                  Tensor::from_int32_vector({1, 0}, {1, 2})};
    EXPECT_THROW((void)trainer.step(batches, 1), std::invalid_argument);

    batches = local_batches();
    batches[1].targets = Tensor::from_int32_vector({2, 1, -100, -100}, {1, 4});
    EXPECT_THROW((void)trainer.step(batches, 1), std::invalid_argument);
}

TEST(DataParallelTrainerTest, EmitsStageLevelProfileRecords) {
    if (runtime::hip_device_count() < 2) GTEST_SKIP() << "two visible HIP devices required";
    DataParallelTrainer trainer(
        config(), 521,
        {.device_indices = {0, 1}, .maximum_bucket_bytes = 4096, .optimizer = {}});
    profiling::TraceOptions options;
    options.phase = "distributed_step";
    options.record_operators = false;
    options.record_layers = true;
    options.record_model = true;
    options.capture_values = false;
    profiling::TraceSession session("microllm", "ddp-unit", options);
    {
        profiling::ScopedTraceSession active(session);
        (void)trainer.step(local_batches(), 1);
    }
    const auto has_name = [&](const char* name) {
        return std::any_of(session.records().begin(), session.records().end(),
                           [&](const auto& record) { return record.name == name; });
    };
    EXPECT_TRUE(has_name("data_parallel.forward_backward"));
    EXPECT_TRUE(has_name("data_parallel.all_reduce"));
    EXPECT_TRUE(has_name("data_parallel.optimizer"));
    EXPECT_TRUE(has_name("data_parallel.parameter_verification"));
    EXPECT_TRUE(has_name("data_parallel.step"));
}

}  // namespace microllm::multi_gpu
