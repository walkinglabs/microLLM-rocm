#include <vector>
#include <tuple>

#include <gtest/gtest.h>
#include <microllm/inference/scheduler.h>

namespace microllm::inference {
namespace {

model::ModelConfig scheduler_config() {
    return {.vocabulary_size = 16,
            .dimension = 8,
            .layers = 1,
            .heads = 2,
            .kv_heads = 1,
            .ffn_dimension = 16,
            .max_sequence_length = 16,
            .rope_base = 10000.0F,
            .tie_embeddings = false};
}

std::vector<std::int32_t> suffix(const std::vector<std::int32_t>& tokens,
                                 std::size_t prompt_size) {
    return {tokens.begin() + static_cast<std::ptrdiff_t>(prompt_size),
            tokens.end()};
}

}  // namespace

TEST(ReferenceSchedulerTest, DelayedRequestsMatchIndependentGeneration) {
    const auto config = scheduler_config();
    model::TransformerModel scheduled_model(config, 83);
    ReferenceScheduler scheduler(scheduled_model);
    const std::vector<std::int32_t> first_prompt{1, 2, 3};
    const std::vector<std::int32_t> second_prompt{4, 5};
    const GenerationConfig first_config{.max_new_tokens = 4,
                                        .temperature = 0.8F,
                                        .top_k = 3,
                                        .seed = 11,
                                        .kv_cache_layer_dtypes = {}};
    const GenerationConfig second_config{.max_new_tokens = 2,
                                         .temperature = 0.0F,
                                         .top_k = 1,
                                         .seed = 17,
                                         .kv_cache_layer_dtypes = {}};
    const auto first = scheduler.submit(first_prompt, first_config);
    scheduler.step();
    EXPECT_EQ(scheduler.request(first).generated.size(), 1U);
    const auto second = scheduler.submit(second_prompt, second_config);
    EXPECT_EQ(scheduler.request(second).arrival_step, 1);
    scheduler.run_until_idle();

    model::TransformerModel first_reference(config, 83);
    model::TransformerModel second_reference(config, 83);
    EXPECT_EQ(scheduler.request(first).generated,
              suffix(generate(first_reference, first_prompt, first_config),
                     first_prompt.size()));
    EXPECT_EQ(scheduler.request(second).generated,
              suffix(generate(second_reference, second_prompt, second_config),
                     second_prompt.size()));
    EXPECT_EQ(scheduler.request(first).completion_step, 4);
    EXPECT_EQ(scheduler.request(second).completion_step, 3);
    const auto metrics = scheduler.metrics();
    EXPECT_EQ(metrics.scheduler_steps, 4);
    EXPECT_EQ(metrics.submitted_requests, 2);
    EXPECT_EQ(metrics.completed_requests, 2);
    EXPECT_EQ(metrics.prefill_calls, 2);
    EXPECT_EQ(metrics.decode_calls, 4);
    EXPECT_EQ(metrics.peak_active_requests, 2);
    EXPECT_GT(metrics.peak_cache_bytes, 0U);
    EXPECT_EQ(metrics.active_cache_bytes, 0U);
}

TEST(ReferenceSchedulerTest, ImmediateCompletionLimitsAndErrorsAreVisible) {
    model::TransformerModel model(scheduler_config(), 89);
    ReferenceScheduler scheduler(model);
    const auto immediate = scheduler.submit(
        {1}, {.max_new_tokens = 0, .kv_cache_layer_dtypes = {}});
    EXPECT_EQ(scheduler.request(immediate).state, RequestState::Completed);
    EXPECT_EQ(scheduler.request(immediate).completion_step, 0);
    EXPECT_FALSE(scheduler.has_active_requests());
    EXPECT_THROW((void)scheduler.submit({}, {}), std::invalid_argument);
    EXPECT_THROW((void)scheduler.request(999), std::out_of_range);

    const auto active = scheduler.submit(
        {1, 2}, {.max_new_tokens = 3,
                 .temperature = 0.0F,
                 .top_k = 1,
                 .kv_cache_layer_dtypes = {}});
    EXPECT_THROW(scheduler.run_until_idle(1), std::runtime_error);
    EXPECT_EQ(scheduler.request(active).generated.size(), 1U);
    scheduler.run_until_idle();
    EXPECT_EQ(scheduler.request(active).state, RequestState::Completed);
    EXPECT_EQ(scheduler.requests().size(), 2U);
    EXPECT_THROW(scheduler.run_until_idle(-2), std::invalid_argument);
}

TEST(AdmissionBatchSchedulerTest, StableBucketsMatchIndependentGeneration) {
    const auto config = scheduler_config();
    model::TransformerModel scheduled_model(config, 107);
    AdmissionBatchScheduler scheduler(scheduled_model);
    const GenerationConfig common{.max_new_tokens = 3,
                                  .temperature = 0.0F,
                                  .top_k = 1,
                                  .seed = 7,
                                  .kv_cache_layer_dtypes = {}};
    const GenerationConfig different_seed{.max_new_tokens = 3,
                                          .temperature = 0.0F,
                                          .top_k = 1,
                                          .seed = 9,
                                          .kv_cache_layer_dtypes = {}};
    const auto first = scheduler.submit({1, 2, 3}, common);
    const auto second = scheduler.submit({4, 5, 6}, common);
    const auto short_request = scheduler.submit({7, 8}, common);
    const auto other_config = scheduler.submit({9, 10, 11}, different_seed);
    scheduler.drain();
    EXPECT_EQ(scheduler.pending_request_count(), 0U);
    for (const auto& [id, prompt, generation] : {
             std::tuple{first, std::vector<std::int32_t>{1, 2, 3}, common},
             std::tuple{second, std::vector<std::int32_t>{4, 5, 6}, common},
             std::tuple{short_request, std::vector<std::int32_t>{7, 8}, common},
             std::tuple{other_config, std::vector<std::int32_t>{9, 10, 11},
                        different_seed}}) {
        model::TransformerModel independent(config, 107);
        EXPECT_EQ(scheduler.request(id).generated,
                  suffix(generate(independent, prompt, generation), prompt.size()));
    }
    auto metrics = scheduler.metrics();
    EXPECT_EQ(metrics.batch_groups, 3);
    EXPECT_EQ(metrics.singleton_groups, 2);
    EXPECT_EQ(metrics.batched_requests, 2);
    EXPECT_EQ(metrics.maximum_batch_size, 2);

    const auto late_first = scheduler.submit({12, 13}, common);
    const auto late_second = scheduler.submit({14, 15}, common);
    EXPECT_EQ(scheduler.request(late_first).arrival_step, 1);
    scheduler.drain();
    EXPECT_EQ(scheduler.request(late_first).completion_step, 2);
    EXPECT_EQ(scheduler.request(late_second).completion_step, 2);
    metrics = scheduler.metrics();
    EXPECT_EQ(metrics.drain_calls, 2);
    EXPECT_EQ(metrics.batch_groups, 4);
    EXPECT_EQ(metrics.batched_requests, 4);
    EXPECT_EQ(metrics.completed_requests, 6);
}

TEST(AdmissionBatchSchedulerTest, ImmediateAndErrorRequestsAreExplicit) {
    model::TransformerModel model(scheduler_config(), 109);
    AdmissionBatchScheduler scheduler(model);
    const auto completed = scheduler.submit(
        {1}, {.max_new_tokens = 0, .kv_cache_layer_dtypes = {}});
    EXPECT_EQ(scheduler.request(completed).state, RequestState::Completed);
    scheduler.drain();
    EXPECT_EQ(scheduler.metrics().batch_groups, 0);
    EXPECT_THROW((void)scheduler.submit(
                     {16}, {.max_new_tokens = 0,
                            .kv_cache_layer_dtypes = {}}),
                 std::out_of_range);
    EXPECT_THROW((void)scheduler.request(999), std::out_of_range);
}

}  // namespace microllm::inference
