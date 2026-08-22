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
                                        .kv_cache_layer_dtypes = {},
                                        .stop_tokens = {}};
    const GenerationConfig second_config{.max_new_tokens = 2,
                                         .temperature = 0.0F,
                                         .top_k = 1,
                                         .seed = 17,
                                         .kv_cache_layer_dtypes = {},
                                         .stop_tokens = {}};
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
        {1}, {.max_new_tokens = 0, .kv_cache_layer_dtypes = {},
              .stop_tokens = {}});
    EXPECT_EQ(scheduler.request(immediate).state, RequestState::Completed);
    EXPECT_EQ(scheduler.request(immediate).completion_step, 0);
    EXPECT_FALSE(scheduler.has_active_requests());
    EXPECT_THROW((void)scheduler.submit({}, {}), std::invalid_argument);
    EXPECT_THROW((void)scheduler.request(999), std::out_of_range);

    const auto active = scheduler.submit(
        {1, 2}, {.max_new_tokens = 3,
                 .temperature = 0.0F,
                 .top_k = 1,
                 .kv_cache_layer_dtypes = {},
                 .stop_tokens = {}});
    EXPECT_THROW(scheduler.run_until_idle(1), std::runtime_error);
    EXPECT_EQ(scheduler.request(active).generated.size(), 1U);
    scheduler.run_until_idle();
    EXPECT_EQ(scheduler.request(active).state, RequestState::Completed);
    EXPECT_EQ(scheduler.requests().size(), 2U);
    EXPECT_THROW(scheduler.run_until_idle(-2), std::invalid_argument);
}

TEST(ReferenceSchedulerTest, CancellationIsTerminalIdempotentAndReleasesCache) {
    const auto config = scheduler_config();
    model::TransformerModel scheduled_model(config, 97);
    ReferenceScheduler scheduler(scheduled_model);
    const GenerationConfig generation{.max_new_tokens = 4,
                                      .temperature = 0.0F,
                                      .top_k = 1,
                                      .seed = 3,
                                      .kv_cache_layer_dtypes = {},
                                      .stop_tokens = {}};
    const auto cancelled = scheduler.submit({1, 2, 3}, generation);
    const auto survivor = scheduler.submit({4, 5, 6}, generation);
    scheduler.step();
    const auto before = scheduler.request(cancelled);
    ASSERT_EQ(before.generated.size(), 1U);
    EXPECT_GT(before.cache_bytes, 0U);

    EXPECT_TRUE(scheduler.cancel(cancelled));
    const auto after = scheduler.request(cancelled);
    EXPECT_EQ(after.state, RequestState::Cancelled);
    EXPECT_EQ(after.generated, before.generated);
    EXPECT_EQ(after.completion_step, 1);
    EXPECT_EQ(after.cache_bytes, 0U);
    EXPECT_FALSE(scheduler.cancel(cancelled));
    EXPECT_EQ(scheduler.active_request_count(), 1U);

    scheduler.run_until_idle();
    model::TransformerModel independent(config, 97);
    EXPECT_EQ(scheduler.request(survivor).generated,
              suffix(generate(independent, {4, 5, 6}, generation), 3));
    const auto metrics = scheduler.metrics();
    EXPECT_EQ(metrics.cancelled_requests, 1);
    EXPECT_EQ(metrics.completed_requests, 1);
    EXPECT_EQ(metrics.active_cache_bytes, 0U);
    EXPECT_FALSE(scheduler.cancel(survivor));
    EXPECT_THROW((void)scheduler.cancel(999), std::out_of_range);
}

TEST(ReferenceSchedulerTest, StopTokenCompletesEarlyAndReleasesOnlyItsCache) {
    const auto config = scheduler_config();
    const std::vector<std::int32_t> stopped_prompt{1, 2, 3};
    const std::vector<std::int32_t> survivor_prompt{4, 5, 6};
    const GenerationConfig baseline_config{.max_new_tokens = 4,
                                           .temperature = 0.0F,
                                           .top_k = 1,
                                           .seed = 13,
                                           .kv_cache_layer_dtypes = {},
                                           .stop_tokens = {}};
    model::TransformerModel oracle(config, 101);
    const auto baseline = generate(oracle, stopped_prompt, baseline_config);
    const auto stop = baseline[stopped_prompt.size()];
    auto stop_config = baseline_config;
    stop_config.stop_tokens = {stop};

    model::TransformerModel scheduled_model(config, 101);
    ReferenceScheduler scheduler(scheduled_model);
    const auto stopped = scheduler.submit(stopped_prompt, stop_config);
    const auto survivor = scheduler.submit(survivor_prompt, baseline_config);
    scheduler.step();
    const auto stopped_snapshot = scheduler.request(stopped);
    EXPECT_EQ(stopped_snapshot.state, RequestState::Completed);
    EXPECT_EQ(stopped_snapshot.completion_reason, CompletionReason::StopToken);
    EXPECT_EQ(stopped_snapshot.generated, (std::vector<std::int32_t>{stop}));
    EXPECT_EQ(stopped_snapshot.cache_bytes, 0U);
    EXPECT_EQ(scheduler.active_request_count(), 1U);
    EXPECT_GT(scheduler.request(survivor).cache_bytes, 0U);
    scheduler.run_until_idle();
    EXPECT_EQ(scheduler.request(survivor).completion_reason,
              CompletionReason::Length);
    EXPECT_EQ(scheduler.metrics().stop_completed_requests, 1);
}

TEST(AdmissionBatchSchedulerTest, StableBucketsMatchIndependentGeneration) {
    const auto config = scheduler_config();
    model::TransformerModel scheduled_model(config, 107);
    AdmissionBatchScheduler scheduler(scheduled_model);
    const GenerationConfig common{.max_new_tokens = 3,
                                  .temperature = 0.0F,
                                  .top_k = 1,
                                  .seed = 7,
                                  .kv_cache_layer_dtypes = {},
                                  .stop_tokens = {}};
    const GenerationConfig different_seed{.max_new_tokens = 3,
                                          .temperature = 0.0F,
                                          .top_k = 1,
                                          .seed = 9,
                                          .kv_cache_layer_dtypes = {},
                                          .stop_tokens = {}};
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
        {1}, {.max_new_tokens = 0, .kv_cache_layer_dtypes = {},
              .stop_tokens = {}});
    EXPECT_EQ(scheduler.request(completed).state, RequestState::Completed);
    scheduler.drain();
    EXPECT_EQ(scheduler.metrics().batch_groups, 0);
    EXPECT_THROW((void)scheduler.submit(
                     {16}, {.max_new_tokens = 0,
                            .kv_cache_layer_dtypes = {},
                            .stop_tokens = {}}),
                 std::out_of_range);
    EXPECT_THROW((void)scheduler.request(999), std::out_of_range);
}

TEST(AdmissionBatchSchedulerTest, CancellationExcludesRowsFromAdmissionGroups) {
    const auto config = scheduler_config();
    model::TransformerModel scheduled_model(config, 113);
    AdmissionBatchScheduler scheduler(scheduled_model);
    const GenerationConfig generation{.max_new_tokens = 3,
                                      .temperature = 0.0F,
                                      .top_k = 1,
                                      .seed = 5,
                                      .kv_cache_layer_dtypes = {},
                                      .stop_tokens = {}};
    const auto first = scheduler.submit({1, 2, 3}, generation);
    const auto cancelled = scheduler.submit({4, 5, 6}, generation);
    const auto third = scheduler.submit({7, 8, 9}, generation);
    EXPECT_TRUE(scheduler.cancel(cancelled));
    EXPECT_FALSE(scheduler.cancel(cancelled));
    EXPECT_EQ(scheduler.pending_request_count(), 2U);
    scheduler.drain();

    EXPECT_EQ(scheduler.request(cancelled).state, RequestState::Cancelled);
    EXPECT_TRUE(scheduler.request(cancelled).generated.empty());
    EXPECT_EQ(scheduler.request(cancelled).completion_step, 0);
    for (const auto& [id, prompt] : {
             std::pair{first, std::vector<std::int32_t>{1, 2, 3}},
             std::pair{third, std::vector<std::int32_t>{7, 8, 9}}}) {
        model::TransformerModel independent(config, 113);
        EXPECT_EQ(scheduler.request(id).generated,
                  suffix(generate(independent, prompt, generation), prompt.size()));
    }
    const auto metrics = scheduler.metrics();
    EXPECT_EQ(metrics.cancelled_requests, 1);
    EXPECT_EQ(metrics.completed_requests, 2);
    EXPECT_EQ(metrics.batch_groups, 1);
    EXPECT_EQ(metrics.batched_requests, 2);
    EXPECT_EQ(metrics.maximum_batch_size, 2);
    EXPECT_FALSE(scheduler.cancel(first));
    EXPECT_THROW((void)scheduler.cancel(999), std::out_of_range);
}

TEST(AdmissionBatchSchedulerTest, StopCompletionReasonsMatchIndependentRows) {
    const auto config = scheduler_config();
    const std::vector<std::vector<std::int32_t>> prompts{{1, 2, 3}, {4, 5, 6}};
    const GenerationConfig baseline_config{.max_new_tokens = 4,
                                           .temperature = 0.0F,
                                           .top_k = 1,
                                           .seed = 19,
                                           .kv_cache_layer_dtypes = {},
                                           .stop_tokens = {}};
    model::TransformerModel oracle(config, 127);
    const auto baseline = generate(oracle, prompts.front(), baseline_config);
    auto stopped_config = baseline_config;
    const auto first_stop = baseline[prompts.front().size()];
    const auto second_stop = static_cast<std::int32_t>(
        (first_stop + 1) % config.vocabulary_size);
    stopped_config.stop_tokens = {first_stop, second_stop};
    auto reversed_config = stopped_config;
    std::reverse(reversed_config.stop_tokens.begin(), reversed_config.stop_tokens.end());
    model::TransformerModel scheduled_model(config, 127);
    AdmissionBatchScheduler scheduler(scheduled_model);
    std::vector<RequestId> ids;
    ids.push_back(scheduler.submit(prompts[0], stopped_config));
    ids.push_back(scheduler.submit(prompts[1], reversed_config));
    scheduler.drain();
    std::int64_t stopped_rows = 0;
    for (std::size_t row = 0; row < prompts.size(); ++row) {
        model::TransformerModel independent(config, 127);
        EXPECT_EQ(scheduler.request(ids[row]).generated,
                  suffix(generate(independent, prompts[row], stopped_config),
                         prompts[row].size()));
        if (scheduler.request(ids[row]).completion_reason ==
            CompletionReason::StopToken) {
            ++stopped_rows;
        }
    }
    EXPECT_GE(stopped_rows, 1);
    EXPECT_EQ(scheduler.metrics().stop_completed_requests, stopped_rows);
    EXPECT_EQ(scheduler.metrics().maximum_batch_size, 2);
}

TEST(ContinuousBatchSchedulerTest, RefillsFreedSlotAndMatchesIndependentRows) {
    const auto config = scheduler_config();
    for (const auto dtype : {DType::Float32, DType::BFloat16}) {
        model::TransformerModel scheduled_model(config, 149);
        ContinuousBatchScheduler scheduler(
            scheduled_model,
            {.max_slots = 2, .kv_cache_dtype = dtype,
             .kv_cache_layer_dtypes = {}});
        const GenerationConfig short_generation{
            .max_new_tokens = 2, .temperature = 0.0F, .top_k = 1,
            .seed = 3, .kv_cache_dtype = dtype,
            .kv_cache_layer_dtypes = {}, .stop_tokens = {}};
        const GenerationConfig long_generation{
            .max_new_tokens = 4, .temperature = 0.0F, .top_k = 1,
            .seed = 5, .kv_cache_dtype = dtype,
            .kv_cache_layer_dtypes = {}, .stop_tokens = {}};
        const std::vector<std::int32_t> first_prompt{1, 2, 3};
        const std::vector<std::int32_t> second_prompt{4, 5, 6};
        const std::vector<std::int32_t> late_prompt{7, 8};
        const auto first = scheduler.submit(first_prompt, short_generation);
        const auto second = scheduler.submit(second_prompt, long_generation);
        scheduler.step();
        EXPECT_EQ(scheduler.request(first).slot, 0);
        EXPECT_EQ(scheduler.request(second).slot, 1);
        const auto late = scheduler.submit(late_prompt, short_generation);
        EXPECT_EQ(scheduler.request(late).slot, -1);
        scheduler.step();
        EXPECT_EQ(scheduler.request(first).state, RequestState::Completed);
        EXPECT_EQ(scheduler.request(first).slot, -1);
        EXPECT_EQ(scheduler.request(late).state, RequestState::PendingPrefill);
        scheduler.step();
        EXPECT_EQ(scheduler.request(late).slot, 0);
        EXPECT_GT(scheduler.request(late).cache_bytes, 0U);
        scheduler.run_until_idle();

        for (const auto& [id, prompt, generation] : {
                 std::tuple{first, first_prompt, short_generation},
                 std::tuple{second, second_prompt, long_generation},
                 std::tuple{late, late_prompt, short_generation}}) {
            model::TransformerModel independent(config, 149);
            EXPECT_EQ(scheduler.request(id).generated,
                      suffix(generate(independent, prompt, generation),
                             prompt.size()));
            EXPECT_EQ(scheduler.request(id).state, RequestState::Completed);
            EXPECT_EQ(scheduler.request(id).slot, -1);
            EXPECT_EQ(scheduler.request(id).cache_bytes, 0U);
            EXPECT_GE(scheduler.request(id).time_to_first_token_ms, 0.0);
            EXPECT_GE(scheduler.request(id).completion_latency_ms,
                      scheduler.request(id).time_to_first_token_ms);
        }
        const auto metrics = scheduler.metrics();
        EXPECT_EQ(metrics.scheduler_steps, 4);
        EXPECT_EQ(metrics.slot_admissions, 3);
        EXPECT_EQ(metrics.slot_refills, 1);
        EXPECT_EQ(metrics.row_prefill_calls, 3);
        EXPECT_EQ(metrics.prefill_batch_calls, 2);
        EXPECT_EQ(metrics.batched_prefill_calls, 1);
        EXPECT_EQ(metrics.batched_prefill_rows, 2);
        EXPECT_EQ(metrics.batch_decode_calls, 3);
        EXPECT_EQ(metrics.uniform_batch_decode_calls, 1);
        EXPECT_EQ(metrics.divergent_batch_decode_calls, 1);
        EXPECT_EQ(metrics.compacted_batch_decode_calls, 2);
        EXPECT_EQ(metrics.positions_aware_batch_decode_calls, 2);
        EXPECT_EQ(metrics.logical_decode_rows, 5);
        EXPECT_EQ(metrics.dummy_decode_rows, 0);
        EXPECT_EQ(metrics.inactive_rows_skipped, 1);
        EXPECT_EQ(metrics.selection_calls, 4);
        EXPECT_EQ(metrics.occupied_slot_steps, 8);
        EXPECT_DOUBLE_EQ(metrics.slot_utilization, 1.0);
        EXPECT_EQ(metrics.occupied_slots, 0);
        EXPECT_EQ(metrics.peak_occupied_slots, 2);
        EXPECT_GT(metrics.allocated_cache_bytes, 0U);
        EXPECT_EQ(metrics.active_cache_bytes, 0U);
        EXPECT_GT(metrics.peak_active_cache_bytes, 0U);
        EXPECT_TRUE(scheduler.selection_diagnostics().empty());

        model::TransformerModel recycled_model(config, 149);
        ContinuousBatchScheduler recycled(
            recycled_model,
            {.max_slots = 1, .max_sequence_length = 6,
             .kv_cache_dtype = dtype, .kv_cache_layer_dtypes = {},
             .capture_selection_diagnostics = true});
        const auto recycled_first = recycled.submit(
            {1, 2, 3}, short_generation);
        const auto recycled_second = recycled.submit(
            {4, 5, 6, 7}, short_generation);
        recycled.run_until_idle();
        for (const auto& [id, prompt] : {
                 std::pair{recycled_first,
                           std::vector<std::int32_t>{1, 2, 3}},
                 std::pair{recycled_second,
                           std::vector<std::int32_t>{4, 5, 6, 7}}}) {
            model::TransformerModel independent(config, 149);
            EXPECT_EQ(recycled.request(id).generated,
                      suffix(generate(independent, prompt, short_generation),
                             prompt.size()));
        }
        EXPECT_EQ(recycled.metrics().slot_admissions, 2);
        EXPECT_EQ(recycled.metrics().slot_refills, 1);
        EXPECT_EQ(recycled.metrics().allocated_cache_bytes,
                  static_cast<std::size_t>(2 * config.layers * config.kv_heads *
                                           config.head_dimension() * 6) *
                      dtype_size(dtype));
        ASSERT_EQ(recycled.selection_diagnostics().size(), 4U);
        for (const auto& diagnostic : recycled.selection_diagnostics()) {
            EXPECT_EQ(diagnostic.logit_batch_size, 1);
            EXPECT_EQ(diagnostic.device_selected_token, diagnostic.top1_token);
            EXPECT_TRUE(diagnostic.device_argmax_matches_top1);
            EXPECT_GE(diagnostic.top1_top2_margin, 0.0F);
            EXPECT_NE(diagnostic.logit_source, "none");
        }

        model::TransformerModel serial_prefill_model(config, 149);
        ContinuousBatchScheduler serial_prefill(
            serial_prefill_model,
            {.max_slots = 2, .max_sequence_length = 6,
             .kv_cache_dtype = dtype, .kv_cache_layer_dtypes = {},
             .batch_equal_length_prefill = false});
        const auto serial_first = serial_prefill.submit(
            {1, 2, 3}, short_generation);
        const auto serial_second = serial_prefill.submit(
            {4, 5, 6}, short_generation);
        serial_prefill.run_until_idle();
        EXPECT_EQ(serial_prefill.metrics().prefill_batch_calls, 2);
        EXPECT_EQ(serial_prefill.metrics().batched_prefill_calls, 0);
        for (const auto& [id, prompt] : {
                 std::pair{serial_first,
                           std::vector<std::int32_t>{1, 2, 3}},
                 std::pair{serial_second,
                           std::vector<std::int32_t>{4, 5, 6}}}) {
            model::TransformerModel independent(config, 149);
            EXPECT_EQ(serial_prefill.request(id).generated,
                      suffix(generate(independent, prompt, short_generation),
                             prompt.size()));
        }
    }
}

TEST(ContinuousBatchSchedulerTest, DelayedSamplingMatchesIndependentRequests) {
    const auto config = scheduler_config();
    model::TransformerModel scheduled_model(config, 151);
    ContinuousBatchScheduler scheduler(
        scheduled_model, {.max_slots = 2, .kv_cache_dtype = DType::Float32,
                          .kv_cache_layer_dtypes = {}});
    const GenerationConfig first_generation{
        .max_new_tokens = 4, .temperature = 0.8F, .top_k = 3,
        .seed = 11, .kv_cache_layer_dtypes = {}, .stop_tokens = {}};
    const GenerationConfig second_generation{
        .max_new_tokens = 2, .temperature = 0.7F, .top_k = 4,
        .seed = 17, .kv_cache_layer_dtypes = {}, .stop_tokens = {}};
    const std::vector<std::int32_t> first_prompt{1, 2, 3};
    const std::vector<std::int32_t> second_prompt{4, 5};
    const auto first = scheduler.submit(first_prompt, first_generation);
    scheduler.step();
    const auto second = scheduler.submit(second_prompt, second_generation);
    scheduler.run_until_idle();
    model::TransformerModel first_oracle(config, 151);
    model::TransformerModel second_oracle(config, 151);
    EXPECT_EQ(scheduler.request(first).generated,
              suffix(generate(first_oracle, first_prompt, first_generation),
                     first_prompt.size()));
    EXPECT_EQ(scheduler.request(second).generated,
              suffix(generate(second_oracle, second_prompt, second_generation),
                     second_prompt.size()));
    EXPECT_EQ(scheduler.request(first).completion_step, 4);
    EXPECT_EQ(scheduler.request(second).completion_step, 3);
    EXPECT_EQ(scheduler.metrics().slot_admissions, 2);
    EXPECT_GE(scheduler.metrics().compacted_batch_decode_calls, 1);
}

TEST(ContinuousBatchSchedulerTest, StopCancelAndPolicyErrorsAreExplicit) {
    const auto config = scheduler_config();
    const std::vector<std::int32_t> stopped_prompt{1, 2, 3};
    const GenerationConfig baseline{
        .max_new_tokens = 4, .temperature = 0.0F, .top_k = 1,
        .seed = 19, .kv_cache_dtype = DType::BFloat16,
        .kv_cache_layer_dtypes = {}, .stop_tokens = {}};
    model::TransformerModel oracle(config, 157);
    auto stopped = baseline;
    stopped.stop_tokens = {
        generate(oracle, stopped_prompt, baseline)[stopped_prompt.size()]};

    model::TransformerModel model(config, 157);
    ContinuousBatchScheduler scheduler(
        model, {.max_slots = 2, .kv_cache_dtype = DType::BFloat16,
                .kv_cache_layer_dtypes = {}});
    const auto stopped_id = scheduler.submit(stopped_prompt, stopped);
    const auto cancelled = scheduler.submit({4, 5, 6}, baseline);
    scheduler.step();
    EXPECT_EQ(scheduler.request(stopped_id).completion_reason,
              CompletionReason::StopToken);
    EXPECT_EQ(scheduler.request(stopped_id).slot, -1);
    EXPECT_TRUE(scheduler.cancel(cancelled));
    EXPECT_FALSE(scheduler.cancel(cancelled));
    const auto replacement = scheduler.submit({7, 8}, baseline);
    scheduler.step();
    EXPECT_EQ(scheduler.request(replacement).slot, 0);
    scheduler.run_until_idle();
    model::TransformerModel replacement_oracle(config, 157);
    EXPECT_EQ(scheduler.request(replacement).generated,
              suffix(generate(replacement_oracle, {7, 8}, baseline), 2));
    EXPECT_EQ(scheduler.metrics().stop_completed_requests, 1);
    EXPECT_EQ(scheduler.metrics().cancelled_requests, 1);
    EXPECT_GE(scheduler.metrics().slot_refills, 1);

    EXPECT_THROW((void)scheduler.submit(
                     {1}, {.max_new_tokens = 1,
                           .kv_cache_dtype = DType::Float32,
                           .kv_cache_layer_dtypes = {}, .stop_tokens = {}}),
                 std::invalid_argument);
    EXPECT_THROW((void)scheduler.request(999), std::out_of_range);
    EXPECT_THROW(scheduler.run_until_idle(-2), std::invalid_argument);
    EXPECT_THROW((void)ContinuousBatchScheduler(
                     model, {.max_slots = 0,
                             .kv_cache_layer_dtypes = {}}),
                 std::invalid_argument);
    EXPECT_THROW((void)ContinuousBatchScheduler(
                     model, {.max_slots = 1,
                             .kv_cache_layer_dtypes = {
                                 DType::Float32, DType::BFloat16}}),
                 std::invalid_argument);
    ContinuousBatchScheduler bounded(
        model, {.max_slots = 1, .max_sequence_length = 4,
                .kv_cache_dtype = DType::BFloat16,
                .kv_cache_layer_dtypes = {}});
    EXPECT_THROW((void)bounded.submit({1, 2, 3}, baseline),
                 std::invalid_argument);
    EXPECT_THROW((void)ContinuousBatchScheduler(
                     model, {.max_slots = 1, .max_sequence_length = 17,
                             .kv_cache_layer_dtypes = {}}),
                 std::invalid_argument);
}

}  // namespace microllm::inference
