#include <algorithm>
#include <random>
#include <vector>

#include <gtest/gtest.h>
#include <microllm/inference/generator.h>

namespace microllm::inference {
namespace {

model::ModelConfig generation_config() {
    return {.vocabulary_size = 8,
            .dimension = 8,
            .layers = 1,
            .heads = 2,
            .kv_heads = 1,
            .ffn_dimension = 16,
            .max_sequence_length = 12,
            .rope_base = 10000.0F,
            .tie_embeddings = false};
}

}  // namespace

TEST(SamplingTest, GreedyAndTopOneChooseMaximumLogit) {
    std::mt19937_64 generator(1);
    EXPECT_EQ(sample_token({-1, 3, 2}, 0.0F, 0, generator), 1);
    EXPECT_EQ(sample_token({-1, 3, 2}, 1.0F, 1, generator), 1);
}

TEST(SamplingTest, FixedSeedMakesTopKSamplingDeterministic) {
    std::mt19937_64 first(17);
    std::mt19937_64 second(17);
    std::vector<std::int32_t> first_tokens;
    std::vector<std::int32_t> second_tokens;
    for (int index = 0; index < 20; ++index) {
        first_tokens.push_back(sample_token({1, 2, 3, 4}, 0.8F, 3, first));
        second_tokens.push_back(sample_token({1, 2, 3, 4}, 0.8F, 3, second));
    }
    EXPECT_EQ(first_tokens, second_tokens);
    for (const auto token : first_tokens) EXPECT_NE(token, 0);
}

TEST(GeneratorTest, UsesCacheAndReturnsRequestedValidTokens) {
    model::TransformerModel model(generation_config(), 47);
    const auto tokens = generate(model, {1, 2, 3},
                                 {.max_new_tokens = 5,
                                  .temperature = 0.0F,
                                  .top_k = 0,
                                  .seed = 99,
                                  .kv_cache_layer_dtypes = {},
                                  .stop_tokens = {}});
    ASSERT_EQ(tokens.size(), 8U);
    EXPECT_EQ(std::vector<std::int32_t>(tokens.begin(), tokens.begin() + 3),
              (std::vector<std::int32_t>{1, 2, 3}));
    for (const auto token : tokens) {
        EXPECT_GE(token, 0);
        EXPECT_LT(token, model.config().vocabulary_size);
    }
    model::TransformerModel bf16_model(generation_config(), 47);
    const auto bf16_tokens = generate(
        bf16_model, {1, 2, 3},
        {.max_new_tokens = 5,
         .temperature = 0.0F,
         .top_k = 0,
         .seed = 99,
         .kv_cache_dtype = DType::BFloat16,
         .kv_cache_layer_dtypes = {},
         .stop_tokens = {}});
    EXPECT_EQ(bf16_tokens, tokens);
    model::TransformerModel explicit_policy_model(generation_config(), 47);
    const auto explicit_policy_tokens = generate(
        explicit_policy_model, {1, 2, 3},
        {.max_new_tokens = 5,
         .temperature = 0.0F,
         .top_k = 0,
         .seed = 99,
         .kv_cache_layer_dtypes = {DType::BFloat16},
         .stop_tokens = {}});
    EXPECT_EQ(explicit_policy_tokens, tokens);
}

TEST(GeneratorTest, RejectsInvalidSamplingAndContext) {
    std::mt19937_64 generator(1);
    EXPECT_THROW((void)sample_token({}, 1.0F, 0, generator), std::invalid_argument);
    EXPECT_THROW((void)sample_token({1, 2}, -1.0F, 0, generator), std::invalid_argument);
    model::TransformerModel model(generation_config(), 53);
    EXPECT_THROW((void)generate(model, {}, {}), std::invalid_argument);
    EXPECT_THROW((void)generate(
                     model, {1, 2, 3},
                     {.max_new_tokens = 10, .kv_cache_layer_dtypes = {},
                      .stop_tokens = {}}),
                 std::invalid_argument);
    EXPECT_THROW((void)generate(
                     model, {8},
                     {.max_new_tokens = 0, .kv_cache_layer_dtypes = {},
                      .stop_tokens = {}}),
                 std::out_of_range);
    EXPECT_THROW((void)generate(
                     model, {1, 2},
                     {.max_new_tokens = 1,
                      .kv_cache_layer_dtypes = {DType::Float32,
                                                DType::BFloat16},
                      .stop_tokens = {}}),
                 std::invalid_argument);
}

TEST(GeneratorTest, StaticBatchMatchesIndependentGreedyAndSampling) {
    const std::vector<std::vector<std::int32_t>> prompts{{1, 2, 3}, {4, 5, 6}};
    for (const auto stochastic : {false, true}) {
        const GenerationConfig generation{
            .max_new_tokens = 4,
            .temperature = stochastic ? 0.8F : 0.0F,
            .top_k = stochastic ? 3 : 1,
            .seed = 23,
            .kv_cache_layer_dtypes = {},
            .stop_tokens = {}};
        model::TransformerModel batched_model(generation_config(), 59);
        const auto batched = generate_batch(batched_model, prompts, generation);
        ASSERT_EQ(batched.size(), prompts.size());
        for (std::size_t row = 0; row < prompts.size(); ++row) {
            model::TransformerModel independent(generation_config(), 59);
            EXPECT_EQ(batched[row], generate(independent, prompts[row], generation));
        }
    }
}

TEST(GeneratorTest, StaticBatchRejectsIncompatibleRequests) {
    model::TransformerModel model(generation_config(), 61);
    EXPECT_THROW((void)generate_batch(model, {}, {}), std::invalid_argument);
    EXPECT_THROW((void)generate_batch(model, {{1, 2}, {3}}, {}),
                 std::invalid_argument);
    EXPECT_THROW((void)generate_batch(
                     model, {{1, 2}, {3, 4}},
                     {.max_new_tokens = 1,
                      .kv_cache_layer_dtypes = {DType::Float32,
                                                DType::BFloat16},
                      .stop_tokens = {}}),
                 std::invalid_argument);
}

TEST(GeneratorTest, StopTokenEndsSingleRequestWithoutExtraDecode) {
    const std::vector<std::int32_t> prompt{1, 2, 3};
    const GenerationConfig baseline_config{
        .max_new_tokens = 6,
        .temperature = 0.0F,
        .top_k = 1,
        .seed = 71,
        .kv_cache_layer_dtypes = {},
        .stop_tokens = {}};
    model::TransformerModel baseline_model(generation_config(), 73);
    const auto baseline = generate(baseline_model, prompt, baseline_config);
    ASSERT_GT(baseline.size(), prompt.size());
    const auto stop = baseline[prompt.size()];

    auto stopped_config = baseline_config;
    stopped_config.stop_tokens = {stop};
    model::TransformerModel stopped_model(generation_config(), 73);
    const auto stopped = generate(stopped_model, prompt, stopped_config);
    EXPECT_EQ(stopped.size(), prompt.size() + 1U);
    EXPECT_EQ(stopped.back(), stop);

    stopped_config.stop_tokens = {stop, stop};
    EXPECT_THROW((void)generate(stopped_model, prompt, stopped_config),
                 std::invalid_argument);
    stopped_config.stop_tokens = {
        static_cast<std::int32_t>(generation_config().vocabulary_size)};
    EXPECT_THROW((void)generate(stopped_model, prompt, stopped_config),
                 std::out_of_range);
}

TEST(GeneratorTest, StaticBatchStopRowsMatchIndependentVariableLengths) {
    auto varied_config = generation_config();
    varied_config.vocabulary_size = 32;
    varied_config.dimension = 16;
    varied_config.layers = 2;
    varied_config.heads = 4;
    varied_config.kv_heads = 2;
    varied_config.ffn_dimension = 32;
    const std::vector<std::vector<std::int32_t>> prompts{
        {1, 2, 3}, {4, 5, 6}, {7, 8, 9}, {10, 11, 12}};
    const GenerationConfig baseline_config{
        .max_new_tokens = 6,
        .temperature = 0.0F,
        .top_k = 1,
        .seed = 79,
        .kv_cache_layer_dtypes = {},
        .stop_tokens = {}};
    std::vector<std::vector<std::int32_t>> baselines;
    for (const auto& prompt : prompts) {
        model::TransformerModel independent(varied_config, 139);
        baselines.push_back(generate(independent, prompt, baseline_config));
    }
    std::int32_t selected_stop = -1;
    for (std::int32_t token = 0; token < varied_config.vocabulary_size; ++token) {
        std::vector<std::size_t> first_positions;
        for (std::size_t row = 0; row < baselines.size(); ++row) {
            const auto begin = baselines[row].begin() +
                               static_cast<std::ptrdiff_t>(prompts[row].size());
            const auto found = std::find(begin, baselines[row].end(), token);
            first_positions.push_back(
                found == baselines[row].end()
                    ? static_cast<std::size_t>(baseline_config.max_new_tokens)
                    : static_cast<std::size_t>(std::distance(begin, found)));
        }
        if (*std::min_element(first_positions.begin(), first_positions.end()) <
                static_cast<std::size_t>(baseline_config.max_new_tokens) &&
            *std::min_element(first_positions.begin(), first_positions.end()) !=
                *std::max_element(first_positions.begin(), first_positions.end())) {
            selected_stop = token;
            break;
        }
    }
    ASSERT_GE(selected_stop, 0);
    auto stopped_config = baseline_config;
    stopped_config.stop_tokens = {selected_stop};
    model::TransformerModel batched_model(varied_config, 139);
    const auto batched = generate_batch(batched_model, prompts, stopped_config);
    std::vector<std::size_t> lengths;
    for (std::size_t row = 0; row < prompts.size(); ++row) {
        model::TransformerModel independent(varied_config, 139);
        const auto expected = generate(independent, prompts[row], stopped_config);
        EXPECT_EQ(batched[row], expected) << "row=" << row;
        lengths.push_back(batched[row].size());
    }
    EXPECT_NE(*std::min_element(lengths.begin(), lengths.end()),
              *std::max_element(lengths.begin(), lengths.end()));
}

}  // namespace microllm::inference
