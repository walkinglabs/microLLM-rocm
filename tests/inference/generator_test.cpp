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
                                  .kv_cache_layer_dtypes = {}});
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
         .kv_cache_layer_dtypes = {}});
    EXPECT_EQ(bf16_tokens, tokens);
    model::TransformerModel explicit_policy_model(generation_config(), 47);
    const auto explicit_policy_tokens = generate(
        explicit_policy_model, {1, 2, 3},
        {.max_new_tokens = 5,
         .temperature = 0.0F,
         .top_k = 0,
         .seed = 99,
         .kv_cache_layer_dtypes = {DType::BFloat16}});
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
                     {.max_new_tokens = 10, .kv_cache_layer_dtypes = {}}),
                 std::invalid_argument);
    EXPECT_THROW((void)generate(
                     model, {8},
                     {.max_new_tokens = 0, .kv_cache_layer_dtypes = {}}),
                 std::out_of_range);
    EXPECT_THROW((void)generate(
                     model, {1, 2},
                     {.max_new_tokens = 1,
                      .kv_cache_layer_dtypes = {DType::Float32,
                                                DType::BFloat16}}),
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
            .kv_cache_layer_dtypes = {}};
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
                                                DType::BFloat16}}),
                 std::invalid_argument);
}

}  // namespace microllm::inference
