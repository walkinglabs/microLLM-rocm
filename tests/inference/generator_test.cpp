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
                                  .seed = 99});
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
         .kv_cache_dtype = DType::BFloat16});
    EXPECT_EQ(bf16_tokens, tokens);
}

TEST(GeneratorTest, RejectsInvalidSamplingAndContext) {
    std::mt19937_64 generator(1);
    EXPECT_THROW((void)sample_token({}, 1.0F, 0, generator), std::invalid_argument);
    EXPECT_THROW((void)sample_token({1, 2}, -1.0F, 0, generator), std::invalid_argument);
    model::TransformerModel model(generation_config(), 53);
    EXPECT_THROW((void)generate(model, {}, {}), std::invalid_argument);
    EXPECT_THROW((void)generate(model, {1, 2, 3}, {.max_new_tokens = 10}),
                 std::invalid_argument);
    EXPECT_THROW((void)generate(model, {8}, {.max_new_tokens = 0}), std::out_of_range);
}

}  // namespace microllm::inference
