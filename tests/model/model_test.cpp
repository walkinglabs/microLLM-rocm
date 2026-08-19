#include <cmath>
#include <set>
#include <string>

#include <gtest/gtest.h>
#include <microllm/model/model.h>

namespace microllm::model {
namespace {

ModelConfig tiny_config(bool gqa = true) {
    return {.vocabulary_size = 16,
            .dimension = 8,
            .layers = 1,
            .heads = 2,
            .kv_heads = gqa ? 1 : 2,
            .ffn_dimension = 16,
            .max_sequence_length = 8,
            .rope_base = 10000.0F,
            .tie_embeddings = false};
}

}  // namespace

TEST(TransformerModelTest, ConstructedParametersMatchConfigAndHaveUniqueNames) {
    TransformerModel model(tiny_config(), 7);
    EXPECT_EQ(model.parameter_count(), model.config().parameter_count());
    const auto named = model.named_parameters();
    std::set<std::string> names;
    for (const auto& [name, parameter] : named) {
        EXPECT_TRUE(names.insert(name).second);
        EXPECT_TRUE(parameter->requires_grad());
    }
}

TEST(TransformerModelTest, ForwardAndBackwardCoverEveryParameter) {
    TransformerModel model(tiny_config(), 11);
    const auto tokens = Tensor::from_int32_vector({1, 2, 3, 4, 4, 3, 2, 1}, {2, 4});
    const auto targets = Tensor::from_int32_vector({2, 3, 4, 5, 3, 2, 1, 0}, {2, 4});
    const auto logits = model.forward(tokens);
    EXPECT_EQ(logits.data().shape(), (Shape{2, 4, 16}));
    for (const auto value : logits.data().to_vector()) EXPECT_TRUE(std::isfinite(value));

    const auto loss = model.loss(tokens, targets);
    EXPECT_TRUE(std::isfinite(loss.data().to_vector()[0]));
    loss.backward();
    for (const auto& [name, parameter] : model.named_parameters()) {
        ASSERT_TRUE(parameter->has_grad()) << name;
        EXPECT_EQ(parameter->grad().shape(), parameter->data().shape()) << name;
        for (const auto value : parameter->grad().to_vector()) {
            EXPECT_TRUE(std::isfinite(value)) << name;
        }
    }
}

TEST(TransformerModelTest, CausalPrefixLogitsIgnoreFutureTokens) {
    TransformerModel model(tiny_config(false), 19);
    const auto first = Tensor::from_int32_vector({1, 2, 3, 4}, {1, 4});
    const auto second = Tensor::from_int32_vector({1, 2, 9, 10}, {1, 4});
    const auto first_logits = model.forward(first).data().to_vector();
    const auto second_logits = model.forward(second).data().to_vector();
    for (std::size_t position = 0; position < 2; ++position) {
        for (std::size_t token = 0; token < 16; ++token) {
            EXPECT_EQ(first_logits[position * 16 + token], second_logits[position * 16 + token]);
        }
    }
}

TEST(TransformerModelTest, RejectsBadTokenShapeAndLongSequence) {
    TransformerModel model(tiny_config(), 23);
    EXPECT_THROW((void)model.forward(Tensor::from_int32_vector({1, 2}, {2})),
                 std::invalid_argument);
    EXPECT_THROW((void)model.forward(Tensor::from_int32_vector(
                                      {1, 2, 3, 4, 5, 6, 7, 8, 9}, {1, 9})),
                 std::invalid_argument);
}

}  // namespace microllm::model
