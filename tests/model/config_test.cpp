#include <gtest/gtest.h>
#include <microllm/model/config.h>

namespace microllm::model {

TEST(ModelConfigTest, ModelSHasExecutableParameterBudget) {
    const auto config = ModelConfig::model_s();
    EXPECT_EQ(config.head_dimension(), 64);
    EXPECT_EQ(config.parameter_count(), 15'586'176U);
    EXPECT_EQ(config.weight_bytes(4), 62'344'704U);
}

TEST(ModelConfigTest, ModelMTargetsTheSecondWeightTier) {
    const auto config = ModelConfig::model_m();
    EXPECT_EQ(config.parameter_count(), 31'334'912U);
    EXPECT_EQ(config.weight_bytes(4), 125'339'648U);
}

TEST(ModelConfigTest, GqaBudgetUsesReducedKeyValueProjection) {
    auto config = ModelConfig::model_s();
    const auto mha = config.parameter_count();
    config.kv_heads = 2;
    EXPECT_LT(config.parameter_count(), mha);
    EXPECT_EQ(config.kv_dimension(), 128);
}

TEST(ModelConfigTest, RejectsInvalidHeadAndRopeConfigurations) {
    auto config = ModelConfig::model_s();
    config.dimension = 383;
    EXPECT_THROW(config.validate(), std::invalid_argument);
    config = ModelConfig::model_s();
    config.kv_heads = 4;
    EXPECT_THROW(config.validate(), std::invalid_argument);
    config = ModelConfig::model_s();
    config.linear_precision = LinearPrecision::Float8E4M3FNUZ;
    config.fp8_activation_scale = 0.0F;
    EXPECT_THROW(config.validate(), std::invalid_argument);
}

}  // namespace microllm::model
