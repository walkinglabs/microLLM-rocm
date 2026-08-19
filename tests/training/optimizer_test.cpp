#include <vector>

#include <gtest/gtest.h>
#include <microllm/autograd/autograd.h>
#include <microllm/training/optimizer.h>

namespace microllm::training {
using namespace microllm::autograd;

TEST(SgdTest, ReducesAQuadraticLoss) {
    Value parameter(Tensor::from_vector({2.0F}, {1}), true);
    SGD optimizer({&parameter}, 0.1F);
    const auto before = sum(multiply(parameter, parameter)).data().to_vector()[0];
    sum(multiply(parameter, parameter)).backward();
    optimizer.step();
    optimizer.zero_grad();
    const auto after = sum(multiply(parameter, parameter)).data().to_vector()[0];
    EXPECT_LT(after, before);
    EXPECT_FALSE(parameter.has_grad());
}

TEST(AdamWTest, FirstStepMatchesBiasCorrectedHandCalculation) {
    Value parameter(Tensor::from_vector({2.0F, -3.0F}, {2}), true);
    const AdamWConfig config{.learning_rate = 0.1F,
                             .beta1 = 0.9F,
                             .beta2 = 0.999F,
                             .epsilon = 1.0e-8F,
                             .weight_decay = 0.1F};
    AdamW optimizer({&parameter}, config);
    sum(scale(parameter, 2.0F)).backward();
    optimizer.step();
    const auto values = parameter.data().to_vector();
    EXPECT_NEAR(values[0], 2.0F * 0.99F - 0.1F, 1.0e-5F);
    EXPECT_NEAR(values[1], -3.0F * 0.99F - 0.1F, 1.0e-5F);
    EXPECT_EQ(optimizer.state().step, 1U);
}

TEST(AdamWTest, RestoredStateProducesTheSameNextStep) {
    const AdamWConfig config{.learning_rate = 0.01F, .weight_decay = 0.0F};
    Value uninterrupted(Tensor::from_vector({1.0F, -2.0F}, {2}), true);
    AdamW first({&uninterrupted}, config);
    sum(scale(uninterrupted, 3.0F)).backward();
    first.step();
    first.zero_grad();

    Value restored(Tensor::from_vector(uninterrupted.data().to_vector(), {2}), true);
    AdamW second({&restored}, config);
    second.load_state(first.state());

    sum(scale(uninterrupted, -2.0F)).backward();
    sum(scale(restored, -2.0F)).backward();
    first.step();
    second.step();
    EXPECT_EQ(restored.data().to_vector(), uninterrupted.data().to_vector());
    EXPECT_EQ(second.state().step, first.state().step);
}

TEST(OptimizerTest, RejectsInvalidParametersAndState) {
    Value constant(Tensor::from_vector({1}, {1}), false);
    EXPECT_THROW((void)SGD({&constant}, 0.1F), std::invalid_argument);
    Value parameter(Tensor::from_vector({1}, {1}), true);
    AdamW optimizer({&parameter});
    EXPECT_THROW(optimizer.load_state({}), std::invalid_argument);
}

}  // namespace microllm::training
