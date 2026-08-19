#include <cmath>
#include <vector>

#include <gtest/gtest.h>
#include <microllm/autograd/autograd.h>

namespace microllm::autograd {

TEST(AutogradTest, AccumulatesGradientAcrossGraphBranches) {
    Value input(Tensor::from_vector({1, 2, 3}, {3}), true);
    Value coefficient(Tensor::from_vector({4, 5, 6}, {3}), true);
    const auto loss = sum(add(multiply(input, coefficient), input));
    loss.backward();
    EXPECT_EQ(input.grad().to_vector(), (std::vector<float>{5, 6, 7}));
    EXPECT_EQ(coefficient.grad().to_vector(), (std::vector<float>{1, 2, 3}));
}

TEST(AutogradTest, MatmulBackwardMatchesHandValues) {
    Value left(Tensor::from_vector({1, 2, 3, 4}, {2, 2}), true);
    Value right(Tensor::from_vector({5, 6, 7, 8}, {2, 2}), true);
    mean(matmul(left, right)).backward();
    EXPECT_EQ(left.grad().to_vector(), (std::vector<float>{2.75F, 3.75F, 2.75F, 3.75F}));
    EXPECT_EQ(right.grad().to_vector(), (std::vector<float>{1, 1, 1.5F, 1.5F}));
}

TEST(AutogradTest, FiniteDifferenceChecksMultiplyAndMean) {
    const std::vector<float> initial{0.5F, -1.25F, 2.0F};
    const auto coefficient = Tensor::from_vector({2, -3, 4}, {3});
    Value input(Tensor::from_vector(initial, {3}), true);
    mean(multiply(input, Value(coefficient))).backward();
    const auto analytical = input.grad().to_vector();

    constexpr float epsilon = 1.0e-3F;
    for (std::size_t index = 0; index < initial.size(); ++index) {
        auto plus = initial;
        auto minus = initial;
        plus[index] += epsilon;
        minus[index] -= epsilon;
        const auto plus_loss = mean(multiply(Value(Tensor::from_vector(plus, {3})),
                                             Value(coefficient))).data().to_vector()[0];
        const auto minus_loss = mean(multiply(Value(Tensor::from_vector(minus, {3})),
                                              Value(coefficient))).data().to_vector()[0];
        const auto numerical = (plus_loss - minus_loss) / (2.0F * epsilon);
        EXPECT_NEAR(analytical[index], numerical, 2.0e-4F) << "index=" << index;
    }
}

TEST(AutogradTest, RejectsImplicitGradientForNonScalarAndBadSeed) {
    Value input(Tensor::from_vector({1, 2}, {2}), true);
    EXPECT_THROW(input.backward(), std::invalid_argument);
    EXPECT_THROW(input.backward(Tensor({1})), std::invalid_argument);
}

TEST(AutogradTest, ZeroGradAndDetachAreExplicit) {
    Value input(Tensor::from_vector({1, 2}, {2}), true);
    sum(input).backward();
    ASSERT_TRUE(input.has_grad());
    input.zero_grad();
    EXPECT_FALSE(input.has_grad());
    EXPECT_FALSE(input.detach().requires_grad());
}

TEST(AutogradTest, RepeatedBackwardAccumulatesLeavesWithoutReusingIntermediateGradients) {
    Value input(Tensor::from_vector({2}, {1}), true);
    const auto loss = sum(multiply(input, input));
    loss.backward();
    loss.backward();
    EXPECT_EQ(input.grad().to_vector(), (std::vector<float>{8}));
}

}  // namespace microllm::autograd
