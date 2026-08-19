#include <vector>

#include <gtest/gtest.h>
#include <microllm/autograd/autograd.h>

namespace microllm::autograd {
namespace {

void expect_near(const Tensor& tensor, const std::vector<float>& expected,
                 float tolerance = 1.0e-6F) {
    const auto actual = tensor.to_vector();
    ASSERT_EQ(actual.size(), expected.size());
    for (std::size_t index = 0; index < actual.size(); ++index) {
        EXPECT_NEAR(actual[index], expected[index], tolerance) << "index=" << index;
    }
}

}  // namespace

TEST(GraphGradientAlignmentTest, CompositeForwardAndBothLeafGradientsMatchHandOracle) {
    Value a(Tensor::from_vector({1, 2, 3, 4}, {2, 2}), true);
    Value b(Tensor::from_vector({5, 6, 7, 8}, {2, 2}), true);
    const auto output = add(multiply(a, b), scale(a, 2.0F));
    expect_near(output.data(), {7, 16, 27, 40});
    const auto loss = mean(output);
    expect_near(loss.data(), {22.5F});
    loss.backward();
    expect_near(a.grad(), {1.75F, 2.0F, 2.25F, 2.5F});
    expect_near(b.grad(), {0.25F, 0.5F, 0.75F, 1.0F});
}

TEST(GraphGradientAlignmentTest, SharedLeafAccumulatesAndRepeatedBackwardIsStable) {
    Value input(Tensor::from_vector({1, 2, 3}, {3}), true);
    const auto loss = sum(add(multiply(input, input), input));
    loss.backward();
    expect_near(input.grad(), {3, 5, 7});
    loss.backward();
    expect_near(input.grad(), {6, 10, 14});
}

TEST(GraphGradientAlignmentTest, ViewGraphRestoresLogicalGradientOrder) {
    Value input(Tensor::from_vector({0, 1, 2, 3, 4, 5}, {2, 3}), true);
    const auto view = transpose(input, 0, 1);
    const auto packed = contiguous(view);
    const Value weights(Tensor::from_vector({1, 2, 3, 4, 5, 6}, {3, 2}));
    sum(multiply(packed, weights)).backward();
    expect_near(input.grad(), {1, 3, 5, 2, 4, 6});
}

TEST(GraphGradientAlignmentTest, RejectsBadBackwardSeedShape) {
    Value input(Tensor::from_vector({1, 2, 3}, {3}), true);
    const auto output = multiply(input, input);
    EXPECT_THROW(output.backward(), std::invalid_argument);
    EXPECT_THROW(output.backward(Tensor::from_vector({1, 2}, {2})), std::invalid_argument);
}

}  // namespace microllm::autograd
