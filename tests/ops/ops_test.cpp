#include <cmath>
#include <stdexcept>
#include <vector>

#include <gtest/gtest.h>
#include <microllm/ops/ops.h>

namespace microllm::ops {
namespace {

void expect_near(const std::vector<float>& actual, const std::vector<float>& expected,
                 float tolerance = 1.0e-5F) {
    ASSERT_EQ(actual.size(), expected.size());
    for (std::size_t index = 0; index < actual.size(); ++index) {
        EXPECT_NEAR(actual[index], expected[index], tolerance) << "index=" << index;
    }
}

}  // namespace

TEST(CpuOpsTest, ElementwiseOpsMatchHandValues) {
    const auto left = Tensor::from_vector({1, -2, 3}, {3});
    const auto right = Tensor::from_vector({4, 5, -6}, {3});
    EXPECT_EQ(add(left, right).to_vector(), (std::vector<float>{5, 3, -3}));
    EXPECT_EQ(multiply(left, right).to_vector(), (std::vector<float>{4, -10, -18}));
    EXPECT_EQ(scale(left, 0.5F).to_vector(), (std::vector<float>{0.5F, -1, 1.5F}));
}

TEST(CpuOpsTest, BatchedMatmulMatchesHandValues) {
    const auto left = Tensor::from_vector({1, 2, 3, 4, 5, 6, 1, 0, 0, 1, 1, 1}, {2, 2, 3});
    const auto right = Tensor::from_vector({1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 6}, {2, 3, 2});
    EXPECT_EQ(matmul(left, right).shape(), (Shape{2, 2, 2}));
    EXPECT_EQ(matmul(left, right).to_vector(),
              (std::vector<float>{22, 28, 49, 64, 1, 2, 9, 12}));
}

TEST(CpuOpsTest, EmbeddingGathersRowsAndRejectsBadIndex) {
    const auto weight = Tensor::from_vector({0, 1, 2, 3, 4, 5}, {3, 2});
    const auto indices = Tensor::from_int32_vector({2, 0, 1}, {3});
    EXPECT_EQ(embedding(weight, indices).to_vector(), (std::vector<float>{4, 5, 0, 1, 2, 3}));
    EXPECT_THROW((void)embedding(weight, Tensor::from_int32_vector({3}, {1})), std::out_of_range);
}

TEST(CpuOpsTest, SoftmaxIsStableAndRowsSumToOne) {
    const auto input = Tensor::from_vector({1000, 1000, 1, 2, 3, 4}, {2, 3});
    const auto output = softmax(input).to_vector();
    EXPECT_NEAR(output[0] + output[1] + output[2], 1.0F, 1.0e-6F);
    EXPECT_NEAR(output[3] + output[4] + output[5], 1.0F, 1.0e-6F);
    EXPECT_TRUE(std::isfinite(output[0]));
}

TEST(CpuOpsTest, RmsNormMatchesManualCalculation) {
    const auto input = Tensor::from_vector({3, 4}, {1, 2});
    const auto weight = Tensor::from_vector({1, 2}, {2});
    const auto denominator = std::sqrt(12.5F + 1.0e-5F);
    expect_near(rms_norm(input, weight).to_vector(), {3.0F / denominator, 8.0F / denominator});
}

TEST(CpuOpsTest, SiluAndSwiGluMatchDefinitions) {
    const auto input = Tensor::from_vector({-1, 0, 1}, {3});
    const auto silu_values = silu(input).to_vector();
    EXPECT_NEAR(silu_values[0], -1.0F / (1.0F + std::exp(1.0F)), 1.0e-6F);
    EXPECT_EQ(silu_values[1], 0.0F);
    EXPECT_NEAR(silu_values[2], 1.0F / (1.0F + std::exp(-1.0F)), 1.0e-6F);
    expect_near(swiglu(input, Tensor::from_vector({2, 2, 2}, {3})).to_vector(),
                {2 * silu_values[0], 0, 2 * silu_values[2]});
}

TEST(CpuOpsTest, RopeLeavesPositionZeroAndRotatesPositionOne) {
    const auto input = Tensor::from_vector({1, 0, 0, 1, 1, 0, 0, 1}, {1, 2, 1, 4});
    const auto output = rope(input).to_vector();
    expect_near({output[0], output[1], output[2], output[3]}, {1, 0, 0, 1});
    EXPECT_NEAR(output[4], std::cos(1.0F), 1.0e-5F);
    EXPECT_NEAR(output[5], std::sin(1.0F), 1.0e-5F);
}

TEST(CpuOpsTest, CrossEntropyMatchesStableLogSoftmax) {
    const auto logits = Tensor::from_vector({2, 1, 0, 0, 1, 2}, {2, 3});
    const auto targets = Tensor::from_int32_vector({0, 2}, {2});
    const auto expected = std::log(std::exp(2.0F) + std::exp(1.0F) + 1.0F) - 2.0F;
    EXPECT_NEAR(cross_entropy(logits, targets).to_vector()[0], expected, 1.0e-6F);
}

TEST(CpuOpsTest, ShapeErrorsAreVisible) {
    const Tensor left({2, 3});
    const Tensor right({3, 2});
    EXPECT_THROW((void)add(left, right), std::invalid_argument);
    EXPECT_THROW((void)softmax(Tensor({2, 0})), std::invalid_argument);
    EXPECT_THROW((void)matmul(Tensor({2, 3}), Tensor({2, 4})), std::invalid_argument);
}

}  // namespace microllm::ops
