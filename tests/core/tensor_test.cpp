#include <cstdint>
#include <limits>
#include <random>
#include <stdexcept>
#include <vector>

#include <gtest/gtest.h>
#include <microllm/core/tensor.h>

namespace microllm {

TEST(ShapeTest, ComputesNumelAndContiguousStrides) {
    EXPECT_EQ(checked_numel({2, 3, 4}), 24);
    EXPECT_EQ(checked_numel({}), 1);
    EXPECT_EQ(checked_numel({2, 0, 4}), 0);
    EXPECT_EQ(contiguous_strides({2, 3, 4}), (Strides{12, 4, 1}));
    EXPECT_THROW((void)checked_numel({2, -1}), std::invalid_argument);
    EXPECT_THROW((void)checked_numel({std::numeric_limits<std::int64_t>::max(), 2}),
                 std::overflow_error);
}

TEST(TensorTest, ConstructsScalarAndZeroSizedTensor) {
    Tensor scalar(Shape{});
    Tensor zero({2, 0, 3});
    EXPECT_EQ(scalar.ndim(), 0);
    EXPECT_EQ(scalar.numel(), 1);
    EXPECT_EQ(zero.numel(), 0);
    EXPECT_TRUE(zero.empty());
    EXPECT_TRUE(zero.is_contiguous());
}

TEST(TensorTest, TransposeIsAZeroCopyViewWithCorrectValues) {
    auto source = Tensor::from_vector({0, 1, 2, 3, 4, 5}, {2, 3});
    auto transposed = source.transpose(0, 1);
    EXPECT_EQ(transposed.shape(), (Shape{3, 2}));
    EXPECT_EQ(transposed.strides(), (Strides{1, 3}));
    EXPECT_FALSE(transposed.is_contiguous());
    EXPECT_EQ(transposed.storage().data(), source.storage().data());
    EXPECT_EQ(transposed.to_vector(), (std::vector<float>{0, 3, 1, 4, 2, 5}));

    transposed.fill(9.0F);
    EXPECT_EQ(source.to_vector(), (std::vector<float>{9, 9, 9, 9, 9, 9}));
}

TEST(TensorTest, SliceSharesStorageAndHonorsStep) {
    auto source = Tensor::from_vector({0, 1, 2, 3, 4, 5, 6, 7}, {2, 4});
    auto every_other = source.slice(1, 1, 4, 2);
    EXPECT_EQ(every_other.shape(), (Shape{2, 2}));
    EXPECT_EQ(every_other.to_vector(), (std::vector<float>{1, 3, 5, 7}));
    every_other.fill(-1.0F);
    EXPECT_EQ(source.to_vector(), (std::vector<float>{0, -1, 2, -1, 4, -1, 6, -1}));
}

TEST(TensorTest, ContiguousCopiesANonContiguousView) {
    auto source = Tensor::from_vector({0, 1, 2, 3, 4, 5}, {2, 3});
    auto copy = source.transpose(0, 1).contiguous();
    EXPECT_TRUE(copy.is_contiguous());
    EXPECT_NE(copy.storage().data(), source.storage().data());
    EXPECT_EQ(copy.to_vector(), (std::vector<float>{0, 3, 1, 4, 2, 5}));
}

TEST(TensorTest, ReshapeSharesOnlyContiguousStorage) {
    auto source = Tensor::from_vector({0, 1, 2, 3, 4, 5}, {2, 3});
    auto reshaped = source.reshape({3, 2});
    EXPECT_EQ(reshaped.shape(), (Shape{3, 2}));
    EXPECT_EQ(reshaped.storage().data(), source.storage().data());
    EXPECT_THROW((void)source.transpose(0, 1).reshape({6}), std::invalid_argument);
    EXPECT_THROW((void)source.reshape({4, 2}), std::invalid_argument);
}

TEST(TensorTest, UnsqueezeAndSqueezePreserveValues) {
    auto source = Tensor::from_vector({1, 2, 3, 4, 5, 6}, {2, 3});
    auto expanded = source.unsqueeze(1);
    EXPECT_EQ(expanded.shape(), (Shape{2, 1, 3}));
    EXPECT_EQ(expanded.squeeze(1).shape(), source.shape());
    EXPECT_EQ(expanded.squeeze().to_vector(), source.to_vector());
}

TEST(TensorTest, RejectsInvalidViewsAndSlices) {
    Storage storage(4 * sizeof(float));
    EXPECT_THROW((void)Tensor::from_storage(storage, {3, 3}, {3, 1}), std::out_of_range);
    EXPECT_THROW((void)Tensor::from_storage(storage, {2, 2}, {2, -1}),
                 std::invalid_argument);
    Tensor source({2, 3});
    EXPECT_THROW((void)source.slice(1, -1, 2), std::out_of_range);
    EXPECT_THROW((void)source.slice(1, 0, 4), std::out_of_range);
    EXPECT_THROW((void)source.slice(1, 0, 2, 0), std::invalid_argument);
}

TEST(TensorTest, RandomContiguousShapesRoundTrip) {
    std::mt19937 generator(20260819U);
    std::uniform_int_distribution<int> rank_distribution(1, 4);
    std::uniform_int_distribution<int> dimension_distribution(1, 5);
    for (int sample = 0; sample < 100; ++sample) {
        Shape shape(static_cast<std::size_t>(rank_distribution(generator)));
        for (auto& dimension : shape) dimension = dimension_distribution(generator);
        const auto elements = checked_numel(shape);
        std::vector<float> values(static_cast<std::size_t>(elements));
        for (std::size_t index = 0; index < values.size(); ++index) {
            values[index] = static_cast<float>(index);
        }
        const auto tensor = Tensor::from_vector(values, shape);
        EXPECT_TRUE(tensor.is_contiguous());
        EXPECT_EQ(tensor.to_vector(), values);
    }
}

}  // namespace microllm
