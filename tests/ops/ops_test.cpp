#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>

#include <gtest/gtest.h>
#include <microllm/ops/ops.h>
#include <microllm/ops/low_level.h>

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

TEST(CpuOpsTest, BiasBroadcastAndReductionMatchHandValues) {
    const auto input = Tensor::from_vector({1, 2, 3, 4, 5, 6}, {2, 3});
    const auto bias = Tensor::from_vector({0.5F, -1.0F, 2.0F}, {3});
    expect_near(add_bias(input, bias).to_vector(), {1.5F, 1.0F, 5.0F, 4.5F, 4.0F, 8.0F});
    expect_near(bias_gradient(input).to_vector(), {5, 7, 9});
}

TEST(CpuOpsTest, BatchedMatmulMatchesHandValues) {
    const auto left = Tensor::from_vector({1, 2, 3, 4, 5, 6, 1, 0, 0, 1, 1, 1}, {2, 2, 3});
    const auto right = Tensor::from_vector({1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 6}, {2, 3, 2});
    EXPECT_EQ(matmul(left, right).shape(), (Shape{2, 2, 2}));
    EXPECT_EQ(matmul(left, right).to_vector(),
              (std::vector<float>{22, 28, 49, 64, 1, 2, 9, 12}));
}

TEST(CpuOpsTest, DeviceStyleCastAndMixedBf16MatmulMatchRoundedReference) {
    const auto input = Tensor::from_vector({1.1F, -2.2F, 3.3F, 4.4F}, {2, 2});
    const auto weight = Tensor::from_vector({0.5F, -1.25F, 2.0F, 0.75F}, {2, 2});
    const auto rounded_input = input.cast(DType::BFloat16).cast(DType::Float32);
    const auto rounded_weight = weight.cast(DType::BFloat16).cast(DType::Float32);
    EXPECT_EQ(cast(input, DType::BFloat16).to_vector(),
              input.cast(DType::BFloat16).to_vector());
    expect_near(bf16_matmul(input, weight.cast(DType::BFloat16)).to_vector(),
                matmul(rounded_input, rounded_weight).to_vector());
}

TEST(CpuOpsTest, TransposeAwareMatmulCoversAllOperandLayoutsWithoutViews) {
    const auto logical_left = Tensor::from_vector({1, 2, 3, 4, 5, 6}, {2, 3});
    const auto logical_right = Tensor::from_vector(
        {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}, {3, 4});
    const auto transposed_left = Tensor::from_vector({1, 4, 2, 5, 3, 6}, {3, 2});
    const auto transposed_right = Tensor::from_vector(
        {1, 5, 9, 2, 6, 10, 3, 7, 11, 4, 8, 12}, {4, 3});
    const std::vector<float> expected{38, 44, 50, 56, 83, 98, 113, 128};
    for (const auto implementation : {MatmulImplementation::Readable,
                                      MatmulImplementation::Auto}) {
        expect_near(matmul_with_implementation(logical_left, logical_right,
                                               implementation, false, false).to_vector(),
                    expected);
        expect_near(matmul_with_implementation(logical_left, transposed_right,
                                               implementation, false, true).to_vector(),
                    expected);
        expect_near(matmul_with_implementation(transposed_left, logical_right,
                                               implementation, true, false).to_vector(),
                    expected);
        expect_near(matmul_with_implementation(transposed_left, transposed_right,
                                               implementation, true, true).to_vector(),
                    expected);
    }
    EXPECT_THROW((void)matmul_with_implementation(
                     logical_left, logical_right, MatmulImplementation::Readable,
                     false, true), std::invalid_argument);
}

TEST(CpuOpsTest, CachedGqaAttentionStoresStablePrefixesAndRejectsBadContracts) {
    Tensor backing({1, 1, 4, 2});
    auto cache = Tensor::from_storage(backing.storage(), {1, 1, 1, 2},
                                      backing.strides(), 0, DType::Float32);
    const auto address = cache.storage().data();
    kv_cache_store_(cache, Tensor::from_vector({3, 4}, {1, 1, 1, 2}), 0);
    const auto one = cached_gqa_attention(
        Tensor::from_vector({1, 0, 0, 1}, {1, 2, 1, 2}), cache, cache, 2, 1.0F);
    expect_near(one.to_vector(), {3, 4, 3, 4});

    cache = Tensor::from_storage(cache.storage(), {1, 1, 2, 2},
                                 cache.strides(), 0, DType::Float32);
    kv_cache_store_(cache, Tensor::from_vector({1, 0}, {1, 1, 1, 2}), 1);
    EXPECT_EQ(cache.storage().data(), address);
    const auto two = cached_gqa_attention(
        Tensor::from_vector({1, 0, 0, 1}, {1, 2, 1, 2}), cache, cache, 2, 1.0F);
    const auto first_probability = std::exp(3.0F) / (std::exp(3.0F) + std::exp(1.0F));
    const auto second_probability = std::exp(4.0F) / (std::exp(4.0F) + 1.0F);
    expect_near(two.to_vector(),
                {first_probability * 3.0F + (1.0F - first_probability),
                 first_probability * 4.0F,
                 second_probability * 3.0F + (1.0F - second_probability),
                 second_probability * 4.0F}, 2.0e-5F);

    EXPECT_THROW(kv_cache_store_(cache,
                                 Tensor::from_vector({1, 2}, {1, 1, 1, 2}), 0),
                 std::invalid_argument);
    EXPECT_THROW((void)cached_gqa_attention(
                     Tensor::from_vector({1, 0, 0, 1}, {1, 2, 1, 2}),
                     cache, cache, 0, 1.0F), std::invalid_argument);
}

TEST(CpuOpsTest, ArgmaxUsesSmallestTieIndexAndMarksNonFiniteInput) {
    EXPECT_EQ(argmax(Tensor::from_vector({-2, 5, 5, 4}, {4})).to_int32_vector(),
              (std::vector<std::int32_t>{1}));
    EXPECT_EQ(argmax(Tensor::from_vector({-3}, {1})).shape(), (Shape{1, 1}));
    EXPECT_EQ(argmax(Tensor::from_vector(
                         {1, std::numeric_limits<float>::infinity()}, {2}))
                  .to_int32_vector(),
              (std::vector<std::int32_t>{-1}));
    EXPECT_THROW((void)argmax(Tensor({0})), std::invalid_argument);
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

TEST(CpuOpsTest, SplitHalfRopeUsesQwenPairLayout) {
    const auto input = Tensor::from_vector({1, 2, 3, 4, 5, 6, 7, 8}, {1, 2, 1, 4});
    const auto output = rope_split_half(input).to_vector();
    EXPECT_EQ(std::vector<float>(output.begin(), output.begin() + 4),
              (std::vector<float>{1, 2, 3, 4}));
    EXPECT_NE(output, rope(input).to_vector());
    const auto angle0_cos = std::cos(1.0F);
    const auto angle0_sin = std::sin(1.0F);
    EXPECT_NEAR(output[4], 5 * angle0_cos - 7 * angle0_sin, 1.0e-5F);
    EXPECT_NEAR(output[6], 5 * angle0_sin + 7 * angle0_cos, 1.0e-5F);
}

TEST(CpuOpsTest, CrossEntropyMatchesStableLogSoftmax) {
    const auto logits = Tensor::from_vector({2, 1, 0, 0, 1, 2}, {2, 3});
    const auto targets = Tensor::from_int32_vector({0, 2}, {2});
    const auto expected = std::log(std::exp(2.0F) + std::exp(1.0F) + 1.0F) - 2.0F;
    EXPECT_NEAR(cross_entropy(logits, targets).to_vector()[0], expected, 1.0e-6F);
}

TEST(CpuOpsTest, CrossEntropyIgnoresMaskedRows) {
    const auto logits = Tensor::from_vector({2, 1, 0, 100, -100, 0}, {2, 3});
    const auto targets = Tensor::from_int32_vector({0, -100}, {2});
    const auto expected = std::log(std::exp(2.0F) + std::exp(1.0F) + 1.0F) - 2.0F;
    EXPECT_NEAR(cross_entropy(logits, targets).to_vector()[0], expected, 1.0e-6F);
    EXPECT_THROW((void)cross_entropy(logits, Tensor::from_int32_vector({-100, -100}, {2})),
                 std::invalid_argument);
}

TEST(CpuOpsTest, ShapeErrorsAreVisible) {
    const Tensor left({2, 3});
    const Tensor right({3, 2});
    EXPECT_THROW((void)add(left, right), std::invalid_argument);
    EXPECT_THROW((void)softmax(Tensor({2, 0})), std::invalid_argument);
    EXPECT_THROW((void)matmul(Tensor({2, 3}), Tensor({2, 4})), std::invalid_argument);
}

TEST(CpuLowPrecisionOpsTest, ForwardFamilyMatchesRoundedFloat32Reference) {
    const auto indices = Tensor::from_int32_vector({2, 0, 2}, {3});
    const auto targets = Tensor::from_int32_vector({0, 2}, {2});
    for (const auto dtype : {DType::Float16, DType::BFloat16}) {
        const auto tolerance = dtype == DType::Float16 ? 3.0e-3F : 3.0e-2F;
        const auto left = Tensor::from_vector({1, -2, 3, 4, 0.5F, -0.25F}, {2, 3}, dtype);
        const auto right = Tensor::from_vector({4, 5, -6, 2, 1.5F, 0.25F}, {2, 3}, dtype);
        const auto left32 = left.cast(DType::Float32);
        const auto right32 = right.cast(DType::Float32);
        const auto compare = [&](const Tensor& actual, const Tensor& reference) {
            EXPECT_EQ(actual.dtype(), dtype);
            EXPECT_EQ(actual.shape(), reference.shape());
            expect_near(actual.to_vector(), reference.cast(dtype).to_vector(), tolerance);
        };
        compare(add(left, right), add(left32, right32));
        compare(multiply(left, right), multiply(left32, right32));
        compare(scale(left, -0.25F), scale(left32, -0.25F));
        compare(silu(left), silu(left32));
        compare(swiglu(left, right), swiglu(left32, right32));

        const auto mat_left = Tensor::from_vector({1, 2, 3, 4, 5, 6}, {2, 3}, dtype);
        const auto mat_right = Tensor::from_vector({1, 2, 3, 4, 5, 6}, {3, 2}, dtype);
        compare(matmul(mat_left, mat_right),
                matmul(mat_left.cast(DType::Float32), mat_right.cast(DType::Float32)));

        const auto embedding_weight = Tensor::from_vector(
            {0, 1, 2, 3, 4, 5, 6, 7}, {4, 2}, dtype);
        compare(embedding(embedding_weight, indices),
                embedding(embedding_weight.cast(DType::Float32), indices));
        compare(softmax(left), softmax(left32));
        const auto norm_weight = Tensor::from_vector({1, 0.5F, 2}, {3}, dtype);
        compare(rms_norm(left, norm_weight),
                rms_norm(left32, norm_weight.cast(DType::Float32)));
        const auto rope_input = Tensor::from_vector(
            {1, 0, 0, 1, 1, 0, 0, 1}, {1, 2, 1, 4}, dtype);
        compare(rope(rope_input), rope(rope_input.cast(DType::Float32)));

        const auto logits = Tensor::from_vector({2, 1, 0, 0, 1, 2}, {2, 3}, dtype);
        const auto loss = cross_entropy(logits, targets);
        EXPECT_EQ(loss.dtype(), DType::Float32);
        EXPECT_NEAR(loss.to_vector()[0],
                    cross_entropy(logits.cast(DType::Float32), targets).to_vector()[0],
                    tolerance);

        auto filled = Tensor({2, 3}, dtype);
        fill_(filled, -1.25F);
        EXPECT_EQ(filled.to_vector(), (std::vector<float>(6, -1.25F)));
    }
}

TEST(CpuLowPrecisionOpsTest, MixedDtypesRequireAnExplicitCast) {
    const auto fp16 = Tensor::from_vector({1, 2}, {2}, DType::Float16);
    const auto bf16 = Tensor::from_vector({1, 2}, {2}, DType::BFloat16);
    EXPECT_THROW((void)add(fp16, bf16), std::invalid_argument);
    EXPECT_THROW((void)multiply(fp16, bf16), std::invalid_argument);
    EXPECT_THROW((void)swiglu(fp16, bf16), std::invalid_argument);
    EXPECT_THROW((void)rms_norm(fp16, bf16), std::invalid_argument);
}

TEST(CpuFp8OpsTest, QuantizeDequantizeAndScaledMatmulMatchFloatReference) {
    const auto input = Tensor::from_vector(
        {-2.0F, -1.0F, -0.25F, 0.0F, 0.25F, 1.0F, 2.0F, 3.0F}, {2, 4});
    for (const auto format : {DType::Float8E4M3FNUZ, DType::Float8E5M2FNUZ}) {
        const auto quantized = quantize_fp8(input, format, 0.025F);
        EXPECT_EQ(quantized.values.dtype(), format);
        EXPECT_EQ(quantized.values.storage().num_bytes(), 8U);
        EXPECT_EQ(quantized.scale.dtype(), DType::Float32);
        const auto restored = dequantize_fp8(quantized, DType::Float32);
        const auto tolerance = format == DType::Float8E4M3FNUZ ? 0.15F : 0.25F;
        expect_near(restored.to_vector(), input.to_vector(), tolerance);
    }

    const auto left = Tensor::from_vector({1, -2, 3, 4, 0.5F, -0.25F}, {2, 3});
    const auto right = Tensor::from_vector({1, 2, 3, 4, 5, 6}, {3, 2});
    const auto fp8_output = fp8_matmul(
        quantize_fp8(left, DType::Float8E4M3FNUZ, 0.025F),
        quantize_fp8(right, DType::Float8E4M3FNUZ, 0.05F), DType::BFloat16);
    EXPECT_EQ(fp8_output.dtype(), DType::BFloat16);
    expect_near(fp8_output.to_vector(), matmul(left, right).to_vector(), 0.8F);

    EXPECT_THROW((void)quantize_fp8(input, DType::Float16, 1.0F),
                 std::invalid_argument);
    EXPECT_THROW((void)quantize_fp8(input, DType::Float8E4M3FNUZ, 0.0F),
                 std::invalid_argument);
}

TEST(LowLevelOpsTest, OperatesOnCallerOwnedCpuBuffers) {
    const Shape shape{2, 2};
    const Strides strides{2, 1};
    const float left[4]{1, 2, 3, 4};
    const float right[4]{5, 6, 7, 8};
    float output[4]{};
    const ConstTensorView left_view{left, DType::Float32, Device::cpu(), shape, strides};
    const ConstTensorView right_view{right, DType::Float32, Device::cpu(), shape, strides};
    const TensorView output_view{output, DType::Float32, Device::cpu(), shape, strides};
    add_out(output_view, left_view, right_view);
    EXPECT_EQ(std::vector<float>(output, output + 4), (std::vector<float>{6, 8, 10, 12}));
    multiply_out(output_view, left_view, right_view);
    EXPECT_EQ(std::vector<float>(output, output + 4), (std::vector<float>{5, 12, 21, 32}));
}

}  // namespace microllm::ops
