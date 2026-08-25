#include <algorithm>
#include <cmath>
#include <utility>
#include <vector>

#include <gtest/gtest.h>
#include <microllm/autograd/autograd.h>
#include <microllm/autograd/diagnostics.h>
#include <microllm/ops/ops.h>
#include <microllm/runtime/memory.h>

namespace microllm::autograd {

TEST(AutogradTest, AccumulatesGradientAcrossGraphBranches) {
    Value input(Tensor::from_vector({1, 2, 3}, {3}), true);
    Value coefficient(Tensor::from_vector({4, 5, 6}, {3}), true);
    const auto loss = sum(add(multiply(input, coefficient), input));
    loss.backward();
    EXPECT_EQ(input.grad().to_vector(), (std::vector<float>{5, 6, 7}));
    EXPECT_EQ(coefficient.grad().to_vector(), (std::vector<float>{1, 2, 3}));
}

TEST(AutogradTest, AddBiasBackwardReducesEveryLeadingDimension) {
    Value input(Tensor::from_vector({1, 2, 3, 4, 5, 6}, {2, 3}), true);
    Value bias(Tensor::from_vector({0.5F, -1.0F, 2.0F}, {3}), true);
    const Value seed(Tensor::from_vector({1, 2, 3, -1, -2, -3}, {2, 3}));
    sum(multiply(add_bias(input, bias), seed)).backward();
    EXPECT_EQ(input.grad().to_vector(), seed.data().to_vector());
    EXPECT_EQ(bias.grad().to_vector(), (std::vector<float>{0, 0, 0}));
}

TEST(AutogradTest, MatmulBackwardMatchesHandValues) {
    Value left(Tensor::from_vector({1, 2, 3, 4}, {2, 2}), true);
    Value right(Tensor::from_vector({5, 6, 7, 8}, {2, 2}), true);
    mean(matmul(left, right)).backward();
    EXPECT_EQ(left.grad().to_vector(), (std::vector<float>{2.75F, 3.75F, 2.75F, 3.75F}));
    EXPECT_EQ(right.grad().to_vector(), (std::vector<float>{1, 1, 1.5F, 1.5F}));
}

TEST(AutogradTest, Fp8MatmulUsesQuantizedForwardAndFp32MasterGradients) {
    Value left(Tensor::from_vector({1, -2, 3, 4, 0.5F, -0.25F}, {2, 3}), true);
    Value right(Tensor::from_vector({1, 2, 3, 4, 5, 6}, {3, 2}), true);
    const Value seed(Tensor::from_vector({1, -1, 0.5F, 2}, {2, 2}));
    const auto output = fp8_matmul(left, right, 0.025F, 0.05F);
    EXPECT_EQ(output.data().dtype(), DType::Float32);
    sum(multiply(output, seed)).backward();

    Value reference_left(left.data(), true);
    Value reference_right(right.data(), true);
    sum(multiply(matmul(reference_left, reference_right), seed)).backward();
    EXPECT_EQ(left.grad().dtype(), DType::Float32);
    EXPECT_EQ(right.grad().dtype(), DType::Float32);
    EXPECT_EQ(left.grad().to_vector(), reference_left.grad().to_vector());
    EXPECT_EQ(right.grad().to_vector(), reference_right.grad().to_vector());
}

TEST(AutogradTest, MixedFp8FormatsUseE5ActivationE4WeightAndFp32Gradients) {
    Value left(Tensor::from_vector(
        {1, -20, 300, 4, 50, -1000}, {2, 3}), true);
    Value right(Tensor::from_vector({1, 2, 3, 4, 5, 6}, {3, 2}), true);
    const auto output = fp8_matmul(
        left, right, 1000.0F / 57344.0F, 6.0F / 240.0F,
        DType::Float8E5M2FNUZ, DType::Float8E4M3FNUZ);
    const auto expected = ops::fp8_matmul(
        ops::quantize_fp8(left.data(), DType::Float8E5M2FNUZ,
                          1000.0F / 57344.0F),
        ops::quantize_fp8(right.data(), DType::Float8E4M3FNUZ,
                          6.0F / 240.0F), DType::Float32);
    EXPECT_EQ(output.data().to_vector(), expected.to_vector());
    mean(output).backward();
    EXPECT_EQ(left.grad().dtype(), DType::Float32);
    EXPECT_EQ(right.grad().dtype(), DType::Float32);
}

TEST(AutogradTest, Bf16MatmulUsesRoundedForwardAndFp32MasterGradients) {
    Value left(Tensor::from_vector({1.1F, -2.2F, 3.3F, 4.4F, 0.55F, -0.27F}, {2, 3}), true);
    Value right(Tensor::from_vector({1.2F, 2.3F, 3.4F, 4.5F, 5.6F, 6.7F}, {3, 2}), true);
    const Value seed(Tensor::from_vector({1, -1, 0.5F, 2}, {2, 2}));
    const auto output = bf16_matmul(left, right);
    sum(multiply(output, seed)).backward();

    Value reference_left(left.data(), true);
    Value reference_right(right.data(), true);
    sum(multiply(matmul(reference_left, reference_right), seed)).backward();
    EXPECT_EQ(left.grad().to_vector(), reference_left.grad().to_vector());
    EXPECT_EQ(right.grad().to_vector(), reference_right.grad().to_vector());
    EXPECT_NE(output.data().to_vector(),
              ops::matmul(left.data(), right.data()).to_vector());
}

TEST(AutogradTest, FusedSplitHalfRopeBiasMatchesComposedForwardAndGradients) {
    const auto raw = Tensor::from_vector(
        {1, 2, 3, 4, 5, 6, 7, 8,
         -1, -2, -3, -4, -5, -6, -7, -8}, {1, 2, 2, 4});
    const auto bias_data = Tensor::from_vector(
        {0.1F, 0.2F, 0.3F, 0.4F, -0.1F, -0.2F, -0.3F, -0.4F}, {8});
    const Value seed(Tensor::from_vector(
        {1, -1, 2, -2, 3, -3, 4, -4,
         -1, 1, -2, 2, -3, 3, -4, 4}, {1, 2, 2, 4}));

    Value fused_input(raw, true);
    Value fused_bias(bias_data, true);
    const auto fused = rope_split_half_bias(fused_input, fused_bias);
    sum(multiply(fused, seed)).backward();

    Value reference_input(raw, true);
    Value reference_bias(bias_data, true);
    auto flat = reshape(contiguous(transpose(reference_input, 1, 2)), {2, 8});
    auto projected = add_bias(flat, reference_bias);
    auto arranged = transpose(reshape(projected, {1, 2, 2, 4}), 1, 2);
    const auto reference = rope_split_half(arranged, 2);
    sum(multiply(reference, seed)).backward();

    EXPECT_EQ(fused.data().to_vector(), reference.data().to_vector());
    EXPECT_EQ(fused_input.grad().to_vector(), reference_input.grad().to_vector());
    EXPECT_EQ(fused_bias.grad().to_vector(), reference_bias.grad().to_vector());
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
    enable_unique_gradient_inplace_add(true);
    Value input(Tensor::from_vector({2}, {1}), true);
    const auto loss = sum(multiply(input, input));
    loss.backward();
    loss.backward();
    EXPECT_EQ(input.grad().to_vector(), (std::vector<float>{8}));
    enable_unique_gradient_inplace_add(false);
}

TEST(AutogradTest, PreparedLeafGradientTargetAccumulatesInPlace) {
    Value input(Tensor::from_vector({2, 3, 4}, {3}), true);
    auto target = Tensor::from_vector({10, 20, 30}, {3});
    const auto* address = target.data();
    input.set_grad_accumulation_target(target);
    sum(add(input, input)).backward();
    EXPECT_EQ(input.grad().data(), address);
    EXPECT_EQ(input.grad().to_vector(), (std::vector<float>{12, 22, 32}));
    sum(input).backward();
    EXPECT_EQ(input.grad().data(), address);
    EXPECT_EQ(input.grad().to_vector(), (std::vector<float>{13, 23, 33}));
}

TEST(AutogradTest, PreparedGradientTargetRejectsNonLeafAndNoncontiguousTensor) {
    Value input(Tensor::from_vector({1, 2, 3, 4}, {2, 2}), true);
    auto output = scale(input, 2.0F);
    EXPECT_THROW(
        output.set_grad_accumulation_target(Tensor({2, 2})),
        std::logic_error);
    EXPECT_THROW(
        input.set_grad_accumulation_target(Tensor({2, 2}).transpose(0, 1)),
        std::invalid_argument);
}

TEST(AutogradTest, PreparedEmbeddingGradientTargetUsesSharedStorageInPlace) {
    Value weight(Tensor::from_vector({0, 1, 2, 3, 4, 5}, {3, 2}), true);
    auto target = Tensor::from_vector({0, 0, 0, 0, 0, 0}, {3, 2});
    const auto alias = target;
    const auto* address = target.data();
    weight.set_grad_accumulation_target(std::move(target));
    sum(embedding(weight, Tensor::from_int32_vector({2, 0, 2}, {3}))).backward();
    EXPECT_EQ(weight.grad().data(), address);
    EXPECT_GT(alias.storage().use_count(), 1);
    EXPECT_EQ(weight.grad().to_vector(),
              (std::vector<float>{1, 1, 0, 0, 2, 2}));
}

TEST(AutogradTest, DirectWeightGradientProducerMatchesOrdinaryMatmulBackward) {
    const auto input_data = Tensor::from_vector({1, 2, 3, 4, 5, 6}, {2, 3});
    const auto weight_data = Tensor::from_vector({1, -1, 2, 0.5F, 3, -2}, {3, 2});
    Value reference_input(input_data, true);
    Value reference_weight(weight_data, true);
    sum(matmul(reference_input, reference_weight)).backward();

    Value direct_input(input_data, true);
    Value direct_weight(weight_data, true);
    auto target = Tensor::from_vector({10, 20, 30, 40, 50, 60}, {3, 2});
    const auto* address = target.data();
    direct_weight.set_overwrite_grad_accumulation_target(std::move(target));
    reset_direct_weight_gradient_producer_calls();
    enable_direct_weight_gradient_producer(true);
    sum(matmul(direct_input, direct_weight)).backward();
    enable_direct_weight_gradient_producer(false);

    EXPECT_EQ(direct_weight.grad().data(), address);
    EXPECT_EQ(direct_weight.grad().to_vector(), reference_weight.grad().to_vector());
    EXPECT_EQ(direct_input.grad().to_vector(), reference_input.grad().to_vector());
    EXPECT_EQ(direct_weight_gradient_producer_calls(), 1U);
    EXPECT_FALSE(direct_weight_gradient_producer_enabled());
}

TEST(AutogradTest, DirectWeightGradientRejectsPreservedNonzeroTarget) {
    const auto input_data = Tensor::from_vector({1, 2, 3, 4, 5, 6}, {2, 3});
    const auto weight_data = Tensor::from_vector({1, -1, 2, 0.5F, 3, -2}, {3, 2});
    Value weight(weight_data, true);
    weight.set_grad_accumulation_target(
        Tensor::from_vector({10, 20, 30, 40, 50, 60}, {3, 2}));
    reset_direct_weight_gradient_producer_calls();
    enable_direct_weight_gradient_producer(true);
    sum(matmul(Value(input_data), weight)).backward();
    enable_direct_weight_gradient_producer(false);
    EXPECT_EQ(direct_weight_gradient_producer_calls(), 0U);
    EXPECT_EQ(weight.grad().to_vector(),
              (std::vector<float>{15, 25, 37, 47, 59, 69}));
}

TEST(AutogradTest, OverwriteGradientTargetFallsBackWithoutReadingOldValues) {
    Value input(Tensor::from_vector({1, 2, 3}, {3}), true);
    auto target = Tensor::from_vector({100, 200, 300}, {3});
    const auto* abandoned_address = target.data();
    input.set_overwrite_grad_accumulation_target(std::move(target));
    reset_direct_weight_gradient_producer_calls();
    enable_direct_weight_gradient_producer(true);
    sum(scale(input, 2.0F)).backward();
    enable_direct_weight_gradient_producer(false);
    EXPECT_EQ(direct_weight_gradient_producer_calls(), 0U);
    EXPECT_NE(input.grad().data(), abandoned_address);
    EXPECT_EQ(input.grad().to_vector(), (std::vector<float>{2, 2, 2}));
}

TEST(AutogradTest, DirectWeightGradientWritesOnlyFirstSharedContribution) {
    const auto weight_data = Tensor::from_vector({1, -1, 2, 0.5F}, {2, 2});
    Value weight(weight_data, true);
    weight.set_overwrite_grad_accumulation_target(Tensor({2, 2}));
    const Value first(Tensor::from_vector({1, 2, 3, 4}, {2, 2}));
    const Value second(Tensor::from_vector({-1, 1, 2, -2}, {2, 2}));
    reset_direct_weight_gradient_producer_calls();
    enable_direct_weight_gradient_producer(true);
    sum(add(matmul(first, weight), matmul(second, weight))).backward();
    enable_direct_weight_gradient_producer(false);
    EXPECT_EQ(direct_weight_gradient_producer_calls(), 1U);
    EXPECT_EQ(weight.grad().to_vector(),
              (std::vector<float>{5, 5, 5, 5}));
}

TEST(AutogradTest, EmbeddingBackwardScattersAndAccumulatesRepeatedIndices) {
    Value weight(Tensor::from_vector({0, 1, 2, 3, 4, 5}, {3, 2}), true);
    const auto indices = Tensor::from_int32_vector({2, 0, 2}, {3});
    sum(embedding(weight, indices)).backward();
    EXPECT_EQ(weight.grad().to_vector(), (std::vector<float>{1, 1, 0, 0, 2, 2}));
}

TEST(AutogradTest, CrossEntropyBackwardMatchesFiniteDifference) {
    const std::vector<float> initial{2, 1, 0, 0, 1, 2};
    const auto targets = Tensor::from_int32_vector({0, 2}, {2});
    Value logits(Tensor::from_vector(initial, {2, 3}), true);
    cross_entropy(logits, targets).backward();
    const auto analytical = logits.grad().to_vector();
    constexpr float epsilon = 1.0e-3F;
    for (std::size_t index = 0; index < initial.size(); ++index) {
        auto plus = initial;
        auto minus = initial;
        plus[index] += epsilon;
        minus[index] -= epsilon;
        const auto positive = ops::cross_entropy(Tensor::from_vector(plus, {2, 3}), targets)
                                  .to_vector()[0];
        const auto negative = ops::cross_entropy(Tensor::from_vector(minus, {2, 3}), targets)
                                  .to_vector()[0];
        EXPECT_NEAR(analytical[index], (positive - negative) / (2.0F * epsilon), 2.0e-4F)
            << "index=" << index;
    }
}

TEST(AutogradTest, IgnoredCrossEntropyRowsHaveZeroLogitGradient) {
    Value logits(Tensor::from_vector({2, 1, 0, 100, -100, 0}, {2, 3}), true);
    cross_entropy(logits, Tensor::from_int32_vector({0, -100}, {2})).backward();
    const auto gradient = logits.grad().to_vector();
    EXPECT_NE(gradient[0], 0.0F);
    EXPECT_EQ(gradient[3], 0.0F);
    EXPECT_EQ(gradient[4], 0.0F);
    EXPECT_EQ(gradient[5], 0.0F);
}

TEST(AutogradTest, RmsNormAndSwiGluBackwardsMatchFiniteDifference) {
    const std::vector<float> input_values{0.5F, -1.0F, 2.0F};
    const std::vector<float> weight_values{1.0F, 0.5F, 1.5F};
    Value input(Tensor::from_vector(input_values, {1, 3}), true);
    Value weight(Tensor::from_vector(weight_values, {3}), true);
    sum(rms_norm(input, weight)).backward();
    constexpr float epsilon = 1.0e-3F;
    for (std::size_t index = 0; index < input_values.size(); ++index) {
        auto plus = input_values;
        auto minus = input_values;
        plus[index] += epsilon;
        minus[index] -= epsilon;
        const auto positive = ops::rms_norm(Tensor::from_vector(plus, {1, 3}), weight.data())
                                  .to_vector();
        const auto negative = ops::rms_norm(Tensor::from_vector(minus, {1, 3}), weight.data())
                                  .to_vector();
        float positive_sum = 0.0F;
        float negative_sum = 0.0F;
        for (const auto value : positive) positive_sum += value;
        for (const auto value : negative) negative_sum += value;
        EXPECT_NEAR(input.grad().to_vector()[index],
                    (positive_sum - negative_sum) / (2.0F * epsilon), 4.0e-4F);
    }

    Value gate(Tensor::from_vector({-1, 0.5F, 2}, {3}), true);
    Value up(Tensor::from_vector({2, 3, 4}, {3}), true);
    sum(swiglu(gate, up)).backward();
    EXPECT_NEAR(up.grad().to_vector()[1], ops::silu(gate.data()).to_vector()[1], 1.0e-6F);
}

TEST(AutogradTest, SoftmaxRopeAndViewBackwardsPreserveShapes) {
    Value input(Tensor::from_vector({1, 2, 3, 4, 5, 6, 7, 8}, {1, 2, 1, 4}), true);
    const auto transformed = rope(input);
    const auto probabilities = softmax(reshape(transformed, {2, 4}));
    sum(transpose(probabilities, 0, 1)).backward();
    EXPECT_EQ(input.grad().shape(), input.data().shape());
    for (const auto value : input.grad().to_vector()) EXPECT_NEAR(value, 0.0F, 1.0e-5F);
}

TEST(AutogradTest, RopeBackwardMatchesFiniteDifference) {
    const std::vector<float> initial{1, 0, 0, 1, 1, 0, 0, 1};
    const auto coefficients = Tensor::from_vector({1, 2, 3, 4, -1, 0.5F, 2, -3},
                                                   {1, 2, 1, 4});
    Value input(Tensor::from_vector(initial, {1, 2, 1, 4}), true);
    sum(multiply(rope(input), Value(coefficients))).backward();
    const auto analytical = input.grad().to_vector();
    constexpr float epsilon = 1.0e-3F;
    for (std::size_t index = 0; index < initial.size(); ++index) {
        auto plus = initial;
        auto minus = initial;
        plus[index] += epsilon;
        minus[index] -= epsilon;
        const auto positive_values =
            ops::multiply(ops::rope(Tensor::from_vector(plus, {1, 2, 1, 4})), coefficients)
                .to_vector();
        const auto negative_values =
            ops::multiply(ops::rope(Tensor::from_vector(minus, {1, 2, 1, 4})), coefficients)
                .to_vector();
        float positive = 0.0F;
        float negative = 0.0F;
        for (const auto value : positive_values) positive += value;
        for (const auto value : negative_values) negative += value;
        EXPECT_NEAR(analytical[index], (positive - negative) / (2.0F * epsilon), 5.0e-4F)
            << "index=" << index;
    }
}

TEST(AutogradTest, CausalSoftmaxMasksFutureAndBackpropagatesOnlyVisiblePositions) {
    Value scores(Tensor::from_vector({1, 100, 100, 1, 2, 100, 1, 2, 3}, {1, 1, 3, 3}), true);
    const auto probabilities = causal_softmax(scores);
    const auto values = probabilities.data().to_vector();
    EXPECT_EQ(values[0], 1.0F);
    EXPECT_EQ(values[1], 0.0F);
    EXPECT_EQ(values[2], 0.0F);
    EXPECT_NEAR(values[3] + values[4], 1.0F, 1.0e-6F);
    EXPECT_EQ(values[5], 0.0F);
    EXPECT_NEAR(values[6] + values[7] + values[8], 1.0F, 1.0e-6F);

    const auto weights = Value(Tensor::from_vector({1, 2, 3, 4, 5, 6, 7, 8, 9},
                                                    {1, 1, 3, 3}));
    sum(multiply(probabilities, weights)).backward();
    const auto gradient = scores.grad().to_vector();
    EXPECT_EQ(gradient[1], 0.0F);
    EXPECT_EQ(gradient[2], 0.0F);
    EXPECT_EQ(gradient[5], 0.0F);
}

TEST(AutogradTest, CausalGqaAttentionMatchesComposedGraphAndLeafGradients) {
    const auto query_data = Tensor::from_vector(
        {0.5F, -1, 1.5F, 0.25F, -0.5F, 1, 0.75F, -0.25F,
         1, 0.5F, -1, 0.25F, 0.5F, 1.25F, -0.75F, 0.5F,
         -0.25F, 0.75F, 1.5F, -1, 0.25F, -0.5F, 1, 0.5F},
        {1, 4, 3, 2});
    const auto key_data = Tensor::from_vector(
        {0.5F, 1, -0.5F, 0.25F, 1.5F, -1,
         0.75F, -0.25F, 1, 0.5F, -1, 1.25F}, {1, 2, 3, 2});
    const auto value_data = Tensor::from_vector(
        {1, 2, 3, 4, 5, 6, -1, -2, -3, -4, -5, -6}, {1, 2, 3, 2});
    const Value seed(Tensor::from_vector(
        {1, -1, 0.5F, 2, -0.5F, 1.5F, 2, 1, -1, 0.25F, 0.75F, -2,
         0.5F, 1, -1.5F, 0.25F, 2, -0.5F, 1.25F, -0.75F, 0.5F, 1.5F, -1, 2},
        {1, 4, 3, 2}));
    constexpr std::int64_t repeats = 2;
    constexpr float scale_factor = 0.5F;

    Value fused_query(query_data, true);
    Value fused_key(key_data, true);
    Value fused_value(value_data, true);
    const auto fused = causal_gqa_attention(
        fused_query, fused_key, fused_value, repeats, scale_factor);
    sum(multiply(fused, seed)).backward();

    Value reference_query(query_data, true);
    Value reference_key(key_data, true);
    Value reference_value(value_data, true);
    const auto expanded_key = repeat_interleave(reference_key, 1, repeats);
    const auto expanded_value = repeat_interleave(reference_value, 1, repeats);
    const auto scores = scale(matmul(reference_query, transpose(expanded_key, -2, -1)),
                              scale_factor);
    const auto reference = matmul(causal_softmax(scores), expanded_value);
    sum(multiply(reference, seed)).backward();

    const auto expect_close = [](const Tensor& actual, const Tensor& expected) {
        const auto left = actual.to_vector();
        const auto right = expected.to_vector();
        ASSERT_EQ(left.size(), right.size());
        for (std::size_t index = 0; index < left.size(); ++index) {
            EXPECT_NEAR(left[index], right[index], 1.0e-5F) << "index=" << index;
        }
    };
    expect_close(fused.data(), reference.data());
    expect_close(fused_query.grad(), reference_query.grad());
    expect_close(fused_key.grad(), reference_key.grad());
    expect_close(fused_value.grad(), reference_value.grad());
}

TEST(AutogradTest, ContiguousPreservesLogicalGradientOrderAfterTranspose) {
    Value input(Tensor::from_vector({1, 2, 3, 4, 5, 6}, {2, 3}), true);
    const auto reordered = contiguous(transpose(input, 0, 1));
    const auto weights = Value(Tensor::from_vector({1, 2, 3, 4, 5, 6}, {3, 2}));
    sum(multiply(reordered, weights)).backward();
    EXPECT_EQ(input.grad().to_vector(), (std::vector<float>{1, 3, 5, 2, 4, 6}));
}

TEST(AutogradTest, ReshapeBackwardMaterializesNonContiguousGradient) {
    Value input(Tensor::from_vector({1, 2, 3, 4, 5, 6}, {6}), true);
    const auto matrix = reshape(input, {2, 3});
    const auto transposed = transpose(matrix, 0, 1);
    sum(transposed).backward();
    EXPECT_EQ(input.grad().to_vector(), (std::vector<float>{1, 1, 1, 1, 1, 1}));
}

TEST(AutogradTest, RepeatInterleaveExpandsHeadsAndReducesTheirGradients) {
    Value input(Tensor::from_vector({1, 2, 3, 4}, {1, 2, 2}), true);
    const auto repeated = repeat_interleave(input, 1, 3);
    EXPECT_EQ(repeated.data().shape(), (Shape{1, 6, 2}));
    EXPECT_EQ(repeated.data().to_vector(),
              (std::vector<float>{1, 2, 1, 2, 1, 2, 3, 4, 3, 4, 3, 4}));
    sum(repeated).backward();
    EXPECT_EQ(input.grad().to_vector(), (std::vector<float>{3, 3, 3, 3}));
}

TEST(AutogradDiagnosticsTest, AttributesAddsAndMaterializationByTargetOperation) {
    reset_gradient_accumulation_diagnostics();
    enable_gradient_accumulation_diagnostics(true);
    Value shared(Tensor::from_vector({1, 2, 3, 4}, {2, 2}), true);
    sum(add(shared, shared)).backward();
    Value viewed(Tensor::from_vector({1, 2, 3, 4}, {2, 2}), true);
    sum(transpose(viewed, 0, 1)).backward();
    enable_gradient_accumulation_diagnostics(false);
    const auto snapshot = gradient_accumulation_diagnostics();
    EXPECT_EQ(snapshot.add_calls, 1U);
    EXPECT_EQ(snapshot.added_elements, 4U);
    EXPECT_EQ(snapshot.materializations, 1U);
    EXPECT_EQ(snapshot.materialized_elements, 4U);
    EXPECT_GE(snapshot.first_assignments, 2U);
    const auto leaf = std::find_if(
        snapshot.records.begin(), snapshot.records.end(), [](const auto& record) {
            return record.target_operation == "leaf" && record.shape == Shape({2, 2});
        });
    ASSERT_NE(leaf, snapshot.records.end());
    EXPECT_EQ(leaf->add_calls, 1U);
    EXPECT_EQ(leaf->materializations, 1U);
    const auto before = gradient_accumulation_diagnostics().add_calls;
    Value disabled(Tensor::from_vector({1, 2}, {2}), true);
    sum(add(disabled, disabled)).backward();
    EXPECT_EQ(gradient_accumulation_diagnostics().add_calls, before);
    reset_gradient_accumulation_diagnostics();
}

TEST(AutogradDiagnosticsTest, TiedEmbeddingUsesSparseAddAfterDenseHeadGradient) {
    const auto values = Tensor::from_vector(
        {1, 2, 3, 4, 5, 6, 7, 8}, {4, 2});
    const auto indices = Tensor::from_int32_vector({1, 3, 1}, {3});
    reset_gradient_accumulation_diagnostics();
    enable_gradient_accumulation_diagnostics(true);
    Value tied(values, true);
    const auto tied_hidden = embedding(tied, indices);
    sum(matmul(tied_hidden, tied, false, true)).backward();
    enable_gradient_accumulation_diagnostics(false);

    Value embedding_weight(values, true);
    Value head_weight(values, true);
    const auto reference_hidden = embedding(embedding_weight, indices);
    sum(matmul(reference_hidden, head_weight, false, true)).backward();
    const auto expected = ops::add(embedding_weight.grad(), head_weight.grad());
    const auto actual = tied.grad().to_vector();
    const auto reference = expected.to_vector();
    ASSERT_EQ(actual.size(), reference.size());
    for (std::size_t index = 0; index < actual.size(); ++index) {
        EXPECT_NEAR(actual[index], reference[index], 1.0e-5F) << "index=" << index;
    }
    const auto diagnostics = gradient_accumulation_diagnostics();
    EXPECT_EQ(diagnostics.sparse_embedding_add_calls, 1U);
    const auto tied_record = std::find_if(
        diagnostics.records.begin(), diagnostics.records.end(),
        [](const auto& record) {
            return record.target_operation == "leaf" &&
                   record.shape == Shape({4, 2});
        });
    ASSERT_NE(tied_record, diagnostics.records.end());
    EXPECT_EQ(tied_record->first_source, "matmul_right");
    EXPECT_EQ(tied_record->last_add_source, "embedding_backward_sparse_add");
    EXPECT_EQ(tied_record->sparse_embedding_add_calls, 1U);
    enable_tied_embedding_sparse_add(false);
    Value dense_baseline(values, true);
    const auto dense_hidden = embedding(dense_baseline, indices);
    sum(matmul(dense_hidden, dense_baseline, false, true)).backward();
    enable_tied_embedding_sparse_add(true);
    const auto dense_values = dense_baseline.grad().to_vector();
    ASSERT_EQ(dense_values.size(), actual.size());
    for (std::size_t index = 0; index < actual.size(); ++index) {
        EXPECT_NEAR(actual[index], dense_values[index], 1.0e-5F) << "index=" << index;
    }
    EXPECT_TRUE(tied_embedding_sparse_add_enabled());
    reset_gradient_accumulation_diagnostics();
}

TEST(AutogradDiagnosticsTest, DenseAddEligibilityRequiresUniqueDestinationStorage) {
    reset_gradient_accumulation_diagnostics();
    enable_gradient_accumulation_diagnostics(true);
    enable_unique_gradient_inplace_add(true);
    runtime::reset_allocation_peak(Device::cpu());
    Value unique_input(Tensor::from_vector({1, 2, 3}, {3}), true);
    sum(add(scale(unique_input, 2.0F), scale(unique_input, 3.0F))).backward();
    auto diagnostics = gradient_accumulation_diagnostics();
    const auto inplace_allocations =
        runtime::allocation_stats(Device::cpu()).allocation_calls;
    EXPECT_EQ(unique_input.grad().to_vector(),
              (std::vector<float>{5, 5, 5}));
    EXPECT_EQ(diagnostics.add_calls, 1U);
    EXPECT_EQ(diagnostics.unique_dense_add_candidates, 1U);
    EXPECT_EQ(diagnostics.unique_dense_add_executed, 1U);
    EXPECT_EQ(diagnostics.unique_dense_add_elements, 3U);
    EXPECT_EQ(diagnostics.unique_dense_add_executed_elements, 3U);

    reset_gradient_accumulation_diagnostics();
    enable_unique_gradient_inplace_add(false);
    runtime::reset_allocation_peak(Device::cpu());
    Value allocating_input(Tensor::from_vector({1, 2, 3}, {3}), true);
    sum(add(scale(allocating_input, 2.0F), scale(allocating_input, 3.0F))).backward();
    diagnostics = gradient_accumulation_diagnostics();
    const auto allocating_allocations =
        runtime::allocation_stats(Device::cpu()).allocation_calls;
    EXPECT_EQ(allocating_input.grad().to_vector(), unique_input.grad().to_vector());
    EXPECT_EQ(diagnostics.unique_dense_add_candidates, 1U);
    EXPECT_EQ(diagnostics.unique_dense_add_executed, 0U);
    EXPECT_EQ(inplace_allocations + 1U, allocating_allocations);

    reset_gradient_accumulation_diagnostics();
    enable_unique_gradient_inplace_add(true);
    Value shared_input(Tensor::from_vector({1, 2, 3}, {3}), true);
    sum(add(shared_input, shared_input)).backward();
    diagnostics = gradient_accumulation_diagnostics();
    EXPECT_EQ(shared_input.grad().to_vector(), (std::vector<float>{2, 2, 2}));
    EXPECT_EQ(diagnostics.add_calls, 1U);
    EXPECT_EQ(diagnostics.unique_dense_add_candidates, 0U);
    EXPECT_EQ(diagnostics.unique_dense_add_executed, 0U);
    EXPECT_EQ(diagnostics.unique_dense_add_elements, 0U);
    EXPECT_EQ(diagnostics.unique_dense_add_executed_elements, 0U);
    enable_unique_gradient_inplace_add(false);
    EXPECT_FALSE(unique_gradient_inplace_add_enabled());
    enable_gradient_accumulation_diagnostics(false);
    reset_gradient_accumulation_diagnostics();
}

}  // namespace microllm::autograd
