#include <bit>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <tuple>
#include <vector>

#include <gtest/gtest.h>
#include <microllm/ops/ops.h>
#include <microllm/ops/low_level.h>
#include <microllm/ops/tuning.h>

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

TEST(CpuOpsTest, InPlaceScalePreservesStorageAndChecksContract) {
    auto input = Tensor::from_vector({1, -2, 3}, {3});
    const auto* address = input.storage().data();
    scale_in_place_(input, -0.25F);
    EXPECT_EQ(input.storage().data(), address);
    EXPECT_EQ(input.to_vector(), (std::vector<float>{-0.25F, 0.5F, -0.75F}));
    EXPECT_THROW(
        scale_in_place_(input, std::numeric_limits<float>::infinity()),
        std::invalid_argument);
    auto integers = Tensor::from_int32_vector({1}, {1});
    EXPECT_THROW(scale_in_place_(integers, 2.0F), std::invalid_argument);
    auto view = Tensor::from_vector({1, 2, 3, 4}, {2, 2}).transpose(0, 1);
    EXPECT_THROW(scale_in_place_(view, 2.0F), std::invalid_argument);
}

TEST(CpuOpsTest, InPlaceAddPreservesStorageAndRejectsUnsafeContracts) {
    auto destination = Tensor::from_vector({1, -2, 3, 4}, {2, 2});
    const auto source = Tensor::from_vector({4, 5, -6, 2}, {2, 2});
    const auto* address = destination.storage().data();
    add_in_place_(destination, source);
    EXPECT_EQ(destination.storage().data(), address);
    EXPECT_EQ(destination.to_vector(), (std::vector<float>{5, 3, -3, 6}));

    add_in_place_(destination, destination);
    EXPECT_EQ(destination.to_vector(), (std::vector<float>{10, 6, -6, 12}));
    EXPECT_THROW(add_in_place_(destination, Tensor({4})), std::invalid_argument);
    EXPECT_THROW(add_in_place_(destination,
                               Tensor({2, 2}, DType::BFloat16)),
                 std::invalid_argument);
    EXPECT_THROW(add_in_place_(destination, destination.transpose(0, 1)),
                 std::invalid_argument);

    auto backing = Tensor::from_vector({1, 2, 3, 4, 5}, {5});
    auto first = backing.slice(0, 0, 4);
    const auto overlapping = backing.slice(0, 1, 5);
    EXPECT_THROW(add_in_place_(first, overlapping), std::invalid_argument);
}

TEST(CpuOpsTest, MatmulOutPreservesCallerStorageAndChecksAliases) {
    const auto left = Tensor::from_vector({1, 2, 3, 4, 5, 6}, {2, 3});
    const auto right = Tensor::from_vector({7, 8, 9, 10, 11, 12}, {3, 2});
    Tensor output({2, 2});
    const auto* address = output.storage().data();
    matmul_out_(output, left, right, MatmulImplementation::Readable);
    EXPECT_EQ(output.storage().data(), address);
    EXPECT_EQ(output.to_vector(), (std::vector<float>{58, 64, 139, 154}));

    Tensor transposed_output({3, 3});
    matmul_out_(transposed_output, left, left,
                MatmulImplementation::Readable, true, false);
    EXPECT_EQ(transposed_output.to_vector(),
              matmul_with_implementation(
                  left, left, MatmulImplementation::Readable,
                  true, false).to_vector());
    Tensor wrong_shape({4});
    EXPECT_THROW(matmul_out_(wrong_shape, left, right,
                             MatmulImplementation::Readable),
                 std::invalid_argument);
    Tensor wrong_dtype({2, 2}, DType::BFloat16);
    EXPECT_THROW(matmul_out_(wrong_dtype, left, right,
                             MatmulImplementation::Readable),
                 std::invalid_argument);
    auto alias = Tensor::from_vector({1, 2, 3, 4}, {2, 2});
    const auto identity = Tensor::from_vector({1, 0, 0, 1}, {2, 2});
    EXPECT_THROW(matmul_out_(alias, alias, identity,
                             MatmulImplementation::Readable),
                 std::invalid_argument);
}

TEST(CpuOpsTest, MatmulWeightGradientOutMatchesReferenceAndPreservesStorage) {
    const auto input = Tensor::from_vector({1, 2, 3, 4, 5, 6}, {2, 3});
    const auto output_gradient = Tensor::from_vector({1, -1, 0.5F, 2}, {2, 2});
    Tensor weight_gradient({3, 2});
    const auto* address = weight_gradient.storage().data();
    matmul_weight_gradient_out_(
        weight_gradient, input, output_gradient,
        MatmulImplementation::Readable);
    EXPECT_EQ(weight_gradient.storage().data(), address);
    EXPECT_EQ(weight_gradient.to_vector(),
              matmul_with_implementation(
                  input, output_gradient, MatmulImplementation::Readable,
                  true, false).to_vector());
    EXPECT_THROW(
        matmul_weight_gradient_out_(
            weight_gradient, input.reshape({1, 2, 3}), output_gradient,
            MatmulImplementation::Readable),
        std::invalid_argument);
    EXPECT_THROW(
        matmul_weight_gradient_out_(
            weight_gradient, input, Tensor({3, 2}),
            MatmulImplementation::Readable),
        std::invalid_argument);
}

TEST(CpuOpsTest, EmbeddingBackwardAddAccumulatesDuplicateRowsInCallerStorage) {
    auto weight_gradient = Tensor::from_vector(
        {10, 20, 30, 40, 50, 60, 70, 80}, {4, 2});
    const auto* address = weight_gradient.storage().data();
    const auto gradient = Tensor::from_vector({1, 2, 3, 4, 5, 6}, {3, 2});
    const auto indices = Tensor::from_int32_vector({1, 3, 1}, {3});
    embedding_backward_add_(weight_gradient, gradient, indices);
    EXPECT_EQ(weight_gradient.storage().data(), address);
    EXPECT_EQ(weight_gradient.to_vector(),
              (std::vector<float>{10, 20, 36, 48, 50, 60, 73, 84}));
    EXPECT_THROW(embedding_backward_add_(
                     weight_gradient, gradient,
                     Tensor::from_int32_vector({1, 3}, {2})),
                 std::invalid_argument);
}

TEST(CpuOpsTest, CastOutAndTransposePreserveCallerStorage) {
    const auto input = Tensor::from_vector({1, 2, 3, 4, 5, 6}, {2, 3},
                                           DType::BFloat16);
    Tensor casted({2, 3});
    Tensor transposed({3, 2});
    const auto* cast_address = casted.storage().data();
    const auto* transpose_address = transposed.storage().data();
    cast_out_(input, casted);
    cast_transpose_2d_out_(input, transposed);
    EXPECT_EQ(casted.storage().data(), cast_address);
    EXPECT_EQ(transposed.storage().data(), transpose_address);
    EXPECT_EQ(casted.to_vector(), (std::vector<float>{1, 2, 3, 4, 5, 6}));
    EXPECT_EQ(transposed.to_vector(), (std::vector<float>{1, 4, 2, 5, 3, 6}));
    Tensor bad_cast({6});
    Tensor bad_transpose({2, 3});
    EXPECT_THROW(cast_out_(input, bad_cast), std::invalid_argument);
    EXPECT_THROW(cast_transpose_2d_out_(input, bad_transpose), std::invalid_argument);
}

TEST(CpuOpsTest, BiasBroadcastAndReductionMatchHandValues) {
    const auto input = Tensor::from_vector({1, 2, 3, 4, 5, 6}, {2, 3});
    const auto bias = Tensor::from_vector({0.5F, -1.0F, 2.0F}, {3});
    expect_near(add_bias(input, bias).to_vector(), {1.5F, 1.0F, 5.0F, 4.5F, 4.0F, 8.0F});
    const auto bf16_input = input.cast(DType::BFloat16);
    EXPECT_EQ(add_bias_bf16(bf16_input, bias).to_vector(),
              add_bias(bf16_input.cast(DType::Float32), bias)
                  .cast(DType::BFloat16).to_vector());
    expect_near(bias_gradient(input).to_vector(), {5, 7, 9});
    expect_near(bias_gradient_with_implementation(
                    input, BiasGradientImplementation::ScalarColumns).to_vector(),
                {5, 7, 9});
    EXPECT_THROW((void)bias_gradient_with_implementation(
                     input, BiasGradientImplementation::CooperativeRows),
                 std::invalid_argument);
}

TEST(CpuOpsTest, FusedSplitHalfRopeBiasMatchesComposedProjectionPath) {
    const auto flat = Tensor::from_vector(
        {1, 2, 3, 4, 5, 6, 7, 8,
         -1, -2, -3, -4, -5, -6, -7, -8}, {2, 8});
    const auto bias = Tensor::from_vector(
        {0.1F, 0.2F, 0.3F, 0.4F, -0.1F, -0.2F, -0.3F, -0.4F}, {8});
    const auto input = flat.reshape({1, 2, 2, 4}).transpose(1, 2).contiguous();
    const auto composed = rope_split_half(
        add_bias(flat, bias).reshape({1, 2, 2, 4}).transpose(1, 2).contiguous(), 2);
    expect_near(rope_split_half_bias(input, bias).to_vector(), composed.to_vector());
    EXPECT_THROW((void)rope_split_half_bias(input, Tensor({4})), std::invalid_argument);
}

TEST(CpuOpsTest, BthdFusedSplitHalfRopeBiasMatchesLayoutMaterialization) {
    const auto input = Tensor::from_vector(
        {1, 2, 3, 4, 5, 6, 7, 8,
         -1, -2, -3, -4, -5, -6, -7, -8,
         0.5F, 1.5F, 2.5F, 3.5F, 4.5F, 5.5F, 6.5F, 7.5F,
         -0.5F, -1.5F, -2.5F, -3.5F, -4.5F, -5.5F, -6.5F, -7.5F},
        {1, 4, 2, 4});
    const auto bias = Tensor::from_vector(
        {0.1F, 0.2F, 0.3F, 0.4F, -0.1F, -0.2F, -0.3F, -0.4F}, {8});
    const auto expected = rope_split_half_bias(
        input.transpose(1, 2).contiguous(), bias, 3, 5000.0F);
    const auto actual = rope_split_half_bias_bthd(input, bias, 3, 5000.0F);
    EXPECT_EQ(actual.shape(), (Shape{1, 2, 4, 4}));
    expect_near(actual.to_vector(), expected.to_vector());
    const auto rounded_input = input.cast(DType::BFloat16);
    const auto rounded_expected = rope_split_half_bias(
        rounded_input.cast(DType::Float32).transpose(1, 2).contiguous(),
        bias, 3, 5000.0F);
    const auto rounded_actual = rope_split_half_bias_bthd(
        rounded_input, bias, 3, 5000.0F);
    EXPECT_EQ(rounded_actual.dtype(), DType::Float32);
    expect_near(rounded_actual.to_vector(), rounded_expected.to_vector());
    const auto direct_bf16 = rope_split_half_bias_bthd_bf16(
        rounded_input, bias, 3, 5000.0F);
    EXPECT_EQ(direct_bf16.dtype(), DType::BFloat16);
    EXPECT_EQ(direct_bf16.to_vector(),
              rounded_expected.cast(DType::BFloat16).to_vector());

    const auto gradient = Tensor::from_vector(
        {1, -1, 2, -2, 3, -3, 4, -4,
         -1, 1, -2, 2, -3, 3, -4, 4,
         0.25F, 0.5F, 0.75F, 1, -0.25F, -0.5F, -0.75F, -1,
         2, 1, 0, -1, -2, -1, 0, 1},
        {1, 2, 4, 4});
    const auto expected_backward = rope_split_half_backward(
        gradient, 2, 3, 5000.0F).transpose(1, 2).contiguous();
    const auto actual_backward =
        rope_split_half_bias_bthd_backward(gradient, 3, 5000.0F);
    EXPECT_EQ(actual_backward.shape(), input.shape());
    expect_near(actual_backward.to_vector(), expected_backward.to_vector());

    EXPECT_THROW((void)rope_split_half_bias_bthd(input, Tensor({4})),
                 std::invalid_argument);
    EXPECT_THROW((void)rope_split_half_bias_bthd(
                     input.cast(DType::Float16), bias),
                 std::invalid_argument);
    EXPECT_THROW((void)rope_split_half_bias_bthd_backward(Tensor({1, 2, 3, 3})),
                 std::invalid_argument);
}

TEST(CpuOpsTest, FusedResidualRmsNormReturnsBothComposedOutputs) {
    const auto left = Tensor::from_vector({1, 2, 3, -1, -2, -3}, {2, 3});
    const auto right = Tensor::from_vector({0.5F, -0.5F, 1, 2, 1, 0}, {2, 3});
    const auto weight = Tensor::from_vector({1, 0.5F, 2}, {3});
    const auto expected_sum = add(left, right);
    const auto expected_norm = rms_norm(expected_sum, weight);
    const auto actual = add_rms_norm(left, right, weight);
    expect_near(actual.first.to_vector(), expected_sum.to_vector());
    expect_near(actual.second.to_vector(), expected_norm.to_vector());
    EXPECT_THROW((void)add_rms_norm(left, Tensor({3}), weight), std::invalid_argument);
}

TEST(CpuOpsTest, BatchedMatmulMatchesHandValues) {
    const auto left = Tensor::from_vector({1, 2, 3, 4, 5, 6, 1, 0, 0, 1, 1, 1}, {2, 2, 3});
    const auto right = Tensor::from_vector({1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 6}, {2, 3, 2});
    EXPECT_EQ(matmul(left, right).shape(), (Shape{2, 2, 2}));
    EXPECT_EQ(matmul(left, right).to_vector(),
              (std::vector<float>{22, 28, 49, 64, 1, 2, 9, 12}));
}

TEST(CpuOpsTest, ScaledMatmulMatchesComposedReferenceWithTranspose) {
    const auto left = Tensor::from_vector({1, 4, 2, 5, 3, 6}, {3, 2});
    const auto right = Tensor::from_vector({1, 2, 3, 4, 5, 6}, {3, 2});
    const auto expected = scale(
        matmul_with_implementation(
            left, right, MatmulImplementation::Readable, true, false),
        -0.25F);
    const auto actual = matmul_scaled_with_implementation(
        left, right, -0.25F, MatmulImplementation::Readable, true, false);
    expect_near(actual.to_vector(), expected.to_vector());
    EXPECT_THROW((void)matmul_scaled_with_implementation(
                     left, right, std::numeric_limits<float>::infinity(),
                     MatmulImplementation::Readable, true, false),
                 std::invalid_argument);
    enable_attention_gemm_scale_fusion(false);
    EXPECT_FALSE(attention_gemm_scale_fusion_enabled());
    enable_attention_gemm_scale_fusion(true);
    EXPECT_TRUE(attention_gemm_scale_fusion_enabled());
    enable_attention_gemm_scale_fusion(false);
}

TEST(CpuOpsTest, MatmulTuningKeyCapturesLayoutModeWorkspaceAndEnvironment) {
    const Tensor left({3, 2});
    const Tensor right({4, 3});
    OpContext context;
    context.mode = OpMode::Inference;
    context.workspace_bytes = 8192;
    const auto key = make_matmul_tuning_key(
        left, right, true, true, context);
    EXPECT_EQ(key.rows, 2);
    EXPECT_EQ(key.inner, 3);
    EXPECT_EQ(key.columns, 4);
    EXPECT_EQ(key.dtype, DType::Float32);
    EXPECT_TRUE(key.transpose_left);
    EXPECT_TRUE(key.transpose_right);
    EXPECT_EQ(key.left_strides, (Strides{2, 1}));
    EXPECT_EQ(key.right_strides, (Strides{3, 1}));
    EXPECT_EQ(key.architecture, "host");
    EXPECT_EQ(key.hip_runtime_version, 0);
    EXPECT_EQ(key.hip_driver_version, 0);
    EXPECT_EQ(key.hipblaslt_version, 0);
    EXPECT_EQ(key.mode, OpMode::Inference);
    EXPECT_EQ(key.workspace_limit, 8192U);

    clear_matmul_implementation_registry();
    register_matmul_implementation(key, MatmulImplementation::Readable);
    EXPECT_EQ(matmul_registered_implementation_count(), 1U);
    EXPECT_THROW(register_matmul_implementation(
                     key, MatmulImplementation::Auto),
                 std::invalid_argument);
    auto incomplete = key;
    incomplete.architecture.clear();
    EXPECT_THROW(register_matmul_implementation(
                     incomplete, MatmulImplementation::Readable),
                 std::invalid_argument);
    clear_matmul_implementation_registry();
    EXPECT_EQ(matmul_registered_implementation_count(), 0U);

    EXPECT_THROW((void)make_matmul_tuning_key(
                     left.transpose(0, 1), right, false, true, context),
                 std::invalid_argument);
}

TEST(CpuOpsTest, Fp32SolutionKeyFlattensExactBatchedDescriptorWithoutAllocation) {
    OpContext context;
    context.mode = OpMode::Inference;
    context.fp32_solution_scope =
        Fp32SolutionScope::PrefillQueryProjection;
    const auto key = make_fp32_matmul_solution_key(
        {2, 14, 512, 64}, {2, 14, 512, 64}, Device::cpu(),
        false, true, context, 0.125F);
    EXPECT_EQ(key.batches, 28);
    EXPECT_EQ(key.left_rows, 512);
    EXPECT_EQ(key.left_columns, 64);
    EXPECT_EQ(key.right_rows, 512);
    EXPECT_EQ(key.right_columns, 64);
    EXPECT_EQ(key.output_rows, 512);
    EXPECT_EQ(key.output_columns, 512);
    EXPECT_EQ(key.left_batch_stride, 32768);
    EXPECT_EQ(key.right_batch_stride, 32768);
    EXPECT_EQ(key.output_batch_stride, 262144);
    EXPECT_FALSE(key.transpose_left);
    EXPECT_TRUE(key.transpose_right);
    EXPECT_EQ(key.alpha_bits, std::bit_cast<std::uint32_t>(0.125F));
    EXPECT_EQ(key.architecture, "host");
    EXPECT_EQ(key.hipblaslt_version, 0);
    EXPECT_EQ(key.mode, OpMode::Inference);
    EXPECT_EQ(key.solution_scope,
              Fp32SolutionScope::PrefillQueryProjection);
    EXPECT_EQ(key.workspace_limit, 0U);
    auto general_context = context;
    general_context.fp32_solution_scope = Fp32SolutionScope::General;
    const auto general = make_fp32_matmul_solution_key(
        {2, 14, 512, 64}, {2, 14, 512, 64}, Device::cpu(),
        false, true, general_context, 0.125F);
    EXPECT_NE(general, key);
    auto attention_qk_context = context;
    attention_qk_context.fp32_solution_scope =
        Fp32SolutionScope::PrefillAttentionQk;
    auto attention_pv_context = context;
    attention_pv_context.fp32_solution_scope =
        Fp32SolutionScope::PrefillAttentionPv;
    const auto attention_qk = make_fp32_matmul_solution_key(
        {2, 14, 512, 64}, {2, 14, 512, 64}, Device::cpu(),
        false, true, attention_qk_context, 0.125F);
    const auto attention_pv = make_fp32_matmul_solution_key(
        {2, 14, 512, 512}, {2, 14, 512, 64}, Device::cpu(),
        false, false, attention_pv_context);
    auto attention_output_context = context;
    attention_output_context.fp32_solution_scope =
        Fp32SolutionScope::PrefillAttentionOutputProjection;
    const auto attention_output = make_fp32_matmul_solution_key(
        {4096, 1536}, {1536, 1536}, Device::cpu(),
        false, false, attention_output_context);
    EXPECT_EQ(attention_qk.solution_scope,
              Fp32SolutionScope::PrefillAttentionQk);
    EXPECT_EQ(attention_pv.solution_scope,
              Fp32SolutionScope::PrefillAttentionPv);
    EXPECT_EQ(attention_output.solution_scope,
              Fp32SolutionScope::PrefillAttentionOutputProjection);
    EXPECT_NE(attention_qk, key);
    EXPECT_NE(attention_pv, general);
    EXPECT_NE(attention_output, key);

    clear_fp32_matmul_solution_registry();
    EXPECT_EQ(fp32_matmul_solution_stats().registered_entries, 0U);
    EXPECT_THROW(register_fp32_matmul_solution(key, 1), std::exception);
    EXPECT_EQ(fp32_matmul_solution_stats().registered_entries, 0U);
    EXPECT_THROW((void)make_fp32_matmul_solution_key(
                     {1, 4, 8, 16}, {2, 4, 8, 16}, Device::cpu()),
                 std::invalid_argument);
    EXPECT_THROW((void)make_fp32_matmul_solution_key(
                     {1, 4, 8, 16}, {1, 4, 7, 15}, Device::cpu(),
                     false, true),
                 std::invalid_argument);
}

TEST(CpuOpsTest, Bf16GroupedQkvKeyIsShapeAndEnvironmentExact) {
    const auto key = make_bf16_grouped_qkv_key(
        512, 896, 896, 128, 128, Device::cpu());
    EXPECT_EQ(key.rows, 512);
    EXPECT_EQ(key.inner, 896);
    EXPECT_EQ(key.query_columns, 896);
    EXPECT_EQ(key.key_columns, 128);
    EXPECT_EQ(key.value_columns, 128);
    EXPECT_EQ(key.architecture, "host");
    EXPECT_EQ(key.hip_runtime_version, 0);
    EXPECT_EQ(key.hip_driver_version, 0);
    EXPECT_EQ(key.hipblaslt_version, 0);
    clear_bf16_grouped_qkv_registry();
    EXPECT_EQ(bf16_grouped_qkv_stats().registered_entries, 0U);
    EXPECT_THROW(register_bf16_grouped_qkv_algorithm(key, 64699),
                 std::exception);
    EXPECT_THROW((void)make_bf16_grouped_qkv_key(
                     0, 896, 896, 128, 128, Device::hip()),
                 std::exception);
}

TEST(CpuOpsTest, Bf16GroupedGateUpKeyIsShapeAndEnvironmentExact) {
    const auto key = make_bf16_grouped_gate_up_key(
        512, 896, 4864, Device::cpu());
    EXPECT_EQ(key.rows, 512);
    EXPECT_EQ(key.inner, 896);
    EXPECT_EQ(key.columns, 4864);
    EXPECT_EQ(key.architecture, "host");
    EXPECT_EQ(key.hip_runtime_version, 0);
    EXPECT_EQ(key.hip_driver_version, 0);
    EXPECT_EQ(key.hipblaslt_version, 0);
    clear_bf16_grouped_gate_up_registry();
    EXPECT_EQ(bf16_grouped_gate_up_stats().registered_entries, 0U);
    EXPECT_THROW(register_bf16_grouped_gate_up_algorithm(key, 65168),
                 std::exception);
    EXPECT_THROW((void)make_bf16_grouped_gate_up_key(
                     0, 896, 4864, Device::hip()),
                 std::exception);
}

TEST(CpuOpsTest, MatmulTuningCacheRoundTripsAndRejectsStaleCorruptData) {
    const auto directory = std::filesystem::temp_directory_path();
    const auto cache = directory / "microllm-matmul-tuning-cache-test.jsonl";
    const auto second = directory / "microllm-matmul-tuning-cache-second.jsonl";
    const auto corrupt = directory / "microllm-matmul-tuning-cache-corrupt.jsonl";
    const auto duplicate = directory / "microllm-matmul-tuning-cache-duplicate.jsonl";
    const Tensor left({2, 3});
    const Tensor right({3, 4});
    const auto key = make_matmul_tuning_key(left, right);
    clear_matmul_implementation_registry();
    register_matmul_implementation(key, MatmulImplementation::Readable);
    save_matmul_tuning_cache(cache);
    std::ifstream first_input(cache);
    const std::string first_payload{
        std::istreambuf_iterator<char>(first_input),
        std::istreambuf_iterator<char>()};
    EXPECT_NE(first_payload.find("\"schema_version\":1"), std::string::npos);
    EXPECT_NE(first_payload.find("\"architecture\":\"host\""),
              std::string::npos);

    clear_matmul_implementation_registry();
    const auto loaded = load_matmul_tuning_cache(cache, Device::cpu());
    EXPECT_EQ(loaded.parsed_entries, 1U);
    EXPECT_EQ(loaded.loaded_entries, 1U);
    EXPECT_EQ(loaded.stale_entries, 0U);
    EXPECT_EQ(matmul_registered_implementation_count(), 1U);
    save_matmul_tuning_cache(second);
    std::ifstream second_input(second);
    const std::string second_payload{
        std::istreambuf_iterator<char>(second_input),
        std::istreambuf_iterator<char>()};
    EXPECT_EQ(second_payload, first_payload);

    {
        std::ofstream output(corrupt);
        output << "{\"schema_version\":2,\"kind\":"
                  "\"microllm_matmul_tuning_cache\"}\n";
    }
    EXPECT_THROW((void)load_matmul_tuning_cache(corrupt, Device::cpu()),
                 std::runtime_error);
    EXPECT_EQ(matmul_registered_implementation_count(), 1U)
        << "a corrupt file must not partially replace the live registry";
    auto malformed_bool = first_payload;
    const auto bool_position = malformed_bool.find("\"transpose_left\":false");
    ASSERT_NE(bool_position, std::string::npos);
    malformed_bool.replace(
        bool_position, std::string("\"transpose_left\":false").size(),
        "\"transpose_left\":falsejunk");
    {
        std::ofstream output(corrupt);
        output << malformed_bool;
    }
    EXPECT_THROW((void)load_matmul_tuning_cache(corrupt, Device::cpu()),
                 std::runtime_error);
    EXPECT_EQ(matmul_registered_implementation_count(), 1U);

    clear_matmul_implementation_registry();
    auto stale_key = key;
    stale_key.architecture = "stale-gfx";
    register_matmul_implementation(stale_key, MatmulImplementation::Readable);
    save_matmul_tuning_cache(cache);
    std::ifstream stale_input(cache);
    std::string header;
    std::string entry;
    ASSERT_TRUE(static_cast<bool>(std::getline(stale_input, header)));
    ASSERT_TRUE(static_cast<bool>(std::getline(stale_input, entry)));
    clear_matmul_implementation_registry();
    const auto stale = load_matmul_tuning_cache(cache, Device::cpu());
    EXPECT_EQ(stale.parsed_entries, 1U);
    EXPECT_EQ(stale.loaded_entries, 0U);
    EXPECT_EQ(stale.stale_entries, 1U);
    EXPECT_EQ(matmul_registered_implementation_count(), 0U);

    register_matmul_implementation(key, MatmulImplementation::Readable);
    {
        std::ofstream output(duplicate);
        output << header << '\n' << entry << '\n' << entry << '\n';
    }
    EXPECT_THROW((void)load_matmul_tuning_cache(duplicate, Device::cpu()),
                 std::runtime_error);
    EXPECT_EQ(matmul_registered_implementation_count(), 1U)
        << "duplicate rejection must be transactional";

    clear_matmul_implementation_registry();
    std::error_code ignored;
    std::filesystem::remove(cache, ignored);
    std::filesystem::remove(second, ignored);
    std::filesystem::remove(corrupt, ignored);
    std::filesystem::remove(duplicate, ignored);
}

TEST(CpuOpsTest, MatmulAutotuneRejectsCpuAndAbstractCandidateLists) {
    const Tensor left({2, 3});
    const Tensor right({3, 4});
    EXPECT_THROW((void)autotune_matmul(left, right), std::invalid_argument);
    MatmulAutotuneOptions automatic;
    automatic.candidates = {MatmulImplementation::Auto};
    EXPECT_THROW((void)autotune_matmul(left, right, false, false, automatic),
                 std::invalid_argument);
    automatic.candidates = {
        MatmulImplementation::Readable, MatmulImplementation::Readable};
    EXPECT_THROW((void)autotune_matmul(left, right, false, false, automatic),
                 std::invalid_argument);
}

TEST(CpuOpsTest, AdamWTuningKeyAndPersistentCacheAreExactAndTransactional) {
    const auto directory = std::filesystem::temp_directory_path();
    const auto cache = directory / "microllm-adamw-tuning-cache-test.jsonl";
    const auto corrupt = directory / "microllm-adamw-tuning-cache-corrupt.jsonl";
    Tensor parameter({17});
    Tensor gradient({17});
    Tensor first({17});
    Tensor second({17});
    Tensor mirror({17}, DType::BFloat16);
    OpContext context;
    context.mode = OpMode::Training;
    const auto key = make_adamw_tuning_key(
        parameter, gradient, first, second, &mirror, context);
    EXPECT_EQ(key.elements, 17);
    EXPECT_EQ(key.parameter_dtype, DType::Float32);
    EXPECT_TRUE(key.bf16_mirror);
    EXPECT_EQ(key.architecture, "host");
    EXPECT_EQ(key.hip_runtime_version, 0);
    EXPECT_EQ(key.hip_driver_version, 0);
    EXPECT_EQ(key.mode, OpMode::Training);

    clear_adamw_implementation_registry();
    register_adamw_implementation(key, AdamWImplementation::Scalar);
    EXPECT_EQ(adamw_registered_implementation_count(), 1U);
    EXPECT_THROW(register_adamw_implementation(key, AdamWImplementation::Auto),
                 std::invalid_argument);
    auto incomplete = key;
    incomplete.architecture.clear();
    EXPECT_THROW(register_adamw_implementation(
                     incomplete, AdamWImplementation::Scalar),
                 std::invalid_argument);
    save_adamw_tuning_cache(cache);
    std::ifstream cache_input(cache);
    const std::string cache_payload{
        std::istreambuf_iterator<char>(cache_input),
        std::istreambuf_iterator<char>()};
    EXPECT_NE(cache_payload.find("\"kind\":\"microllm_adamw_tuning_cache\""),
              std::string::npos);
    clear_adamw_implementation_registry();
    const auto loaded = load_adamw_tuning_cache(cache, Device::cpu());
    EXPECT_EQ(loaded.parsed_entries, 1U);
    EXPECT_EQ(loaded.loaded_entries, 1U);
    EXPECT_EQ(loaded.stale_entries, 0U);
    EXPECT_EQ(adamw_registered_implementation_count(), 1U);

    {
        std::ofstream output(corrupt);
        output << "{\"schema_version\":2,\"kind\":"
                  "\"microllm_adamw_tuning_cache\"}\n";
    }
    EXPECT_THROW((void)load_adamw_tuning_cache(corrupt, Device::cpu()),
                 std::runtime_error);
    EXPECT_EQ(adamw_registered_implementation_count(), 1U)
        << "a corrupt cache must not replace the live registry";

    auto malformed_bool = cache_payload;
    const auto bool_position = malformed_bool.find("\"bf16_mirror\":true");
    ASSERT_NE(bool_position, std::string::npos);
    malformed_bool.replace(
        bool_position, std::string("\"bf16_mirror\":true").size(),
        "\"bf16_mirror\":truejunk");
    {
        std::ofstream output(corrupt);
        output << malformed_bool;
    }
    EXPECT_THROW((void)load_adamw_tuning_cache(corrupt, Device::cpu()),
                 std::runtime_error);
    EXPECT_EQ(adamw_registered_implementation_count(), 1U);

    std::istringstream cache_lines(cache_payload);
    std::string cache_header;
    std::string cache_entry;
    ASSERT_TRUE(static_cast<bool>(std::getline(cache_lines, cache_header)));
    ASSERT_TRUE(static_cast<bool>(std::getline(cache_lines, cache_entry)));
    {
        std::ofstream output(corrupt);
        output << cache_header << '\n' << cache_entry << '\n'
               << cache_entry << '\n';
    }
    EXPECT_THROW((void)load_adamw_tuning_cache(corrupt, Device::cpu()),
                 std::runtime_error);
    EXPECT_EQ(adamw_registered_implementation_count(), 1U)
        << "duplicate rejection must be transactional";

    auto unsafe_vector = cache_payload;
    const auto alignment_position =
        unsafe_vector.find("\"parameter_aligned16\":true");
    const auto implementation_position =
        unsafe_vector.find("\"implementation\":\"scalar\"");
    ASSERT_NE(alignment_position, std::string::npos);
    ASSERT_NE(implementation_position, std::string::npos);
    unsafe_vector.replace(
        implementation_position, std::string("\"implementation\":\"scalar\"").size(),
        "\"implementation\":\"vectorized\"");
    unsafe_vector.replace(
        alignment_position, std::string("\"parameter_aligned16\":true").size(),
        "\"parameter_aligned16\":false");
    {
        std::ofstream output(corrupt);
        output << unsafe_vector;
    }
    EXPECT_THROW((void)load_adamw_tuning_cache(corrupt, Device::cpu()),
                 std::invalid_argument);
    EXPECT_EQ(adamw_registered_implementation_count(), 1U)
        << "an unsafe cached choice must not replace the live registry";

    clear_adamw_implementation_registry();
    auto stale_key = key;
    stale_key.architecture = "stale-host";
    register_adamw_implementation(stale_key, AdamWImplementation::Scalar);
    save_adamw_tuning_cache(cache);
    clear_adamw_implementation_registry();
    const auto stale = load_adamw_tuning_cache(cache, Device::cpu());
    EXPECT_EQ(stale.parsed_entries, 1U);
    EXPECT_EQ(stale.loaded_entries, 0U);
    EXPECT_EQ(stale.stale_entries, 1U);
    EXPECT_EQ(adamw_registered_implementation_count(), 0U);

    EXPECT_EQ(choose_adamw_implementation(
                  parameter, gradient, first, second, &mirror, context),
              AdamWImplementation::Scalar);
    EXPECT_THROW((void)autotune_adamw(
                     parameter, gradient, first, second, &mirror),
                 std::invalid_argument);
    AdamWAutotuneOptions invalid;
    invalid.candidates = {AdamWImplementation::Auto};
    EXPECT_THROW((void)autotune_adamw(
                     parameter, gradient, first, second, &mirror, invalid),
                 std::invalid_argument);

    std::error_code ignored;
    std::filesystem::remove(cache, ignored);
    std::filesystem::remove(corrupt, ignored);
}

TEST(CpuOpsTest, TransposeAwareBatchedReadableMatchesMaterializedReference) {
    const auto left = Tensor::from_vector(
        {1, 2, 3, 4, 5, 6, 6, 5, 4, 3, 2, 1}, {2, 2, 3});
    const auto right = Tensor::from_vector(
        {1, 2, 3, 4, 5, 6, 6, 5, 4, 3, 2, 1}, {2, 3, 2});
    const auto expected = matmul(left, right).to_vector();
    const auto left_t = left.transpose(-2, -1).contiguous();
    const auto right_t = right.transpose(-2, -1).contiguous();
    for (const auto& test : {
             std::tuple<const Tensor*, const Tensor*, bool, bool>{
                 &left, &right, false, false},
             std::tuple<const Tensor*, const Tensor*, bool, bool>{
                 &left, &right_t, false, true},
             std::tuple<const Tensor*, const Tensor*, bool, bool>{
                 &left_t, &right, true, false},
             std::tuple<const Tensor*, const Tensor*, bool, bool>{
                 &left_t, &right_t, true, true}}) {
        EXPECT_EQ(matmul_with_implementation(
                      *std::get<0>(test), *std::get<1>(test),
                      MatmulImplementation::Readable,
                      std::get<2>(test), std::get<3>(test)).to_vector(),
                  expected);
    }
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
    const auto bf16_output = bf16_matmul_output(
        input.cast(DType::BFloat16), weight.cast(DType::BFloat16), DType::BFloat16);
    EXPECT_EQ(bf16_output.dtype(), DType::BFloat16);
    expect_near(bf16_output.cast(DType::Float32).to_vector(),
                matmul(rounded_input, rounded_weight).cast(DType::BFloat16)
                    .cast(DType::Float32).to_vector());
    auto input_bf16 = input.cast(DType::BFloat16);
    const auto weight_bf16 = weight.cast(DType::BFloat16);
    Tensor caller_output({2, 2});
    Tensor caller_fallback({2, 2}, DType::BFloat16);
    bf16_matmul_output_out_(caller_output, input_bf16, weight_bf16,
                            caller_fallback);
    expect_near(caller_output.to_vector(),
                bf16_matmul_output(input_bf16, weight_bf16,
                                   DType::Float32).to_vector(), 0.0F);
    EXPECT_THROW(
        bf16_matmul_output_out_(caller_output, input_bf16, weight_bf16,
                                caller_output),
        std::invalid_argument);
    EXPECT_THROW(
        bf16_matmul_output_out_(caller_output, input_bf16, weight_bf16,
                                input_bf16),
        std::invalid_argument);
}

TEST(CpuOpsTest, Bf16WeightGradientRoundsBothOperandsAndKeepsFp32Output) {
    const auto input = Tensor::from_vector(
        {1.1F, -2.2F, 3.3F, 4.4F, 0.55F, -0.27F}, {2, 3});
    const auto gradient = Tensor::from_vector({1, -1, 0.5F, 2}, {2, 2});
    const auto rounded_input = input.cast(DType::BFloat16).cast(DType::Float32);
    const auto rounded_gradient = gradient.cast(DType::BFloat16).cast(DType::Float32);
    const auto expected = matmul_with_implementation(
        rounded_input, rounded_gradient,
        MatmulImplementation::Readable, true, false);
    const auto actual = bf16_weight_gradient(input, gradient);
    EXPECT_EQ(actual.dtype(), DType::Float32);
    EXPECT_EQ(actual.shape(), (Shape{3, 2}));
    EXPECT_EQ(actual.to_vector(), expected.to_vector());
    EXPECT_THROW(
        (void)bf16_weight_gradient(input, Tensor({3, 2})),
        std::invalid_argument);
    EXPECT_THROW(
        (void)bf16_weight_gradient(
            input.cast(DType::BFloat16), gradient),
        std::invalid_argument);
}

TEST(CpuOpsTest, Bf16FfnKeepsIntermediateActivationsLowPrecision) {
    const auto input = Tensor::from_vector(
        {1.1F, -0.5F, 0.25F, 2.0F, -1.0F, 0.75F}, {2, 3});
    const auto gate = Tensor::from_vector(
        {0.5F, -1.0F, 0.25F, 0.75F, 1.5F, -0.5F,
         -0.25F, 0.5F, 1.0F, -1.25F, 0.125F, 0.875F},
        {3, 4}, DType::BFloat16);
    const auto up = Tensor::from_vector(
        {1.0F, 0.5F, -0.75F, 0.25F, -0.5F, 1.25F,
         0.625F, -1.0F, 0.75F, -0.25F, 1.5F, 0.5F},
        {3, 4}, DType::BFloat16);
    const auto down = Tensor::from_vector(
        {0.25F, -0.5F, 1.0F, 0.75F, -1.25F, 0.5F, 0.125F, -0.875F},
        {4, 2}, DType::BFloat16);
    const auto rounded_input = input.cast(DType::BFloat16);
    const auto gate_output = bf16_matmul_output(
        rounded_input, gate, DType::BFloat16);
    const auto up_output = bf16_matmul_output(
        rounded_input, up, DType::BFloat16);
    const auto expected = bf16_matmul_output(
        swiglu(gate_output, up_output), down, DType::Float32);
    const auto diagnostics = bf16_ffn_diagnostics(input, gate, up, down);
    const auto actual = diagnostics.output;
    EXPECT_EQ(diagnostics.input_bf16.dtype(), DType::BFloat16);
    EXPECT_EQ(diagnostics.gate.dtype(), DType::BFloat16);
    EXPECT_EQ(diagnostics.up.dtype(), DType::BFloat16);
    EXPECT_EQ(diagnostics.activated.dtype(), DType::BFloat16);
    EXPECT_EQ(actual.dtype(), DType::Float32);
    EXPECT_EQ(actual.shape(), (Shape{2, 2}));
    expect_near(actual.to_vector(), expected.to_vector(), 0.0F);
    expect_near(bf16_ffn(input, gate, up, down).to_vector(),
                actual.to_vector(), 0.0F);

    Bf16FfnWorkspace workspace{
        .input_bf16 = Tensor({2, 3}, DType::BFloat16),
        .gate = Tensor({2, 4}, DType::BFloat16),
        .up = Tensor({2, 4}, DType::BFloat16),
        .activated = Tensor({2, 4}, DType::BFloat16),
        .output_fallback_bf16 = Tensor({2, 2}, DType::BFloat16)};
    Tensor caller_output({2, 2});
    bf16_ffn_out_(caller_output, workspace, input, gate, up, down);
    expect_near(caller_output.to_vector(), actual.to_vector(), 0.0F);
    EXPECT_THROW(
        bf16_ffn_out_(caller_output, workspace, input, gate, up,
                      Tensor({8}, DType::BFloat16)),
        std::invalid_argument);
    workspace.activated = workspace.gate;
    EXPECT_THROW(
        bf16_ffn_out_(caller_output, workspace, input, gate, up, down),
        std::invalid_argument);

    EXPECT_THROW((void)bf16_ffn(input.cast(DType::BFloat16), gate, up, down),
                 std::invalid_argument);
    EXPECT_THROW((void)bf16_ffn(input, gate, up, Tensor({3, 2}, DType::BFloat16)),
                 std::invalid_argument);
}

TEST(CpuOpsTest, Bf16QkvProjectionCastsSharedInputOnceAndKeepsFp32Outputs) {
    const auto input = Tensor::from_vector({1.1F, -0.5F, 0.25F, 2.0F, -1.0F, 0.75F},
                                           {2, 3});
    const auto query = Tensor::from_vector({0.5F, -1, 0.25F, 0.75F, -0.5F, 1.25F},
                                           {3, 2}, DType::BFloat16);
    const auto key = Tensor::from_vector({0.25F, 0.5F, -0.75F},
                                         {3, 1}, DType::BFloat16);
    const auto value = Tensor::from_vector({-1.0F, 0.125F, 0.875F},
                                           {3, 1}, DType::BFloat16);
    const auto output = bf16_qkv_projection(input, query, key, value);
    expect_near(output.first.to_vector(), bf16_matmul(input, query).to_vector(), 0.0F);
    expect_near(output.second.to_vector(), bf16_matmul(input, key).to_vector(), 0.0F);
    expect_near(output.third.to_vector(), bf16_matmul(input, value).to_vector(), 0.0F);
    EXPECT_EQ(output.first.dtype(), DType::Float32);
    Bf16QkvWorkspace workspace{
        .input_bf16 = Tensor({2, 3}, DType::BFloat16),
        .query_fallback_bf16 = Tensor({2, 2}, DType::BFloat16),
        .key_fallback_bf16 = Tensor({2, 1}, DType::BFloat16),
        .value_fallback_bf16 = Tensor({2, 1}, DType::BFloat16)};
    Tensor query_output({2, 2});
    Tensor key_output({2, 1});
    Tensor value_output({2, 1});
    const auto retained_query_key = bf16_qkv_projection_out_(
        query_output, key_output, value_output, workspace,
        input, query, key, value, {}, true);
    EXPECT_FALSE(retained_query_key);
    EXPECT_EQ(query_output.to_vector(), output.first.to_vector());
    EXPECT_EQ(key_output.to_vector(), output.second.to_vector());
    EXPECT_EQ(value_output.to_vector(), output.third.to_vector());
    EXPECT_THROW(
        bf16_qkv_projection_out_(
            query_output, key_output, value_output, workspace,
            input, query, key, value, {}, false, true),
        std::invalid_argument);
    workspace.value_fallback_bf16 = workspace.key_fallback_bf16;
    EXPECT_THROW(
        bf16_qkv_projection_out_(
            query_output, key_output, value_output, workspace,
            input, query, key, value),
        std::invalid_argument);
    EXPECT_THROW((void)bf16_qkv_projection(input, query, key,
                                            Tensor({4, 1}, DType::BFloat16)),
                 std::invalid_argument);
}

TEST(CpuOpsTest, Bf16PlanCacheIsEmptyWithoutHipblaslt) {
    if (hipblaslt_available()) GTEST_SKIP() << "HIP build exercises cache statistics";
    clear_bf16_plan_cache();
    const auto stats = bf16_plan_cache_stats();
    EXPECT_EQ(stats.entries, 0U);
    EXPECT_EQ(stats.hits, 0U);
    EXPECT_EQ(stats.misses, 0U);
}

TEST(CpuOpsTest, CausalGqaAttentionMatchesComposedForwardAndBackward) {
    const auto query = Tensor::from_vector(
        {0.5F, -1, 1.5F, 0.25F, -0.5F, 1, 0.75F, -0.25F,
         1, 0.5F, -1, 0.25F, 0.5F, 1.25F, -0.75F, 0.5F,
         -0.25F, 0.75F, 1.5F, -1, 0.25F, -0.5F, 1, 0.5F},
        {1, 4, 3, 2});
    const auto key = Tensor::from_vector(
        {0.5F, 1, -0.5F, 0.25F, 1.5F, -1,
         0.75F, -0.25F, 1, 0.5F, -1, 1.25F}, {1, 2, 3, 2});
    const auto value = Tensor::from_vector(
        {1, 2, 3, 4, 5, 6, -1, -2, -3, -4, -5, -6}, {1, 2, 3, 2});
    const auto gradient = Tensor::from_vector(
        {1, -1, 0.5F, 2, -0.5F, 1.5F, 2, 1, -1, 0.25F, 0.75F, -2,
         0.5F, 1, -1.5F, 0.25F, 2, -0.5F, 1.25F, -0.75F, 0.5F, 1.5F, -1, 2},
        {1, 4, 3, 2});
    constexpr std::int64_t repeats = 2;
    constexpr float scale_factor = 0.5F;
    const auto expanded_key = repeat_interleave(key, 1, repeats);
    const auto expanded_value = repeat_interleave(value, 1, repeats);
    const auto scores = scale(matmul(
        query, expanded_key.transpose(-2, -1).contiguous()), scale_factor);
    const auto probabilities = causal_softmax(scores);
    const auto expected = matmul(probabilities, expanded_value);
    expect_near(causal_gqa_attention(query, key, value, repeats, scale_factor).to_vector(),
                expected.to_vector(), 1.0e-6F);
    const auto diagnostics = causal_gqa_attention_diagnostics(
        query, key, value, repeats, scale_factor);
    EXPECT_EQ(diagnostics.scaled_query.shape(), query.shape());
    EXPECT_EQ(diagnostics.scores.shape(), (Shape{1, 4, 3, 3}));
    EXPECT_EQ(diagnostics.probabilities.shape(), diagnostics.scores.shape());
    expect_near(diagnostics.scaled_query.to_vector(),
                scale(query, scale_factor).to_vector(), 1.0e-6F);
    expect_near(diagnostics.scores.to_vector(), scores.to_vector(), 1.0e-6F);
    expect_near(diagnostics.probabilities.to_vector(),
                probabilities.to_vector(), 1.0e-6F);
    expect_near(diagnostics.output.to_vector(), expected.to_vector(), 1.0e-6F);
    CausalGqaAttentionWorkspace workspace{
        .scaled_query = Tensor(query.shape()),
        .expanded_kv = Tensor(query.shape()),
        .probabilities = Tensor({1, 4, 3, 3})};
    Tensor caller_output(query.shape());
    causal_gqa_attention_out_(
        caller_output, workspace, query, key, value, repeats,
        scale_factor);
    expect_near(caller_output.to_vector(), expected.to_vector(), 1.0e-6F);
    workspace.expanded_kv = workspace.scaled_query;
    EXPECT_THROW(
        causal_gqa_attention_out_(
            caller_output, workspace, query, key, value, repeats,
            scale_factor),
        std::invalid_argument);

    const auto probability_gradient = matmul(
        gradient, expanded_value.transpose(-2, -1).contiguous());
    const auto score_gradient = scale(
        causal_softmax_backward(probabilities, probability_gradient), scale_factor);
    const auto expected_query = matmul(score_gradient, expanded_key);
    const auto expected_key = repeat_interleave_backward(
        matmul(score_gradient.transpose(-2, -1).contiguous(), query),
        key.shape(), 1, repeats);
    const auto expected_value = repeat_interleave_backward(
        matmul(probabilities.transpose(-2, -1).contiguous(), gradient),
        value.shape(), 1, repeats);
    const auto actual_backward = causal_gqa_attention_backward(
        query, key, value, gradient, repeats, scale_factor);
    expect_near(actual_backward.first.to_vector(), expected_query.to_vector(), 1.0e-6F);
    expect_near(actual_backward.second.to_vector(), expected_key.to_vector(), 1.0e-6F);
    expect_near(actual_backward.third.to_vector(), expected_value.to_vector(), 1.0e-6F);
    EXPECT_THROW((void)causal_gqa_attention(query, key, value, 3, scale_factor),
                 std::invalid_argument);
    EXPECT_THROW((void)causal_gqa_attention_diagnostics(
                     query, key, value, 3, scale_factor),
                 std::invalid_argument);
}

TEST(CpuOpsTest, AttentionProbabilityValueWritesInterleavedBthdLayout) {
    const auto probabilities = Tensor::from_vector(
        {1, 0, 0, 0.25F, 0.75F, 0, 0.1F, 0.2F, 0.7F,
         1, 0, 0, 0.5F, 0.5F, 0, 0.2F, 0.3F, 0.5F,
         1, 0, 0, 0.4F, 0.6F, 0, 0.3F, 0.3F, 0.4F,
         1, 0, 0, 0.6F, 0.4F, 0, 0.2F, 0.5F, 0.3F},
        {2, 2, 3, 3});
    const auto value = Tensor::from_vector(
        {1, 2, 10, 20, 3, 4, 30, 40, 5, 6, 50, 60,
         -1, -2, -10, -20, -3, -4, -30, -40, -5, -6, -50, -60},
        {2, 3, 2, 2});
    const auto expected = matmul(
        probabilities, value.transpose(1, 2).contiguous())
                              .transpose(1, 2).contiguous();
    const auto actual = attention_probability_value_bthd(probabilities, value);
    EXPECT_EQ(actual.shape(), value.shape());
    expect_near(actual.to_vector(), expected.to_vector());
    EXPECT_THROW((void)attention_probability_value_bthd(
                     probabilities, Tensor({2, 3, 1, 2})),
                 std::invalid_argument);
}

TEST(CpuOpsTest, AttentionProbabilityValueGqaBroadcastMatchesRepeatedReference) {
    const auto probabilities = Tensor::from_vector(
        {1, 0, 0.25F, 0.75F, 1, 0, 0.5F, 0.5F,
         1, 0, 0.75F, 0.25F, 1, 0, 0.1F, 0.9F},
        {1, 4, 2, 2});
    const auto value = Tensor::from_vector(
        {1, 2, 10, 20, 3, 4, 30, 40}, {1, 2, 2, 2});
    const auto expected = attention_probability_value_bthd(
        probabilities, repeat_interleave(value, 2, 2));
    const auto actual = attention_probability_value_gqa_bthd(
        probabilities, value, 2);
    EXPECT_EQ(actual.shape(), (Shape{1, 2, 4, 2}));
    expect_near(actual.to_vector(), expected.to_vector());
    const auto output_gradient = Tensor::from_vector(
        {1, -1, 2, -2, 3, -3, 4, -4,
         5, -5, 6, -6, 7, -7, 8, -8}, {1, 2, 4, 2});
    const auto expected_gradient = attention_probability_gradient_bthd(
        output_gradient, repeat_interleave(value, 2, 2));
    const auto actual_gradient = attention_probability_gradient_gqa_bthd(
        output_gradient, value, 2);
    expect_near(actual_gradient.to_vector(), expected_gradient.to_vector());
    EXPECT_THROW((void)attention_probability_value_gqa_bthd(
                     probabilities, Tensor({1, 2, 1, 2}), 2),
                 std::invalid_argument);
    EXPECT_THROW((void)attention_probability_gradient_gqa_bthd(
                     output_gradient, Tensor({1, 2, 1, 2}), 2),
                 std::invalid_argument);
    enable_attention_gqa_value_broadcast(false);
    enable_attention_gqa_forward_value_broadcast(false);
    EXPECT_FALSE(attention_gqa_value_broadcast_enabled());
    EXPECT_FALSE(attention_gqa_forward_value_broadcast_enabled());
    enable_attention_gqa_value_broadcast(true);
    EXPECT_TRUE(attention_gqa_value_broadcast_enabled());
    enable_attention_gqa_value_broadcast(false);
    enable_attention_gqa_forward_value_broadcast(true);
    EXPECT_TRUE(attention_gqa_forward_value_broadcast_enabled());
    enable_attention_gqa_forward_value_broadcast(false);
    enable_inference_bthd_attention(false);
    EXPECT_FALSE(inference_bthd_attention_enabled());
    enable_inference_bthd_attention(true);
    EXPECT_TRUE(inference_bthd_attention_enabled());
    enable_inference_bthd_attention(false);
    enable_inference_bthd_online_attention(false);
    EXPECT_FALSE(inference_bthd_online_attention_enabled());
    enable_inference_bthd_online_attention(true);
    EXPECT_TRUE(inference_bthd_online_attention_enabled());
    enable_inference_bthd_online_attention(false);
    enable_inference_bthd_bf16_qk(false);
    EXPECT_FALSE(inference_bthd_bf16_qk_enabled());
    enable_inference_bthd_bf16_qk(true);
    EXPECT_TRUE(inference_bthd_bf16_qk_enabled());
    enable_inference_bthd_bf16_qk(false);
}

TEST(CpuOpsTest, AttentionLayoutPlanCacheIsUnavailableWithoutHipblaslt) {
    if (hipblaslt_available()) GTEST_SKIP() << "hipBLASLt build has the real cache";
    clear_attention_layout_plan_cache();
    enable_attention_layout_plan_cache(true);
    EXPECT_FALSE(attention_layout_plan_cache_enabled());
    const auto stats = attention_layout_plan_cache_stats();
    EXPECT_EQ(stats.entries, 0U);
    EXPECT_EQ(stats.hits, 0U);
    EXPECT_EQ(stats.misses, 0U);
}

TEST(CpuOpsTest, AttentionBthdBackwardPrimitivesAndGqaMatchMaterializedReference) {
    const auto probabilities = Tensor::from_vector(
        {1, 0, 0, 0.25F, 0.75F, 0, 0.1F, 0.2F, 0.7F,
         1, 0, 0, 0.5F, 0.5F, 0, 0.2F, 0.3F, 0.5F},
        {1, 2, 3, 3});
    const auto value = Tensor::from_vector(
        {1, 2, 10, 20, 3, 4, 30, 40, 5, 6, 50, 60},
        {1, 3, 2, 2});
    const auto output_gradient = Tensor::from_vector(
        {1, -1, 0.5F, -0.5F, 2, -2, 1.5F, -1.5F, 3, -3, 2.5F, -2.5F},
        {1, 3, 2, 2});
    const auto gradient_bhtd = output_gradient.transpose(1, 2).contiguous();
    const auto value_bhtd = value.transpose(1, 2).contiguous();
    const auto expected_probability_gradient = matmul_with_implementation(
        gradient_bhtd, value_bhtd, MatmulImplementation::Readable,
        false, true);
    const auto expected_value_gradient = matmul_with_implementation(
        probabilities, gradient_bhtd, MatmulImplementation::Readable,
        true, false).transpose(1, 2).contiguous();
    expect_near(attention_probability_gradient_bthd(
                    output_gradient, value).to_vector(),
                expected_probability_gradient.to_vector());
    expect_near(attention_value_gradient_bthd(
                    probabilities, output_gradient).to_vector(),
                expected_value_gradient.to_vector());

    const auto query = Tensor::from_vector(
        {0.5F, -1, 1.5F, 0.25F, -0.5F, 1,
         0.75F, -0.25F, 1, 0.5F, -1, 0.25F}, {1, 2, 3, 2});
    const auto key = Tensor::from_vector(
        {0.5F, 1, -0.5F, 0.25F, 1.5F, -1}, {1, 1, 3, 2});
    const auto gqa_value_bhtd = Tensor::from_vector(
        {1, 2, 3, 4, 5, 6}, {1, 1, 3, 2});
    const auto gqa_value = gqa_value_bhtd.transpose(1, 2).contiguous();
    const auto seed = Tensor::from_vector(
        {1, -1, 0.5F, -0.5F, 2, -2,
         1.5F, -1.5F, 3, -3, 2.5F, -2.5F}, {1, 3, 2, 2});
    const auto expected_output = causal_gqa_attention(
        query, key, gqa_value_bhtd, 2, 0.5F).transpose(1, 2).contiguous();
    const auto expected_gradients = causal_gqa_attention_backward(
        query, key, gqa_value_bhtd, seed.transpose(1, 2).contiguous(),
        2, 0.5F);
    const auto actual_saved = causal_gqa_attention_bthd_saved(
        query, key, gqa_value, 2, 0.5F);
    const auto actual_gradients = causal_gqa_attention_bthd_backward_saved(
        query, key, gqa_value, actual_saved.second, seed, 2, 0.5F);
    expect_near(actual_saved.first.to_vector(), expected_output.to_vector());
    expect_near(actual_gradients.first.to_vector(),
                expected_gradients.first.to_vector());
    expect_near(actual_gradients.second.to_vector(),
                expected_gradients.second.to_vector());
    expect_near(actual_gradients.third.to_vector(),
                expected_gradients.third.transpose(1, 2).contiguous().to_vector());
    const auto recomputed = causal_gqa_attention_bthd_backward(
        query, key, gqa_value, seed, 2, 0.5F);
    expect_near(recomputed.first.to_vector(), actual_gradients.first.to_vector());
    expect_near(recomputed.second.to_vector(), actual_gradients.second.to_vector());
    expect_near(recomputed.third.to_vector(), actual_gradients.third.to_vector());
    EXPECT_THROW((void)attention_probability_gradient_bthd(
                     output_gradient, Tensor({1, 3, 1, 2})),
                 std::invalid_argument);
    EXPECT_THROW((void)attention_value_gradient_bthd(
                     probabilities, Tensor({1, 2, 3, 2})),
                 std::invalid_argument);
}

TEST(CpuOpsTest, OnlineCausalGqaBthdFallsBackFromBf16AndChecksContract) {
    const auto query = Tensor::from_vector(
        {0.5F, -1, 1.5F, 0.25F, -0.5F, 1,
         0.75F, -0.25F, 1, 0.5F, -1, 0.25F},
        {1, 2, 3, 2}, DType::BFloat16);
    const auto key = Tensor::from_vector(
        {0.5F, 1, -0.5F, 0.25F, 1.5F, -1},
        {1, 1, 3, 2}, DType::BFloat16);
    const auto value = Tensor::from_vector(
        {1, 2, 3, 4, 5, 6}, {1, 3, 1, 2}, DType::BFloat16);
    const auto expected = causal_gqa_attention_bthd(
        query.cast(DType::Float32), key.cast(DType::Float32),
        value.cast(DType::Float32), 2, 0.5F);
    const auto actual = online_causal_gqa_attention_bthd(
        query, key, value, 2, 0.5F);
    EXPECT_EQ(actual.dtype(), DType::Float32);
    EXPECT_EQ(actual.shape(), (Shape{1, 3, 2, 2}));
    expect_near(actual.to_vector(), expected.to_vector());
    EXPECT_THROW((void)online_causal_gqa_attention_bthd(
                     query.cast(DType::Float32), key, value, 2, 0.5F),
                 std::invalid_argument);
    EXPECT_THROW((void)online_causal_gqa_attention_bthd(
                     query, key, Tensor({1, 3, 2, 2}, DType::BFloat16),
                     2, 0.5F),
                 std::invalid_argument);
}

TEST(CpuOpsTest, PairedGqaRepeatMatchesSeparateKeyValuePaths) {
    const auto key = Tensor::from_vector(
        {1, 2, 3, 4, 10, 20, 30, 40}, {1, 2, 2, 2});
    const auto value = Tensor::from_vector(
        {5, 6, 50, 60, 7, 8, 70, 80}, {1, 2, 2, 2});
    const auto actual = repeat_gqa_kv_bthd(key, value, 2);
    expect_near(actual.first.to_vector(),
                repeat_interleave(key, 1, 2).to_vector());
    expect_near(actual.second.to_vector(),
                repeat_interleave(value, 2, 2).to_vector());
    const auto key_gradient = Tensor::from_vector(
        {1, 2, 3, 4, 5, 6, 7, 8,
         9, 10, 11, 12, 13, 14, 15, 16}, {1, 4, 2, 2});
    const auto value_gradient = Tensor::from_vector(
        {16, 15, 14, 13, 12, 11, 10, 9,
         8, 7, 6, 5, 4, 3, 2, 1}, {1, 2, 4, 2});
    const auto gradients = repeat_gqa_kv_bthd_backward(
        key_gradient, value_gradient, 2);
    expect_near(gradients.first.to_vector(),
                repeat_interleave_backward(
                    key_gradient, key.shape(), 1, 2).to_vector());
    expect_near(gradients.second.to_vector(),
                repeat_interleave_backward(
                    value_gradient, value.shape(), 2, 2).to_vector());
    EXPECT_THROW((void)repeat_gqa_kv_bthd(key, Tensor({1, 2, 1, 2}), 2),
                 std::invalid_argument);
    EXPECT_THROW((void)repeat_gqa_kv_bthd_backward(
                     key_gradient, Tensor({1, 2, 3, 2}), 2),
                 std::invalid_argument);
    enable_attention_paired_gqa_repeat(false);
    EXPECT_FALSE(attention_paired_gqa_repeat_enabled());
    enable_attention_paired_gqa_repeat(true);
    EXPECT_TRUE(attention_paired_gqa_repeat_enabled());
    enable_attention_paired_gqa_repeat(false);
}

TEST(CpuOpsTest, Bf16RepeatInterleaveFusesCastWithoutChangingValues) {
    const auto input = Tensor::from_vector(
        {1.003F, -2.011F, 3.019F, 4.027F, -5.035F, 6.043F,
         7.051F, -8.059F},
        {1, 2, 2, 2}, DType::BFloat16);
    const auto expected = repeat_interleave(
        input.cast(DType::Float32), 2, 3);
    const auto actual = repeat_interleave_bf16_to_float(input, 2, 3);
    EXPECT_EQ(actual.dtype(), DType::Float32);
    EXPECT_EQ(actual.shape(), (Shape{1, 2, 6, 2}));
    EXPECT_EQ(actual.to_vector(), expected.to_vector());
    EXPECT_THROW((void)repeat_interleave_bf16_to_float(
                     input.cast(DType::Float32), 2, 3),
                 std::invalid_argument);
    EXPECT_THROW((void)repeat_interleave_bf16_to_float(input, 2, 0),
                 std::invalid_argument);
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

TEST(CpuOpsTest, CachedGqaAttentionScoresExposeEveryScaledDotWithoutMutation) {
    const auto query = Tensor::from_vector(
        {1, 0, 0, 1}, {1, 2, 1, 2});
    auto backing = Tensor::from_vector(
        {3, 4, 1, 0, -1, 2, 9, 9}, {1, 1, 4, 2});
    auto cache = Tensor::from_storage(
        backing.storage(), {1, 1, 3, 2}, backing.strides(), 0,
        DType::Float32);
    const auto query_before = query.to_vector();
    const auto cache_before = cache.to_vector();

    const auto scores = cached_gqa_attention_scores(
        query, cache, 2, 0.5F);
    EXPECT_EQ(scores.shape(), (Shape{1, 2, 1, 3}));
    EXPECT_EQ(scores.dtype(), DType::Float32);
    EXPECT_TRUE(scores.is_contiguous());
    expect_near(scores.to_vector(), {1.5F, 0.5F, -0.5F, 2, 0, 1});
    EXPECT_EQ(query.to_vector(), query_before);
    EXPECT_EQ(cache.to_vector(), cache_before);

    auto value_backing = Tensor::from_vector(
        {1, 2, 3, 4, 5, 6, 9, 9}, {1, 1, 4, 2});
    auto value_cache = Tensor::from_storage(
        value_backing.storage(), {1, 1, 3, 2}, value_backing.strides(), 0,
        DType::Float32);
    const auto probabilities = softmax(scores, -1);
    const auto context = cached_gqa_attention_context(
        probabilities, value_cache, 2);
    const auto fused = cached_gqa_attention(
        query, cache, value_cache, 2, 0.5F);
    const auto split = cached_gqa_attention_split_sequence(
        query, cache, value_cache, 2, 0.5F, 2);
    const auto materialized = cached_gqa_attention_materialized_scores(
        query, cache, value_cache, 2, 0.5F);
    const auto materialized_64 = cached_gqa_attention_materialized_scores(
        query, cache, value_cache, 2, 0.5F, 64);
    const auto materialized_128 = cached_gqa_attention_materialized_scores(
        query, cache, value_cache, 2, 0.5F, 128);
    const auto split_pv_1 = cached_gqa_attention_split_pv_exact_softmax(
        query, cache, value_cache, 2, 0.5F, 1);
    const auto split_pv_2 = cached_gqa_attention_split_pv_exact_softmax(
        query, cache, value_cache, 2, 0.5F, 2);
    const auto value_reuse_8 = cached_gqa_attention_gqa_value_reuse(
        query, cache, value_cache, 2, 0.5F, 8);
    const auto value_reuse_64 = cached_gqa_attention_gqa_value_reuse(
        query, cache, value_cache, 2, 0.5F, 64);
    EXPECT_EQ(context.shape(), (Shape{1, 2, 1, 2}));
    expect_near(context.to_vector(), fused.to_vector(), 2.0e-5F);
    EXPECT_EQ(split.to_vector(), fused.to_vector());
    EXPECT_EQ(materialized.to_vector(), fused.to_vector());
    EXPECT_EQ(materialized_64.to_vector(), fused.to_vector());
    EXPECT_EQ(materialized_128.to_vector(), fused.to_vector());
    EXPECT_EQ(split_pv_1.to_vector(), fused.to_vector());
    EXPECT_EQ(split_pv_2.to_vector(), fused.to_vector());
    EXPECT_EQ(value_reuse_8.to_vector(), fused.to_vector());
    EXPECT_EQ(value_reuse_64.to_vector(), fused.to_vector());

    const auto bf16_cache = cache.cast(DType::BFloat16);
    const auto bf16_value = value_cache.cast(DType::BFloat16);
    const auto bf16_scores = cached_gqa_attention_scores(
        query, bf16_cache, 2, 0.5F);
    EXPECT_EQ(bf16_scores.to_vector(), scores.to_vector());
    const auto bf16_context = cached_gqa_attention_context(
        softmax(bf16_scores, -1), bf16_value, 2);
    expect_near(
        bf16_context.to_vector(),
        cached_gqa_attention(
            query, bf16_cache, bf16_value, 2, 0.5F).to_vector(),
        2.0e-5F);
    EXPECT_THROW(
        (void)cached_gqa_attention_scores(query, cache, 3, 0.5F),
        std::invalid_argument);
    EXPECT_THROW(
        (void)cached_gqa_attention_scores(query, cache, 2, 0.0F),
        std::invalid_argument);
    EXPECT_THROW(
        (void)cached_gqa_attention_scores(
            query, cache.transpose(2, 3), 2, 0.5F),
        std::invalid_argument);
    EXPECT_THROW(
        (void)cached_gqa_attention_context(
            probabilities, value_cache, 3),
        std::invalid_argument);
    EXPECT_THROW(
        (void)cached_gqa_attention_context(
            probabilities, value_cache.transpose(2, 3), 2),
        std::invalid_argument);
    EXPECT_THROW(
        (void)cached_gqa_attention_split_sequence(
            query, cache, value_cache, 2, 0.5F, 0),
        std::invalid_argument);
    EXPECT_THROW(
        (void)cached_gqa_attention_split_sequence(
            query, cache, value_cache, 2, 0.5F, 4),
        std::invalid_argument);
    EXPECT_THROW(
        (void)cached_gqa_attention_materialized_scores(
            query, cache.transpose(2, 3), value_cache, 2, 0.5F),
        std::invalid_argument);
    EXPECT_THROW(
        (void)cached_gqa_attention_materialized_scores(
            query, cache, value_cache, 2, 0.5F, 32),
        std::invalid_argument);
    EXPECT_THROW(
        (void)cached_gqa_attention_split_pv_exact_softmax(
            query, cache, value_cache, 2, 0.5F, 0),
        std::invalid_argument);
    EXPECT_THROW(
        (void)cached_gqa_attention_split_pv_exact_softmax(
            query, cache, value_cache, 2, 0.5F, 4),
        std::invalid_argument);
    EXPECT_THROW(
        (void)cached_gqa_attention_gqa_value_reuse(
            query, cache, value_cache, 2, 0.5F, 7),
        std::invalid_argument);
}

TEST(CpuOpsTest, BatchedCacheStoreAndGqaAttentionKeepRowsIndependent) {
    Tensor backing({2, 1, 2, 2});
    auto cache = Tensor::from_storage(backing.storage(), {2, 1, 1, 2},
                                      backing.strides(), 0, DType::Float32);
    kv_cache_store_(cache,
                    Tensor::from_vector({3, 4, 5, 6}, {2, 1, 1, 2}), 0);
    const auto query = Tensor::from_vector(
        {1, 0, 0, 1, 1, 0, 0, 1}, {2, 2, 1, 2});
    const auto output = cached_gqa_attention(query, cache, cache, 2, 1.0F);
    EXPECT_EQ(output.shape(), (Shape{2, 2, 1, 2}));
    expect_near(output.to_vector(), {3, 4, 3, 4, 5, 6, 5, 6});
}

TEST(CpuOpsTest, PositionedRopeStoreAndAttentionMatchRowReferences) {
    const auto rope_input = Tensor::from_vector(
        {1, 2, 3, 4, 5, 6, 7, 8,
         2, 3, 4, 5, 6, 7, 8, 9},
        {2, 2, 1, 4});
    const auto positions = Tensor::from_int32_vector({0, 2}, {2});
    const auto bias = Tensor::from_vector(
        {0.1F, 0.2F, 0.3F, 0.4F, -0.1F, -0.2F, -0.3F, -0.4F}, {8});
    const auto interleaved = rope_positions(rope_input, positions);
    const auto split = rope_split_half_positions(rope_input, positions);
    for (std::int64_t row = 0; row < 2; ++row) {
        const auto position = row == 0 ? 0 : 2;
        const auto input_row = rope_input.slice(0, row, row + 1);
        expect_near(interleaved.slice(0, row, row + 1).to_vector(),
                    rope(input_row, 2, position).to_vector(), 2.0e-5F);
        expect_near(split.slice(0, row, row + 1).to_vector(),
                    rope_split_half(input_row, 2, position).to_vector(),
                    2.0e-5F);
    }
    const auto biased = rope_split_half_bias_positions(
        rope_input, bias, positions);
    for (std::int64_t row = 0; row < 2; ++row) {
        expect_near(
            biased.slice(0, row, row + 1).to_vector(),
            rope_split_half_bias(rope_input.slice(0, row, row + 1), bias,
                                 row == 0 ? 0 : 2).to_vector(),
            2.0e-5F);
    }

    for (const auto dtype : {DType::Float32, DType::BFloat16}) {
        Tensor key_backing({3, 1, 4, 2}, dtype);
        Tensor value_backing({3, 1, 4, 2}, dtype);
        fill_(key_backing, 0.0F);
        fill_(value_backing, 0.0F);
        auto key_cache = Tensor::from_storage(
            key_backing.storage(), {3, 1, 3, 2}, key_backing.strides(), 0, dtype);
        auto value_cache = Tensor::from_storage(
            value_backing.storage(), {3, 1, 3, 2}, value_backing.strides(), 0, dtype);
        const auto current_key = Tensor::from_vector(
            {3.01953125F, 4.02734375F, 1.00390625F, 2.01171875F},
            {2, 1, 1, 2});
        const auto current_value = Tensor::from_vector(
            {7.05078125F, 8.05859375F, 5.03515625F, 6.04296875F},
            {2, 1, 1, 2});
        const auto rows = Tensor::from_int32_vector({2, 0}, {2});
        kv_cache_store_pair_positions_(
            key_cache, value_cache, current_key, current_value, positions, rows);
        const auto query = Tensor::from_vector(
            {1, 0, 0, 1, 1, 0, 0, 1}, {2, 2, 1, 2});
        const auto actual = cached_gqa_attention_positions(
            query, key_cache, value_cache, positions, rows, 2, 1.0F);
        for (std::int64_t active = 0; active < 2; ++active) {
            const auto cache_row = active == 0 ? 2 : 0;
            const auto visible = active == 0 ? 1 : 3;
            const auto key_row = Tensor::from_storage(
                key_cache.storage(), {1, 1, visible, 2}, key_cache.strides(),
                cache_row * key_cache.stride(0), dtype);
            const auto value_row = Tensor::from_storage(
                value_cache.storage(), {1, 1, visible, 2}, value_cache.strides(),
                cache_row * value_cache.stride(0), dtype);
            expect_near(
                actual.slice(0, active, active + 1).to_vector(),
                cached_gqa_attention(
                    query.slice(0, active, active + 1), key_row, value_row, 2,
                    1.0F).to_vector(),
                dtype == DType::Float32 ? 2.0e-5F : 3.0e-2F);
        }
        EXPECT_THROW(kv_cache_store_pair_positions_(
                         key_cache, value_cache, current_key, current_value,
                         positions, Tensor::from_int32_vector({2, 3}, {2})),
                     std::out_of_range);
    }
}

TEST(CpuOpsTest, Bf16CacheRoundsStorageAndKeepsFp32AttentionOutput) {
    Tensor backing({2, 1, 3, 2}, DType::BFloat16);
    auto cache = Tensor::from_storage(backing.storage(), {2, 1, 1, 2},
                                      backing.strides(), 0, DType::BFloat16);
    const auto current = Tensor::from_vector(
        {1.00390625F, -2.01171875F, 3.01953125F, 4.02734375F},
        {2, 1, 1, 2});
    kv_cache_store_(cache, current, 0);
    EXPECT_EQ(cache.dtype(), DType::BFloat16);
    EXPECT_EQ(cache.storage().num_bytes(), 2U * 1U * 3U * 2U * 2U);
    expect_near(cache.to_vector(), current.cast(DType::BFloat16).to_vector());

    const auto query = Tensor::from_vector({1, 0, 0, 1}, {2, 1, 1, 2});
    const auto output = cached_gqa_attention(query, cache, cache, 1, 1.0F);
    EXPECT_EQ(output.dtype(), DType::Float32);
    expect_near(output.to_vector(), cache.to_vector());
    EXPECT_THROW((void)cached_gqa_attention(
                     query, cache, cache.cast(DType::Float32), 1, 1.0F),
                 std::invalid_argument);
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

TEST(CpuOpsTest, ArgmaxLastDimReducesRowsWithTieAndNonFiniteContracts) {
    const auto input = Tensor::from_vector(
        {1.0F, 3.0F, 2.0F, 5.0F, 5.0F, 4.0F}, {2, 3});
    const auto selected = argmax_last_dim(input);
    EXPECT_EQ(selected.shape(), (Shape{2}));
    EXPECT_EQ(selected.to_int32_vector(), (std::vector<std::int32_t>{1, 0}));
    const auto non_finite = Tensor::from_vector(
        {1.0F, 2.0F, 3.0F, 1.0F,
         4.0F, std::numeric_limits<float>::quiet_NaN(), 2.0F, 3.0F},
        {2, 2, 2});
    EXPECT_EQ(argmax_last_dim(non_finite).shape(), (Shape{2, 2}));
    EXPECT_EQ(argmax_last_dim(non_finite).to_int32_vector(),
              (std::vector<std::int32_t>{1, 0, -1, 1}));
    EXPECT_THROW((void)argmax_last_dim(Tensor({2, 0})), std::invalid_argument);
}

TEST(CpuOpsTest, ArgmaxOutWritesCallerOwnedHistoryViews) {
    const auto scalar = Tensor::from_vector({-2.0F, 4.0F, 4.0F}, {3});
    Tensor scalar_output({1, 1}, DType::Int32);
    argmax_out_(scalar, scalar_output);
    EXPECT_EQ(scalar_output.to_int32_vector(), (std::vector<std::int32_t>{1}));

    Tensor history({2, 2}, DType::Int32);
    auto first = history.slice(0, 0, 1).reshape({2});
    auto second = history.slice(0, 1, 2).reshape({2});
    argmax_last_dim_out_(
        Tensor::from_vector({1.0F, 5.0F, 2.0F, 7.0F, 3.0F, 4.0F}, {2, 3}),
        first);
    argmax_last_dim_out_(
        Tensor::from_vector({9.0F, 8.0F, 7.0F, 1.0F, 2.0F, 6.0F}, {2, 3}),
        second);
    EXPECT_EQ(history.to_int32_vector(),
              (std::vector<std::int32_t>{1, 0, 0, 2}));

    Tensor wrong_dtype({1, 1});
    EXPECT_THROW(argmax_out_(scalar, wrong_dtype), std::invalid_argument);
    Tensor wrong_shape({1}, DType::Int32);
    EXPECT_THROW(argmax_out_(scalar, wrong_shape), std::invalid_argument);
}

TEST(CpuOpsTest, EmbeddingGathersRowsAndRejectsBadIndex) {
    const auto weight = Tensor::from_vector({0, 1, 2, 3, 4, 5}, {3, 2});
    const auto indices = Tensor::from_int32_vector({2, 0, 1}, {3});
    EXPECT_EQ(embedding(weight, indices).to_vector(), (std::vector<float>{4, 5, 0, 1, 2, 3}));
    Tensor caller_output({3, 2});
    embedding_out_(caller_output, weight, indices);
    EXPECT_EQ(caller_output.to_vector(), embedding(weight, indices).to_vector());
    auto alias = weight;
    EXPECT_THROW(embedding_out_(alias, weight, indices), std::invalid_argument);
    EXPECT_THROW((void)embedding(weight, Tensor::from_int32_vector({3}, {1})), std::out_of_range);
}

TEST(CpuOpsTest, SoftmaxIsStableAndRowsSumToOne) {
    const auto input = Tensor::from_vector({1000, 1000, 1, 2, 3, 4}, {2, 3});
    const auto reference = softmax(input);
    const auto output = reference.to_vector();
    EXPECT_NEAR(output[0] + output[1] + output[2], 1.0F, 1.0e-6F);
    EXPECT_NEAR(output[3] + output[4] + output[5], 1.0F, 1.0e-6F);
    EXPECT_TRUE(std::isfinite(output[0]));
    Tensor caller_output(input.shape());
    softmax_out_(caller_output, input);
    EXPECT_EQ(caller_output.to_vector(), reference.to_vector());
    auto alias = input;
    EXPECT_THROW(softmax_out_(alias, input), std::invalid_argument);
    Tensor wrong_dtype(input.shape(), DType::BFloat16);
    EXPECT_THROW(softmax_out_(wrong_dtype, input), std::invalid_argument);
    EXPECT_THROW(softmax_out_(caller_output, input, 0), std::invalid_argument);
}

TEST(CpuOpsTest, RmsNormMatchesManualCalculation) {
    const auto input = Tensor::from_vector({3, 4}, {1, 2});
    const auto weight = Tensor::from_vector({1, 2}, {2});
    const auto denominator = std::sqrt(12.5F + 1.0e-5F);
    expect_near(rms_norm(input, weight).to_vector(), {3.0F / denominator, 8.0F / denominator});
    Tensor fp32_output(input.shape());
    rms_norm_out_(fp32_output, input, weight);
    EXPECT_EQ(fp32_output.to_vector(), rms_norm(input, weight).to_vector());
    Tensor bf16_output(input.shape(), DType::BFloat16);
    rms_norm_bf16_out_(bf16_output, input, weight);
    EXPECT_EQ(bf16_output.to_vector(),
              rms_norm(input, weight).cast(DType::BFloat16).to_vector());
    Tensor wrong({1, 2});
    EXPECT_THROW(rms_norm_bf16_out_(wrong, input, weight), std::invalid_argument);
    auto alias = input;
    EXPECT_THROW(rms_norm_bf16_out_(alias, input, weight), std::invalid_argument);
}

TEST(CpuOpsTest, SiluAndSwiGluMatchDefinitions) {
    const auto input = Tensor::from_vector({-1, 0, 1}, {3});
    const auto silu_values = silu(input).to_vector();
    EXPECT_NEAR(silu_values[0], -1.0F / (1.0F + std::exp(1.0F)), 1.0e-6F);
    EXPECT_EQ(silu_values[1], 0.0F);
    EXPECT_NEAR(silu_values[2], 1.0F / (1.0F + std::exp(-1.0F)), 1.0e-6F);
    expect_near(swiglu(input, Tensor::from_vector({2, 2, 2}, {3})).to_vector(),
                {2 * silu_values[0], 0, 2 * silu_values[2]});
    const auto up = Tensor::from_vector({2, 2, 2}, {3});
    Tensor multiplied({3});
    multiply_out_(multiplied, input, up);
    expect_near(multiplied.to_vector(), multiply(input, up).to_vector());
    Tensor caller_output({3});
    swiglu_out_(caller_output, input, up);
    expect_near(caller_output.to_vector(), swiglu(input, up).to_vector());
    EXPECT_THROW(
        (void)swiglu_with_implementation(
            input, up, SwiGLUImplementation::Vectorized),
        std::invalid_argument);
    auto alias_output = input;
    EXPECT_THROW(multiply_out_(alias_output, input, up), std::invalid_argument);
    EXPECT_THROW(swiglu_out_(alias_output, input, up), std::invalid_argument);
    Tensor wrong({2});
    EXPECT_THROW(swiglu_out_(wrong, input, up), std::invalid_argument);
}

TEST(CpuOpsTest, RopeLeavesPositionZeroAndRotatesPositionOne) {
    const auto input = Tensor::from_vector({1, 0, 0, 1, 1, 0, 0, 1}, {1, 2, 1, 4});
    const auto output = rope(input).to_vector();
    expect_near({output[0], output[1], output[2], output[3]}, {1, 0, 0, 1});
    EXPECT_NEAR(output[4], std::cos(1.0F), 1.0e-5F);
    EXPECT_NEAR(output[5], std::sin(1.0F), 1.0e-5F);
    Tensor caller_output(input.shape());
    rope_out_(caller_output, input);
    EXPECT_EQ(caller_output.to_vector(), rope(input).to_vector());
    auto alias = input;
    EXPECT_THROW(rope_out_(alias, input), std::invalid_argument);
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
    Tensor caller_output(Shape{});
    Tensor row_workspace({2, 2});
    cross_entropy_out_(caller_output, row_workspace, logits, targets);
    EXPECT_EQ(caller_output.to_vector(), cross_entropy(logits, targets).to_vector());
    Tensor wrong_workspace({2, 1});
    EXPECT_THROW(cross_entropy_out_(
                     caller_output, wrong_workspace, logits, targets),
                 std::invalid_argument);
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
        const auto reused = quantize_fp8_with_scale(
            input, format, 0.025F, quantized.scale);
        EXPECT_EQ(quantized.values.dtype(), format);
        EXPECT_EQ(reused.values.to_vector(), quantized.values.to_vector());
        EXPECT_EQ(reused.scale.storage().data(), quantized.scale.storage().data());
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
    EXPECT_THROW((void)quantize_fp8_with_scale(
                     input, DType::Float8E4M3FNUZ, 0.025F,
                     Tensor::from_vector({0.025F, 0.025F}, {2})),
                 std::invalid_argument);
}

TEST(CpuFp8OpsTest, DynamicTensorScaleUsesAmaxAndMinimumWithoutLosingHostOracle) {
    clear_fp8_dynamic_quant_stats();
    const auto input = Tensor::from_vector({-3.0F, -0.5F, 0.0F, 2.0F}, {2, 2});
    const auto dynamic = quantize_fp8_dynamic(
        input, DType::Float8E4M3FNUZ, 0.001F);
    EXPECT_TRUE(dynamic.host_scale_available);
    EXPECT_FLOAT_EQ(dynamic.scale_value, 3.0F / 240.0F);
    EXPECT_FLOAT_EQ(dynamic.scale.to_vector()[0], dynamic.scale_value);
    expect_near(dequantize_fp8(dynamic, DType::Float32).to_vector(),
                input.to_vector(), 0.15F);
    const auto minimum = quantize_fp8_dynamic(
        Tensor::from_vector({-0.01F, 0.01F}, {2}),
        DType::Float8E4M3FNUZ, 0.001F);
    EXPECT_FLOAT_EQ(minimum.scale_value, 0.001F);
    const auto clipped = quantize_fp8_dynamic(
        input, DType::Float8E4M3FNUZ, 0.001F, {}, 0.5F);
    EXPECT_FLOAT_EQ(clipped.scale_value, 1.5F / 240.0F);
    const auto clipped_values = dequantize_fp8(
        clipped, DType::Float32).to_vector();
    EXPECT_NEAR(clipped_values[0], -1.5F, 0.02F);
    EXPECT_EQ(fp8_dynamic_quant_stats().clipped_tensor_calls, 1U);
    const auto e5 = quantize_fp8_dynamic(
        Tensor::from_vector({-57344.0F, 57344.0F}, {2}),
        DType::Float8E5M2FNUZ, 0.001F);
    EXPECT_FLOAT_EQ(e5.scale_value, 1.0F);
    EXPECT_THROW((void)quantize_fp8_dynamic(
                     Tensor::from_vector(
                         {std::numeric_limits<float>::infinity()}, {1}),
                     DType::Float8E4M3FNUZ, 0.001F),
                 std::invalid_argument);
    EXPECT_THROW((void)quantize_fp8_dynamic(
                     input, DType::Float8E4M3FNUZ, 0.001F, {}, 0.0F),
                 std::invalid_argument);
}

TEST(CpuFp8OpsTest, DynamicRowScalesPreserveIndependentRanges) {
    const auto input = Tensor::from_vector(
        {1.0F, -2.0F, 0.5F, 100.0F, -200.0F, 50.0F}, {2, 3});
    const auto rows = quantize_fp8_rows_dynamic(
        input, DType::Float8E4M3FNUZ, 1.0e-4F);
    EXPECT_EQ(rows.scale_mode, Fp8ScaleMode::OuterRow);
    EXPECT_FALSE(rows.host_scale_available);
    const auto scales = rows.scale.to_vector();
    ASSERT_EQ(scales.size(), 2U);
    EXPECT_FLOAT_EQ(scales[0], 2.0F / 240.0F);
    EXPECT_FLOAT_EQ(scales[1], 200.0F / 240.0F);
    expect_near(dequantize_fp8(rows, DType::Float32).to_vector(),
                input.to_vector(), 5.0F);
}

TEST(CpuFp8OpsTest, DynamicColumnScalesPreserveIndependentWeightRanges) {
    clear_fp8_dynamic_quant_stats();
    const auto weight = Tensor::from_vector(
        {1.0F, 100.0F, 2.0F, 200.0F, 3.0F, 300.0F}, {3, 2});
    const auto columns = quantize_fp8_columns_dynamic(
        weight, DType::Float8E4M3FNUZ, 1.0e-4F);
    EXPECT_EQ(columns.scale_mode, Fp8ScaleMode::OuterColumn);
    EXPECT_FALSE(columns.host_scale_available);
    const auto scales = columns.scale.to_vector();
    ASSERT_EQ(scales.size(), 2U);
    EXPECT_FLOAT_EQ(scales[0], 3.0F / 240.0F);
    EXPECT_FLOAT_EQ(scales[1], 300.0F / 240.0F);
    EXPECT_EQ(fp8_dynamic_quant_stats().column_calls, 1U);
    EXPECT_EQ(fp8_dynamic_quant_stats().column_elements, 6U);
    expect_near(dequantize_fp8(columns, DType::Float32).to_vector(),
                weight.to_vector(), 8.0F);

    const auto left = Tensor::from_vector({1.0F, -2.0F, 0.5F}, {1, 3});
    const auto output = fp8_matmul(
        quantize_fp8_dynamic(left, DType::Float8E4M3FNUZ, 1.0e-4F),
        columns, DType::Float32);
    expect_near(output.to_vector(), matmul(left, weight).to_vector(), 10.0F);
    EXPECT_THROW((void)quantize_fp8_columns_dynamic(
                     Tensor::from_vector({1.0F, 2.0F}, {2}),
                     DType::Float8E4M3FNUZ, 1.0e-4F),
                 std::invalid_argument);
}

TEST(CpuFp8OpsTest, MixedE5ActivationAndE4WeightMatchDequantizedReference) {
    const auto activation = Tensor::from_vector(
        {1.0F, -20.0F, 300.0F, -4.0F, 50.0F, -1000.0F}, {2, 3});
    const auto weight = Tensor::from_vector(
        {1.0F, 2.0F, -3.0F, 4.0F, 5.0F, -6.0F}, {3, 2});
    const auto activation_fp8 = quantize_fp8_dynamic(
        activation, DType::Float8E5M2FNUZ, 1.0e-4F);
    const auto weight_fp8 = quantize_fp8_dynamic(
        weight, DType::Float8E4M3FNUZ, 1.0e-4F);
    const auto output = fp8_matmul(
        activation_fp8, weight_fp8, DType::Float32);
    const auto reference = matmul(
        dequantize_fp8(activation_fp8, DType::Float32),
        dequantize_fp8(weight_fp8, DType::Float32));
    EXPECT_EQ(output.to_vector(), reference.to_vector());
    EXPECT_FLOAT_EQ(activation_fp8.scale_value, 1000.0F / 57344.0F);
}

TEST(CallerOwnedBackwardTest, CpuOutputsMatchAllocatingReferences) {
    const auto input = Tensor::from_vector({-2, -1, 0, 1, 2, 3}, {2, 3});
    const auto weight = Tensor::from_vector({1, 0.5F, 2}, {3});
    const auto gradient = Tensor::from_vector({1, 2, 3, -1, -2, -3}, {2, 3});
    const auto probabilities = softmax(input);
    Tensor softmax_gradient(input.shape());
    softmax_backward_out_(softmax_gradient, probabilities, gradient);
    EXPECT_EQ(softmax_gradient.to_vector(),
              softmax_backward(probabilities, gradient).to_vector());

    Tensor input_gradient(input.shape());
    Tensor weight_gradient(weight.shape());
    Tensor row_workspace({2});
    rms_norm_backward_out_(input_gradient, weight_gradient, row_workspace,
                           input, weight, gradient);
    const auto rms_reference = rms_norm_backward(input, weight, gradient);
    EXPECT_EQ(input_gradient.to_vector(), rms_reference.first.to_vector());
    EXPECT_EQ(weight_gradient.to_vector(), rms_reference.second.to_vector());

    const auto up = Tensor::from_vector({1, 2, 3, 4, 5, 6}, {2, 3});
    Tensor gate_gradient(input.shape());
    Tensor up_gradient(input.shape());
    swiglu_backward_out_(gate_gradient, up_gradient, input, up, gradient);
    const auto swiglu_reference = swiglu_backward(input, up, gradient);
    EXPECT_EQ(gate_gradient.to_vector(), swiglu_reference.first.to_vector());
    EXPECT_EQ(up_gradient.to_vector(), swiglu_reference.second.to_vector());
    const auto scalar_seed = Tensor::from_vector({0.5F}, {});
    Tensor scalar_gate_gradient(input.shape());
    Tensor scalar_up_gradient(input.shape());
    swiglu_backward_scalar_seed_out_(
        scalar_gate_gradient, scalar_up_gradient, input, up, scalar_seed);
    const auto expanded_seed = Tensor::from_vector(
        std::vector<float>(static_cast<std::size_t>(input.numel()), 0.5F),
        input.shape());
    const auto scalar_reference = swiglu_backward(input, up, expanded_seed);
    EXPECT_EQ(scalar_gate_gradient.to_vector(),
              scalar_reference.first.to_vector());
    EXPECT_EQ(scalar_up_gradient.to_vector(),
              scalar_reference.second.to_vector());
    EXPECT_THROW(
        swiglu_backward_scalar_seed_out_(
            scalar_gate_gradient, scalar_up_gradient, input, up,
            Tensor::from_vector({1, 2}, {2})),
        std::invalid_argument);
    for (const auto dtype : {DType::Float16, DType::BFloat16}) {
        const auto low_gate = input.cast(dtype);
        const auto low_up = up.cast(dtype);
        const auto low_gradient = gradient.cast(dtype);
        Tensor low_gate_gradient(input.shape(), dtype);
        Tensor low_up_gradient(input.shape(), dtype);
        swiglu_backward_typed_out_(
            low_gate_gradient, low_up_gradient,
            low_gate, low_up, low_gradient);
        const auto low_reference = swiglu_backward(
            low_gate.cast(DType::Float32), low_up.cast(DType::Float32),
            low_gradient.cast(DType::Float32));
        EXPECT_EQ(low_gate_gradient.to_vector(),
                  low_reference.first.cast(dtype).to_vector());
        EXPECT_EQ(low_up_gradient.to_vector(),
                  low_reference.second.cast(dtype).to_vector());
    }

    const auto rope_gradient = Tensor::from_vector(
        {1, 2, 3, 4, -1, -2, -3, -4}, {1, 2, 1, 4});
    Tensor rope_input_gradient(rope_gradient.shape());
    rope_backward_out_(rope_input_gradient, rope_gradient);
    EXPECT_EQ(rope_input_gradient.to_vector(),
              rope_backward(rope_gradient).to_vector());

    const auto logits = Tensor::from_vector({2, 1, 0, 100, -100, 0}, {2, 3});
    const auto targets = Tensor::from_int32_vector({0, -100}, {2});
    const auto seed = Tensor::from_vector({0.75F}, {});
    Tensor logits_gradient(logits.shape());
    Tensor loss_rows({2, 2});
    Tensor loss_factor(Shape{});
    cross_entropy_backward_out_(
        logits_gradient, loss_rows, loss_factor, logits, targets, seed);
    EXPECT_EQ(logits_gradient.to_vector(),
              cross_entropy_backward(logits, targets, seed).to_vector());

    const auto indices = Tensor::from_int32_vector({2, 0, 2}, {3});
    const auto embedding_gradient = Tensor::from_vector(
        {1, 2, 3, 4, 5, 6}, {3, 2});
    Tensor embedding_weight_gradient({4, 2});
    embedding_weight_gradient.fill(0.0F);
    embedding_backward_add_(
        embedding_weight_gradient, embedding_gradient, indices);
    EXPECT_EQ(embedding_weight_gradient.to_vector(),
              embedding_backward(embedding_gradient, indices, 4).to_vector());

    auto alias = probabilities;
    EXPECT_THROW(softmax_backward_out_(alias, probabilities, gradient),
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

TEST(LowLevelOpsTest, CallerOwnedCpuBuffersCoverFloat16AndBfloat16) {
    for (const auto dtype : {DType::Float16, DType::BFloat16}) {
        const auto left = Tensor::from_vector({1, -2, 3, 4}, {2, 2}, dtype);
        const auto right = Tensor::from_vector({5, 6, -7, 0.5F}, {2, 2}, dtype);
        Tensor output({2, 2}, dtype);
        const auto* address = output.storage().data();
        add_out(output.view(), left.view(), right.view());
        EXPECT_EQ(output.storage().data(), address);
        EXPECT_EQ(output.to_vector(), add(left, right).to_vector());
        multiply_out(output.view(), left.view(), right.view());
        EXPECT_EQ(output.storage().data(), address);
        EXPECT_EQ(output.to_vector(), multiply(left, right).to_vector());
    }
    auto integers = Tensor::from_int32_vector({1, 2}, {2});
    const auto& integer_inputs = integers;
    EXPECT_THROW(add_out(integers.view(), integer_inputs.view(),
                         integer_inputs.view()),
                 std::invalid_argument);
}

}  // namespace microllm::ops
