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
