#include <cmath>
#include <limits>
#include <tuple>
#include <utility>
#include <vector>

#include <gtest/gtest.h>
#include <microllm/ops/ops.h>
#include <microllm/ops/low_level.h>
#include <microllm/runtime/runtime.h>
#include <microllm/runtime/memory.h>
#include <microllm/inference/generator.h>
#include <microllm/model/model.h>
#include <microllm/training/trainer.h>

namespace microllm::ops {

TEST(HipTensorDTypeTest, LowPrecisionTransferViewAndCastRoundTrip) {
    if (runtime::hip_device_count() == 0) GTEST_SKIP() << "No visible HIP device";
    const auto gpu = Device::hip(0);
    for (const auto dtype : {DType::Float16, DType::BFloat16}) {
        const auto cpu = Tensor::from_vector({0, 1, 2, 3, 4, 5}, {2, 3}, dtype);
        const auto device_tensor = cpu.to(gpu);
        EXPECT_EQ(device_tensor.dtype(), dtype);
        EXPECT_EQ(device_tensor.storage().num_bytes(), 12U);
        EXPECT_EQ(device_tensor.to_vector(), cpu.to_vector());
        EXPECT_EQ(device_tensor.transpose(0, 1).contiguous().to_vector(),
                  (std::vector<float>{0, 3, 1, 4, 2, 5}));
        const auto float32 = device_tensor.cast(DType::Float32);
        EXPECT_EQ(float32.device(), gpu);
        EXPECT_EQ(float32.dtype(), DType::Float32);
        EXPECT_EQ(float32.to_vector(), cpu.to_vector());
    }
}
namespace {

void require_gpu() {
    if (runtime::hip_device_count() == 0) GTEST_SKIP() << "No visible HIP device";
}

void expect_near(const std::vector<float>& actual, const std::vector<float>& expected,
                 float tolerance = 1.0e-5F) {
    ASSERT_EQ(actual.size(), expected.size());
    for (std::size_t index = 0; index < actual.size(); ++index) {
        EXPECT_NEAR(actual[index], expected[index], tolerance) << "index=" << index;
    }
}

}  // namespace

TEST(HipOpsTest, FillAndElementwiseMatchCpuReference) {
    require_gpu();
    const auto gpu = Device::hip();
    const auto left_cpu = Tensor::from_vector({1, -2, 3, 4}, {2, 2});
    const auto right_cpu = Tensor::from_vector({4, 5, -6, 2}, {2, 2});
    auto filled = Tensor({17}, DType::Float32, gpu);
    fill_(filled, 3.5F);
    EXPECT_EQ(filled.to_vector(), std::vector<float>(17, 3.5F));

    const auto left = left_cpu.to(gpu);
    const auto right = right_cpu.to(gpu);
    expect_near(add(left, right).to_vector(), add(left_cpu, right_cpu).to_vector());
    expect_near(multiply(left, right).to_vector(), multiply(left_cpu, right_cpu).to_vector());
    expect_near(scale(left, -0.25F).to_vector(), scale(left_cpu, -0.25F).to_vector());
}

TEST(HipOpsTest, BiasForwardAndGradientStayOnDevice) {
    require_gpu();
    const auto gpu = Device::hip(0);
    const auto input_cpu = Tensor::from_vector({1, 2, 3, 4, 5, 6}, {2, 3});
    const auto bias_cpu = Tensor::from_vector({0.5F, -1.0F, 2.0F}, {3});
    const auto input = input_cpu.to(gpu);
    const auto bias = bias_cpu.to(gpu);
    runtime::reset_transfer_stats();
    const auto output = add_bias(input, bias);
    const auto reduced = bias_gradient(input);
    runtime::synchronize(gpu);
    const auto transfers = runtime::transfer_stats();
    EXPECT_EQ(transfers.host_to_device_calls, 0U);
    EXPECT_EQ(transfers.device_to_host_calls, 0U);
    expect_near(output.to_vector(), add_bias(input_cpu, bias_cpu).to_vector());
    expect_near(reduced.to_vector(), bias_gradient(input_cpu).to_vector());
}

TEST(HipLowPrecisionOpsTest, NativeBasicKernelsMatchCpuAndAvoidHostTransfers) {
    require_gpu();
    const auto gpu = Device::hip(0);
    for (const auto dtype : {DType::Float16, DType::BFloat16}) {
        const auto tolerance = dtype == DType::Float16 ? 3.0e-3F : 3.0e-2F;
        const auto left_cpu = Tensor::from_vector(
            {1, -2, 3, 4, 0.5F, -0.25F}, {2, 3}, dtype);
        const auto right_cpu = Tensor::from_vector(
            {4, 5, -6, 2, 1.5F, 0.25F}, {2, 3}, dtype);
        const auto mat_right_cpu = Tensor::from_vector(
            {1, 2, 3, 4, 5, 6}, {3, 2}, dtype);
        const auto left = left_cpu.to(gpu);
        const auto right = right_cpu.to(gpu);
        const auto mat_right = mat_right_cpu.to(gpu);
        runtime::reset_transfer_stats();
        auto filled = Tensor({6}, dtype, gpu);
        fill_(filled, -1.25F);
        const auto added = add(left, right);
        const auto multiplied = multiply(left, right);
        const auto scaled = scale(left, -0.25F);
        const auto matrix = matmul(left, mat_right);
        const auto activated = silu(left);
        const auto gated = swiglu(left, right);
        runtime::synchronize(gpu);
        const auto transfers = runtime::transfer_stats();
        EXPECT_EQ(transfers.host_to_device_calls, 0U);
        EXPECT_EQ(transfers.device_to_host_calls, 0U);

        expect_near(filled.to_vector(), std::vector<float>(6, -1.25F), tolerance);
        expect_near(added.to_vector(), add(left_cpu, right_cpu).to_vector(), tolerance);
        expect_near(multiplied.to_vector(), multiply(left_cpu, right_cpu).to_vector(), tolerance);
        expect_near(scaled.to_vector(), scale(left_cpu, -0.25F).to_vector(), tolerance);
        expect_near(matrix.to_vector(), matmul(left_cpu, mat_right_cpu).to_vector(), tolerance);
        expect_near(activated.to_vector(), silu(left_cpu).to_vector(), tolerance);
        expect_near(gated.to_vector(), swiglu(left_cpu, right_cpu).to_vector(), tolerance);
    }
}

TEST(HipBf16MixedGemmTest, NativeCastAndFp32OutputMatchRoundedCpuReference) {
    require_gpu();
    constexpr std::int64_t size = 128;
    std::vector<float> left_values(static_cast<std::size_t>(size * size));
    std::vector<float> right_values(left_values.size());
    for (std::size_t index = 0; index < left_values.size(); ++index) {
        left_values[index] = static_cast<float>(static_cast<int>(index % 37) - 18) * 0.015625F;
        right_values[index] = static_cast<float>(static_cast<int>(index % 41) - 20) * 0.015625F;
    }
    const auto left = Tensor::from_vector(left_values, {size, size});
    const auto right = Tensor::from_vector(right_values, {size, size});
    const auto rounded_right = cast(right, DType::BFloat16);
    const auto expected = bf16_matmul(left, rounded_right).to_vector();
    const auto gpu = Device::hip();
    const auto device_left = left.to(gpu);
    const auto device_right = right.to(gpu);
    runtime::reset_transfer_stats();
    const auto device_right_bf16 = cast(device_right, DType::BFloat16);
    const auto restored = cast(device_right_bf16, DType::Float32);
    const auto actual = bf16_matmul(device_left, device_right_bf16);
    runtime::synchronize(gpu);
    const auto transfers = runtime::transfer_stats();
    EXPECT_EQ(transfers.host_to_device_calls, 0U);
    EXPECT_EQ(transfers.device_to_host_calls, 0U);
    expect_near(restored.to_vector(), rounded_right.cast(DType::Float32).to_vector(), 0.0F);
    expect_near(actual.to_vector(), expected, 2.0e-2F);
    EXPECT_EQ(actual.dtype(), DType::Float32);

    const auto special = Tensor::from_vector(
        {0.0F, -0.0F, std::numeric_limits<float>::infinity(),
         -std::numeric_limits<float>::infinity(),
         std::numeric_limits<float>::quiet_NaN()}, {5}).to(gpu);
    const auto special_values = cast(cast(special, DType::BFloat16),
                                     DType::Float32).to_vector();
    EXPECT_EQ(special_values[0], 0.0F);
    EXPECT_TRUE(std::signbit(special_values[1]));
    EXPECT_TRUE(std::isinf(special_values[2]));
    EXPECT_TRUE(std::isinf(special_values[3]));
    EXPECT_TRUE(std::isnan(special_values[4]));
}

TEST(HipFp8OpsTest, QuantizeDequantizeAndScaledGemmAreDeviceNative) {
    require_gpu();
    if (!hipblaslt_available()) GTEST_SKIP() << "hipBLASLt is unavailable";
    const auto gpu = Device::hip(0);
    constexpr std::int64_t size = 128;
    std::vector<float> left_values(static_cast<std::size_t>(size * size));
    std::vector<float> right_values(left_values.size());
    for (std::size_t index = 0; index < left_values.size(); ++index) {
        left_values[index] = static_cast<float>(static_cast<int>(index % 31) - 15) / 31.0F;
        right_values[index] = static_cast<float>(static_cast<int>(index % 19) - 9) / 19.0F;
    }
    const auto left_cpu = Tensor::from_vector(left_values, {size, size});
    const auto right_cpu = Tensor::from_vector(right_values, {size, size});
    const auto left = left_cpu.to(gpu);
    const auto right = right_cpu.to(gpu);

    for (const auto format : {DType::Float8E4M3FNUZ, DType::Float8E5M2FNUZ}) {
        runtime::reset_transfer_stats();
        const auto quantized = quantize_fp8(left, format, 1.0F / 240.0F);
        const auto restored = dequantize_fp8(quantized, DType::Float32);
        runtime::synchronize(gpu);
        const auto transfers = runtime::transfer_stats();
        // One host-to-device scalar scale metadata copy is expected; tensor payloads stay on GPU.
        EXPECT_EQ(transfers.host_to_device_calls, 1U);
        EXPECT_EQ(transfers.device_to_host_calls, 0U);
        const auto restored_values = restored.to_vector();
        float maximum_error = 0.0F;
        for (std::size_t index = 0; index < restored_values.size(); ++index) {
            maximum_error = std::max(
                maximum_error, std::abs(restored_values[index] - left_values[index]));
        }
        const auto tolerance = format == DType::Float8E4M3FNUZ ? 0.035F : 0.07F;
        EXPECT_LE(maximum_error, tolerance) << "format=" << dtype_name(format);
    }

    const auto left_fp8 = quantize_fp8(left, DType::Float8E4M3FNUZ, 1.0F / 240.0F);
    const auto right_fp8 = quantize_fp8(right, DType::Float8E4M3FNUZ, 1.0F / 240.0F);
    runtime::reset_transfer_stats();
    const auto output = fp8_matmul(left_fp8, right_fp8, DType::BFloat16);
    runtime::synchronize(gpu);
    const auto transfers = runtime::transfer_stats();
    EXPECT_EQ(transfers.host_to_device_calls, 0U);
    EXPECT_EQ(transfers.device_to_host_calls, 0U);
    EXPECT_EQ(output.dtype(), DType::BFloat16);
    const auto output_values = output.to_vector();
    const auto reference_values = matmul(left_cpu, right_cpu).to_vector();
    float maximum_gemm_error = 0.0F;
    for (std::size_t index = 0; index < output_values.size(); ++index) {
        maximum_gemm_error = std::max(
            maximum_gemm_error, std::abs(output_values[index] - reference_values[index]));
    }
    EXPECT_LE(maximum_gemm_error, 0.5F);
}

TEST(HipFp8TrainingTest, ForwardAndStraightThroughBackwardKeepFp32MastersOnDevice) {
    require_gpu();
    if (!hipblaslt_available()) GTEST_SKIP() << "hipBLASLt is unavailable";
    constexpr std::int64_t size = 128;
    std::vector<float> left_values(static_cast<std::size_t>(size * size));
    std::vector<float> right_values(left_values.size());
    for (std::size_t index = 0; index < left_values.size(); ++index) {
        left_values[index] = static_cast<float>(static_cast<int>(index % 17) - 8) / 17.0F;
        right_values[index] = static_cast<float>(static_cast<int>(index % 13) - 6) / 13.0F;
    }
    autograd::Value left(Tensor::from_vector(left_values, {size, size}).to(Device::hip()), true);
    autograd::Value right(Tensor::from_vector(right_values, {size, size}).to(Device::hip()), true);
    runtime::reset_transfer_stats();
    const auto loss = autograd::mean(autograd::fp8_matmul(
        left, right, 1.0F / 240.0F, 1.0F / 240.0F));
    loss.backward();
    runtime::synchronize(Device::hip());
    const auto transfers = runtime::transfer_stats();
    EXPECT_EQ(transfers.device_to_host_calls, 0U);
    EXPECT_TRUE(left.has_grad());
    EXPECT_TRUE(right.has_grad());
    EXPECT_EQ(left.grad().dtype(), DType::Float32);
    EXPECT_EQ(right.grad().dtype(), DType::Float32);
    EXPECT_TRUE(std::isfinite(loss.data().to_vector()[0]));
}

TEST(HipFp8TrainingTest, TransformerLinearPolicyRunsEndToEndOnMi300) {
    require_gpu();
    if (!hipblaslt_available()) GTEST_SKIP() << "hipBLASLt is unavailable";
    model::ModelConfig config{.vocabulary_size = 128,
                              .dimension = 128,
                              .layers = 1,
                              .heads = 4,
                              .kv_heads = 4,
                              .ffn_dimension = 256,
                              .max_sequence_length = 16,
                              .rope_base = 10000.0F,
                              .tie_embeddings = false,
                              .linear_precision = model::LinearPrecision::Float8E4M3FNUZ,
                              .fp8_activation_scale = 0.025F,
                              .fp8_weight_scale = 0.005F};
    model::TransformerModel model(config, 211);
    model.to(Device::hip(0));
    std::vector<std::int32_t> token_values(16);
    std::vector<std::int32_t> target_values(16);
    for (std::size_t index = 0; index < token_values.size(); ++index) {
        token_values[index] = static_cast<std::int32_t>(index);
        target_values[index] = static_cast<std::int32_t>(index + 1);
    }
    const auto tokens = Tensor::from_int32_vector(token_values, {1, 16}).to(Device::hip(0));
    const auto targets = Tensor::from_int32_vector(target_values, {1, 16}).to(Device::hip(0));
    runtime::reset_transfer_stats();
    const auto loss = model.loss(tokens, targets);
    loss.backward();
    runtime::synchronize(Device::hip(0));
    const auto transfers = runtime::transfer_stats();
    EXPECT_EQ(transfers.device_to_host_calls, 0U);
    EXPECT_TRUE(std::isfinite(loss.data().to_vector()[0]));
    for (const auto& [name, parameter] : model.named_parameters()) {
        ASSERT_TRUE(parameter->has_grad()) << name;
        EXPECT_EQ(parameter->data().dtype(), DType::Float32) << name;
        EXPECT_EQ(parameter->grad().dtype(), DType::Float32) << name;
    }
    inference::KVCache cache(config.layers, config.max_sequence_length);
    const auto decode_logits = model.forward_cached(
        Tensor::from_int32_vector({1}, {1, 1}).to(Device::hip(0)), cache);
    EXPECT_EQ(decode_logits.shape(), (Shape{1, 1, 128}));
    for (const auto value : decode_logits.to_vector()) EXPECT_TRUE(std::isfinite(value));
}

TEST(HipOpsTest, NaiveBatchedMatmulMatchesCpuReference) {
    require_gpu();
    const auto left_cpu = Tensor::from_vector({1, 2, 3, 4, 5, 6, 1, 0, 0, 1, 1, 1}, {2, 2, 3});
    const auto right_cpu = Tensor::from_vector({1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 6}, {2, 3, 2});
    const auto actual = matmul(left_cpu.to(Device::hip()), right_cpu.to(Device::hip())).to_vector();
    expect_near(actual, matmul(left_cpu, right_cpu).to_vector());
}

TEST(HipOpsTest, RejectsMixedDevicesAndNonContiguousInput) {
    require_gpu();
    const auto cpu = Tensor::from_vector({1, 2, 3, 4}, {2, 2});
    const auto gpu = cpu.to(Device::hip());
    EXPECT_THROW((void)add(cpu, gpu), std::invalid_argument);
    EXPECT_THROW((void)add(gpu.transpose(0, 1), gpu), std::invalid_argument);
}

TEST(HipOpsTest, EmbeddingAndActivationsMatchCpuReference) {
    require_gpu();
    const auto weight_cpu = Tensor::from_vector({0, 1, 2, 3, 4, 5}, {3, 2});
    const auto indices_cpu = Tensor::from_int32_vector({2, 0, 1}, {3});
    expect_near(embedding(weight_cpu.to(Device::hip()), indices_cpu.to(Device::hip())).to_vector(),
                embedding(weight_cpu, indices_cpu).to_vector());

    const auto input_cpu = Tensor::from_vector({-2, -1, 0, 1, 2, 3}, {2, 3});
    const auto up_cpu = Tensor::from_vector({1, 2, 3, 4, 5, 6}, {2, 3});
    expect_near(silu(input_cpu.to(Device::hip())).to_vector(), silu(input_cpu).to_vector());
    expect_near(swiglu(input_cpu.to(Device::hip()), up_cpu.to(Device::hip())).to_vector(),
                swiglu(input_cpu, up_cpu).to_vector());
}

TEST(HipOpsTest, SoftmaxAndRmsNormMatchCpuReference) {
    require_gpu();
    const auto input_cpu = Tensor::from_vector({1000, 1000, 999, 1, 2, 3}, {2, 3});
    const auto weight_cpu = Tensor::from_vector({1, 0.5F, 2}, {3});
    expect_near(softmax(input_cpu.to(Device::hip())).to_vector(), softmax(input_cpu).to_vector());
    expect_near(rms_norm(input_cpu.to(Device::hip()), weight_cpu.to(Device::hip())).to_vector(),
                rms_norm(input_cpu, weight_cpu).to_vector(), 2.0e-4F);
}

TEST(HipRmsNormTest, BlockParallelForwardBackwardCoverModelWidthsWithoutHostTransfers) {
    require_gpu();
    const auto gpu = Device::hip(0);
    for (const auto width : {16LL, 384LL, 512LL, 896LL, 1536LL}) {
        for (const auto rows : {1LL, 3LL, 32LL}) {
            const auto epsilon = width == 1536 ? 1.0e-6F : 1.0e-5F;
            std::vector<float> input_values(static_cast<std::size_t>(rows * width));
            std::vector<float> gradient_values(input_values.size());
            std::vector<float> weight_values(static_cast<std::size_t>(width));
            for (std::size_t index = 0; index < input_values.size(); ++index) {
                input_values[index] = index % 97 == 0
                                          ? 0.0F
                                          : static_cast<float>(static_cast<int>(index % 41) - 20) *
                                                0.03125F;
                gradient_values[index] =
                    static_cast<float>(static_cast<int>(index % 29) - 14) * 0.015625F;
            }
            input_values.front() = 1000.0F;
            if (input_values.size() > 1) input_values[1] = -1000.0F;
            for (std::size_t column = 0; column < weight_values.size(); ++column) {
                weight_values[column] = 0.5F + static_cast<float>(column % 17) * 0.03125F;
            }
            const auto input = Tensor::from_vector(input_values, {rows, width});
            const auto weight = Tensor::from_vector(weight_values, {width});
            const auto gradient = Tensor::from_vector(gradient_values, {rows, width});
            const auto expected = rms_norm(input, weight, epsilon).to_vector();
            const auto expected_backward =
                rms_norm_backward(input, weight, gradient, epsilon);
            const auto device_input = input.to(gpu);
            const auto device_weight = weight.to(gpu);
            const auto device_gradient = gradient.to(gpu);
            runtime::reset_transfer_stats();
            const auto actual = rms_norm(device_input, device_weight, epsilon);
            const auto actual_backward =
                rms_norm_backward(device_input, device_weight, device_gradient, epsilon);
            runtime::synchronize(gpu);
            const auto transfers = runtime::transfer_stats();
            EXPECT_EQ(transfers.host_to_device_calls, 0U);
            EXPECT_EQ(transfers.device_to_host_calls, 0U);
            expect_near(actual.to_vector(), expected, 4.0e-4F);
            expect_near(actual_backward.first.to_vector(),
                        expected_backward.first.to_vector(), 4.0e-4F);
            expect_near(actual_backward.second.to_vector(),
                        expected_backward.second.to_vector(), 4.0e-4F);
        }
    }
}

TEST(HipCachedAttentionTest, FusedMhaGqaAndLongSequenceFallbackMatchCpu) {
    require_gpu();
    const auto gpu = Device::hip(0);
    constexpr std::int64_t heads = 4;
    constexpr std::int64_t width = 16;
    for (const auto gqa : {false, true}) {
        const auto kv_heads = gqa ? 1LL : heads;
        const auto repeats = heads / kv_heads;
        std::vector<float> query_values(static_cast<std::size_t>(heads * width));
        for (std::size_t index = 0; index < query_values.size(); ++index) {
            query_values[index] =
                static_cast<float>(static_cast<int>(index % 17) - 8) * 0.03125F;
        }
        const auto query = Tensor::from_vector(query_values, {1, heads, 1, width});
        for (const auto sequence : {1LL, 32LL, 128LL, 512LL, 4097LL}) {
            std::vector<float> key_values(
                static_cast<std::size_t>(kv_heads * sequence * width));
            std::vector<float> value_values(key_values.size());
            for (std::size_t index = 0; index < key_values.size(); ++index) {
                key_values[index] =
                    static_cast<float>(static_cast<int>(index % 29) - 14) * 0.015625F;
                value_values[index] =
                    static_cast<float>(static_cast<int>(index % 31) - 15) * 0.0234375F;
            }
            const auto key = Tensor::from_vector(
                key_values, {1, kv_heads, sequence, width});
            const auto value = Tensor::from_vector(
                value_values, {1, kv_heads, sequence, width});
            const auto expected = cached_gqa_attention(
                query, key, value, repeats, 0.25F).to_vector();
            const auto device_query = query.to(gpu);
            const auto device_key = key.to(gpu);
            const auto device_value = value.to(gpu);
            runtime::reset_transfer_stats();
            const auto actual = cached_gqa_attention(
                device_query, device_key, device_value, repeats, 0.25F);
            runtime::synchronize(gpu);
            const auto transfers = runtime::transfer_stats();
            EXPECT_EQ(transfers.host_to_device_calls, 0U);
            EXPECT_EQ(transfers.device_to_host_calls, 0U);
            expect_near(actual.to_vector(), expected, 5.0e-4F);
        }
    }
}

TEST(HipArgmaxTest, CoversLargeVocabulariesTiesAndScalarTransferContract) {
    require_gpu();
    const auto gpu = Device::hip(0);
    for (const auto vocabulary : {32LL, 8192LL, 151936LL}) {
        std::vector<float> values(static_cast<std::size_t>(vocabulary), -3.0F);
        const auto first = 1LL;
        values[static_cast<std::size_t>(first)] = 7.0F;
        values.back() = 7.0F;
        const auto input = Tensor::from_vector(values, {1, 1, vocabulary}).to(gpu);
        runtime::reset_transfer_stats();
        const auto selected = argmax(input);
        runtime::synchronize(gpu);
        auto transfers = runtime::transfer_stats();
        EXPECT_EQ(transfers.device_to_host_calls, 0U);
        EXPECT_EQ(selected.to_int32_vector(),
                  (std::vector<std::int32_t>{static_cast<std::int32_t>(first)}));
        transfers = runtime::transfer_stats();
        EXPECT_EQ(transfers.device_to_host_calls, 1U);
        EXPECT_EQ(transfers.device_to_host_bytes, sizeof(std::int32_t));
    }
    const auto non_finite = Tensor::from_vector(
        {1.0F, std::numeric_limits<float>::quiet_NaN()}, {2}).to(gpu);
    EXPECT_EQ(argmax(non_finite).to_int32_vector(),
              (std::vector<std::int32_t>{-1}));
    std::vector<float> large_non_finite(40000, -1.0F);
    large_non_finite[17] = 3.0F;
    large_non_finite.back() = std::numeric_limits<float>::infinity();
    EXPECT_EQ(argmax(Tensor::from_vector(large_non_finite, {40000}).to(gpu))
                  .to_int32_vector(),
              (std::vector<std::int32_t>{-1}));
}

TEST(HipGenerationTest, GreedyLoopKeepsSelectedTokenOnDevice) {
    require_gpu();
    const model::ModelConfig config{.vocabulary_size = 16,
                                    .dimension = 8,
                                    .layers = 1,
                                    .heads = 2,
                                    .kv_heads = 1,
                                    .ffn_dimension = 16,
                                    .max_sequence_length = 8,
                                    .rope_base = 10000.0F,
                                    .tie_embeddings = true};
    model::TransformerModel model(config, 67);
    model.to(Device::hip());
    const std::vector<std::int32_t> prompt{1, 2};
    runtime::reset_transfer_stats();
    const auto generated = inference::generate(
        model, prompt, {.max_new_tokens = 4, .temperature = 0.0F, .top_k = 1, .seed = 9});
    const auto transfers = runtime::transfer_stats();
    EXPECT_EQ(generated.size(), 6U);
    EXPECT_EQ(transfers.host_to_device_calls, prompt.size());
    EXPECT_EQ(transfers.host_to_device_bytes, prompt.size() * sizeof(std::int32_t));
    EXPECT_EQ(transfers.device_to_host_calls, 4U);
    EXPECT_EQ(transfers.device_to_host_bytes, 4U * sizeof(std::int32_t));
}

TEST(HipAllocatorStressTest, ReusedDefaultStreamBlocksPreserveAsyncKernelOrder) {
    require_gpu();
    const auto gpu = Device::hip(0);
    runtime::enable_hip_caching_allocator(gpu);
    runtime::reset_allocation_peak(gpu);
    for (int iteration = 0; iteration < 256; ++iteration) {
        Tensor temporary({4096}, DType::Float32, gpu);
        fill_(temporary, static_cast<float>(iteration));
        if (iteration % 16 == 15) runtime::synchronize(gpu);
    }
    try {
        Tensor temporary({4096}, DType::Float32, gpu);
        fill_(temporary, -7.0F);
        throw std::runtime_error("exercise exceptional destruction");
    } catch (const std::runtime_error&) {
    }
    runtime::synchronize(gpu);
    Tensor final({4096}, DType::Float32, gpu);
    fill_(final, 42.0F);
    const auto values = final.to_vector();
    ASSERT_EQ(values.size(), 4096U);
    for (const auto value : values) EXPECT_EQ(value, 42.0F);
    const auto stats = runtime::allocation_stats(gpu);
    EXPECT_GT(stats.cache_reuse_calls, 0U);
    EXPECT_LT(stats.backend_allocation_calls, stats.allocation_calls);
}

TEST(HipOpsTest, RopeAndCrossEntropyMatchCpuReference) {
    require_gpu();
    const auto rope_input = Tensor::from_vector({1, 0, 0, 1, 1, 0, 0, 1}, {1, 2, 1, 4});
    expect_near(rope(rope_input.to(Device::hip())).to_vector(), rope(rope_input).to_vector());
    expect_near(rope_split_half(rope_input.to(Device::hip())).to_vector(),
                rope_split_half(rope_input).to_vector());

    const auto logits_cpu = Tensor::from_vector({2, 1, 0, 0, 1, 2}, {2, 3});
    const auto targets_cpu = Tensor::from_int32_vector({0, 2}, {2});
    expect_near(cross_entropy(logits_cpu.to(Device::hip()), targets_cpu.to(Device::hip())).to_vector(),
                cross_entropy(logits_cpu, targets_cpu).to_vector());
}

TEST(HipOpsTest, MaskedCrossEntropyMatchesCpuReference) {
    require_gpu();
    const auto logits = Tensor::from_vector({2, 1, 0, 100, -100, 0}, {2, 3});
    const auto targets = Tensor::from_int32_vector({0, -100}, {2});
    expect_near(cross_entropy(logits.to(Device::hip()), targets.to(Device::hip())).to_vector(),
                cross_entropy(logits, targets).to_vector());
}

TEST(HipCrossEntropyTest, ParallelForwardBackwardCoverLargeVocabularyWithoutHostTransfers) {
    require_gpu();
    const auto gpu = Device::hip(0);
    for (const auto& [rows, classes] :
         {std::pair<std::int64_t, std::int64_t>{1, 2}, {3, 32}, {32, 8192},
          {3, 151936}}) {
        std::vector<float> values(static_cast<std::size_t>(rows * classes));
        std::vector<std::int32_t> labels(static_cast<std::size_t>(rows));
        for (std::size_t index = 0; index < values.size(); ++index) {
            values[index] = static_cast<float>(static_cast<int>(index % 251) - 125) * 0.03125F;
        }
        for (std::int64_t row = 0; row < rows; ++row) {
            labels[static_cast<std::size_t>(row)] =
                static_cast<std::int32_t>((row * 7919 + 17) % classes);
        }
        if (rows > 1) labels.back() = -100;
        const auto logits = Tensor::from_vector(values, {rows, classes});
        const auto targets = Tensor::from_int32_vector(labels, {rows});
        const auto seed = Tensor::from_vector({0.75F}, {});
        const auto expected_loss = cross_entropy(logits, targets).to_vector();
        const auto expected_gradient = cross_entropy_backward(logits, targets, seed).to_vector();
        const auto device_logits = logits.to(gpu);
        const auto device_targets = targets.to(gpu);
        const auto device_seed = seed.to(gpu);
        runtime::reset_transfer_stats();
        const auto actual_loss = cross_entropy(device_logits, device_targets);
        const auto actual_gradient =
            cross_entropy_backward(device_logits, device_targets, device_seed);
        runtime::synchronize(gpu);
        const auto transfers = runtime::transfer_stats();
        EXPECT_EQ(transfers.host_to_device_calls, 0U);
        EXPECT_EQ(transfers.device_to_host_calls, 0U);
        expect_near(actual_loss.to_vector(), expected_loss, 3.0e-5F);
        expect_near(actual_gradient.to_vector(), expected_gradient, 3.0e-5F);
    }
}

TEST(HipBackwardOpsTest, DeviceNativePrimitivesMatchCpuReference) {
    require_gpu();
    const auto gpu = Device::hip();
    const auto input = Tensor::from_vector({-2, -1, 0, 1, 2, 3}, {2, 3});
    const auto weight = Tensor::from_vector({1, 0.5F, 2}, {3});
    const auto gradient = Tensor::from_vector({1, 2, 3, -1, -2, -3}, {2, 3});

    expect_near(reduce_sum(input.to(gpu)).to_vector(), reduce_sum(input).to_vector());
    expect_near(broadcast_scalar(Tensor::from_vector({2.5F}, {}).to(gpu), {2, 3}).to_vector(),
                std::vector<float>(6, 2.5F));
    expect_near(silu_backward(input.to(gpu), gradient.to(gpu)).to_vector(),
                silu_backward(input, gradient).to_vector());

    const auto indices = Tensor::from_int32_vector({2, 0, 2}, {3});
    const auto embedding_gradient =
        Tensor::from_vector({1, 2, 3, 4, 5, 6}, {3, 2});
    expect_near(embedding_backward(embedding_gradient.to(gpu), indices.to(gpu), 4).to_vector(),
                embedding_backward(embedding_gradient, indices, 4).to_vector());

    const auto cpu_rms = rms_norm_backward(input, weight, gradient);
    const auto hip_rms = rms_norm_backward(input.to(gpu), weight.to(gpu), gradient.to(gpu));
    expect_near(hip_rms.first.to_vector(), cpu_rms.first.to_vector(), 2.0e-4F);
    expect_near(hip_rms.second.to_vector(), cpu_rms.second.to_vector(), 2.0e-4F);

    const auto up = Tensor::from_vector({1, 2, 3, 4, 5, 6}, {2, 3});
    const auto cpu_swiglu = swiglu_backward(input, up, gradient);
    const auto hip_swiglu = swiglu_backward(input.to(gpu), up.to(gpu), gradient.to(gpu));
    expect_near(hip_swiglu.first.to_vector(), cpu_swiglu.first.to_vector());
    expect_near(hip_swiglu.second.to_vector(), cpu_swiglu.second.to_vector());

    const auto probabilities = softmax(input);
    expect_near(softmax_backward(probabilities.to(gpu), gradient.to(gpu)).to_vector(),
                softmax_backward(probabilities, gradient).to_vector());

    const auto logits = Tensor::from_vector({2, 1, 0, 100, -100, 0}, {2, 3});
    const auto targets = Tensor::from_int32_vector({0, -100}, {2});
    const auto seed = Tensor::from_vector({0.75F}, {});
    expect_near(cross_entropy_backward(logits.to(gpu), targets.to(gpu), seed.to(gpu)).to_vector(),
                cross_entropy_backward(logits, targets, seed).to_vector());

    const auto rope_gradient =
        Tensor::from_vector({1, 2, 3, 4, -1, -2, -3, -4}, {1, 2, 1, 4});
    expect_near(rope_backward(rope_gradient.to(gpu)).to_vector(),
                rope_backward(rope_gradient).to_vector(), 2.0e-5F);
    expect_near(rope_split_half_backward(rope_gradient.to(gpu)).to_vector(),
                rope_split_half_backward(rope_gradient).to_vector(), 2.0e-5F);

    const auto scores = Tensor::from_vector({1, 2, 3, 4, 5, 6, 7, 8, 9}, {1, 3, 3});
    const auto score_gradient =
        Tensor::from_vector({1, 2, 3, -1, 0, 1, 2, -2, 0.5F}, {1, 3, 3});
    const auto cpu_causal = causal_softmax(scores);
    const auto hip_causal = causal_softmax(scores.to(gpu));
    expect_near(hip_causal.to_vector(), cpu_causal.to_vector());
    expect_near(causal_softmax_backward(hip_causal, score_gradient.to(gpu)).to_vector(),
                causal_softmax_backward(cpu_causal, score_gradient).to_vector());

    const auto repeat_input = Tensor::from_vector({1, 2, 3, 4}, {2, 2});
    const auto repeat_gradient = Tensor::from_vector({1, 2, 3, 4, 5, 6, 7, 8}, {4, 2});
    expect_near(repeat_interleave(repeat_input.to(gpu), 0, 2).to_vector(),
                repeat_interleave(repeat_input, 0, 2).to_vector());
    expect_near(repeat_interleave_backward(repeat_gradient.to(gpu), {2, 2}, 0, 2).to_vector(),
                repeat_interleave_backward(repeat_gradient, {2, 2}, 0, 2).to_vector());
}

TEST(HipAutogradTest, RepeatInterleaveMaterializesTransposedGqaValue) {
    require_gpu();
    autograd::Value input(
        Tensor::from_vector({1, 2, 3, 4, 5, 6, 7, 8}, {1, 2, 2, 2}).to(Device::hip()),
        true);
    const auto transposed = autograd::transpose(input, 1, 2);
    const auto repeated = autograd::repeat_interleave(transposed, 1, 2);
    EXPECT_TRUE(repeated.data().is_contiguous());
    autograd::sum(repeated).backward();
    expect_near(input.grad().to_vector(), std::vector<float>(8, 2.0F));
}

TEST(HipOpsTest, ExplicitStreamContextOrdersKernelAndEvent) {
    require_gpu();
    const auto gpu = Device::hip();
    const auto left = Tensor::from_vector({1, 2, 3, 4}, {4}).to(gpu);
    const auto right = Tensor::from_vector({4, 3, 2, 1}, {4}).to(gpu);
    runtime::Stream stream(gpu);
    runtime::Event completion(gpu);
    const OpContext context{&stream, nullptr, 0};
    const auto output = add(left, right, context);
    completion.record(stream);
    completion.synchronize();
    EXPECT_EQ(output.to_vector(), (std::vector<float>{5, 5, 5, 5}));
}

TEST(HipOpsTest, ExplicitStreamRejectsDeviceMismatch) {
    require_gpu();
    const auto gpu = Tensor::from_vector({1, 2}, {2}).to(Device::hip());
    runtime::Stream cpu_stream(Device::cpu());
    const OpContext context{&cpu_stream, nullptr, 0};
    EXPECT_THROW((void)add(gpu, gpu, context), std::invalid_argument);
}

TEST(HipModelTest, PreallocatedGqaCacheMatchesCpuAndAvoidsPayloadTransfers) {
    require_gpu();
    const model::ModelConfig config{.vocabulary_size = 16,
                                    .dimension = 8,
                                    .layers = 1,
                                    .heads = 2,
                                    .kv_heads = 1,
                                    .ffn_dimension = 16,
                                    .max_sequence_length = 4,
                                    .rope_base = 10000.0F,
                                    .tie_embeddings = false};
    model::TransformerModel cpu_model(config, 61);
    inference::KVCache cpu_cache(config.layers, config.max_sequence_length);
    model::TransformerModel hip_model(config, 61);
    hip_model.to(Device::hip());
    inference::KVCache hip_cache(config.layers, config.max_sequence_length);
    const std::vector<std::int32_t> tokens{3, 4, 5, 6};
    const void* key_address = nullptr;
    const void* value_address = nullptr;
    for (const auto token : tokens) {
        const auto host_token = Tensor::from_int32_vector({token}, {1, 1});
        const auto expected = cpu_model.forward_cached(host_token, cpu_cache).to_vector();
        const auto device_token = host_token.to(Device::hip());
        runtime::reset_transfer_stats();
        const auto actual = hip_model.forward_cached(device_token, hip_cache);
        runtime::synchronize(Device::hip());
        const auto transfers = runtime::transfer_stats();
        EXPECT_EQ(transfers.host_to_device_calls, 0U);
        EXPECT_EQ(transfers.device_to_host_calls, 0U);
        expect_near(actual.to_vector(), expected, 3.0e-4F);
        if (key_address == nullptr) {
            key_address = hip_cache.layer(0).key.storage().data();
            value_address = hip_cache.layer(0).value.storage().data();
        } else {
            EXPECT_EQ(hip_cache.layer(0).key.storage().data(), key_address);
            EXPECT_EQ(hip_cache.layer(0).value.storage().data(), value_address);
        }
    }
}

TEST(HipTensorTest, NonContiguousTransposeMaterializesInLogicalOrder) {
    require_gpu();
    const auto cpu = Tensor::from_vector({0, 1, 2, 3, 4, 5}, {2, 3});
    const auto transposed = cpu.to(Device::hip()).transpose(0, 1);
    ASSERT_FALSE(transposed.is_contiguous());
    const auto contiguous = transposed.contiguous();
    EXPECT_TRUE(contiguous.is_contiguous());
    EXPECT_EQ(contiguous.to_vector(), (std::vector<float>{0, 3, 1, 4, 2, 5}));
}

TEST(HipTrainingTest, TinyTransformerRunsBackwardAndLowersLoss) {
    require_gpu();
    const model::ModelConfig config{.vocabulary_size = 8,
                                    .dimension = 8,
                                    .layers = 1,
                                    .heads = 2,
                                    .kv_heads = 1,
                                    .ffn_dimension = 16,
                                    .max_sequence_length = 4,
                                    .rope_base = 10000.0F,
                                    .tie_embeddings = false};
    model::TransformerModel model(config, 71);
    model.to(Device::hip());
    training::AdamW optimizer(model.parameters(), {.learning_rate = 0.02F,
                                                    .beta1 = 0.9F,
                                                    .beta2 = 0.99F,
                                                    .epsilon = 1.0e-8F,
                                                    .weight_decay = 0.0F});
    const io::TokenBatch batch{Tensor::from_int32_vector({0, 1, 2, 3}, {1, 4}),
                               Tensor::from_int32_vector({1, 2, 3, 0}, {1, 4})};
    float first_loss = 0.0F;
    float final_loss = 0.0F;
    for (std::uint64_t step = 1; step <= 5; ++step) {
        const auto metrics = training::train_step(model, optimizer, batch, step);
        if (step == 1) first_loss = metrics.loss;
        final_loss = metrics.loss;
    }
    EXPECT_LT(final_loss, first_loss);
    EXPECT_EQ(model.device(), Device::hip());
}

TEST(HipTrainingTest, AdamWStepIsDeviceNativeAndMatchesCpu) {
    require_gpu();
    autograd::Value cpu(Tensor::from_vector({1.0F, -2.0F}, {2}), true);
    autograd::Value gpu(cpu.data().to(Device::hip()), true);
    cpu.set_grad(Tensor::from_vector({0.5F, -0.25F}, {2}));
    gpu.set_grad(cpu.grad().to(Device::hip()));
    const training::AdamWConfig config{.learning_rate = 0.01F,
                                       .beta1 = 0.9F,
                                       .beta2 = 0.99F,
                                       .epsilon = 1.0e-8F,
                                       .weight_decay = 0.1F};
    training::AdamW cpu_optimizer({&cpu}, config);
    training::AdamW gpu_optimizer({&gpu}, config);
    cpu_optimizer.step();
    runtime::reset_transfer_stats();
    gpu_optimizer.step();
    runtime::synchronize(Device::hip());
    const auto transfers = runtime::transfer_stats();
    EXPECT_EQ(transfers.host_to_device_calls, 0U);
    EXPECT_EQ(transfers.device_to_host_calls, 0U);
    expect_near(gpu.data().to_vector(), cpu.data().to_vector(), 1.0e-6F);
}

TEST(HipAutogradTest, FullTransformerBackwardMatchesCpuWithoutHostTransfers) {
    require_gpu();
    const model::ModelConfig config{.vocabulary_size = 8,
                                    .dimension = 8,
                                    .layers = 1,
                                    .heads = 2,
                                    .kv_heads = 1,
                                    .ffn_dimension = 16,
                                    .max_sequence_length = 4,
                                    .rope_base = 10000.0F,
                                    .tie_embeddings = false};
    const auto tokens = Tensor::from_int32_vector({0, 1, 2, 3}, {1, 4});
    const auto targets = Tensor::from_int32_vector({1, 2, 3, 0}, {1, 4});

    model::TransformerModel cpu_model(config, 113);
    const auto cpu_loss = cpu_model.loss(tokens, targets);
    cpu_loss.backward();

    model::TransformerModel hip_model(config, 113);
    hip_model.to(Device::hip());
    const auto hip_tokens = tokens.to(Device::hip());
    const auto hip_targets = targets.to(Device::hip());
    runtime::reset_transfer_stats();
    const auto hip_loss = hip_model.loss(hip_tokens, hip_targets);
    hip_loss.backward();
    runtime::synchronize(Device::hip());
    const auto transfers = runtime::transfer_stats();
    EXPECT_EQ(transfers.host_to_device_calls, 0U);
    EXPECT_EQ(transfers.device_to_host_calls, 0U);

    expect_near(hip_loss.data().to_vector(), cpu_loss.data().to_vector(), 3.0e-4F);
    const auto cpu_parameters = cpu_model.named_parameters();
    const auto hip_parameters = hip_model.named_parameters();
    ASSERT_EQ(cpu_parameters.size(), hip_parameters.size());
    for (std::size_t index = 0; index < cpu_parameters.size(); ++index) {
        EXPECT_EQ(cpu_parameters[index].first, hip_parameters[index].first);
        ASSERT_TRUE(cpu_parameters[index].second->has_grad()) << cpu_parameters[index].first;
        ASSERT_TRUE(hip_parameters[index].second->has_grad()) << hip_parameters[index].first;
        expect_near(hip_parameters[index].second->grad().to_vector(),
                    cpu_parameters[index].second->grad().to_vector(), 2.0e-3F);
    }
}

TEST(HipTensorViewTest, UsesCallerOwnedBuffersAndExplicitStream) {
    require_gpu();
    const auto gpu = Device::hip();
    const auto left = Tensor::from_vector({1, 2, 3, 4}, {2, 2}).to(gpu);
    const auto right = Tensor::from_vector({5, 6, 7, 8}, {2, 2}).to(gpu);
    Tensor output({2, 2}, DType::Float32, gpu);
    runtime::Stream stream(gpu);
    const auto context = OpContext::from_external_stream(gpu, stream.native_handle());
    add_out(output.view(), left.view(), right.view(), context);
    stream.synchronize();
    EXPECT_EQ(output.to_vector(), (std::vector<float>{6, 8, 10, 12}));
}

#if MICROLLM_HAS_HIPBLASLT
TEST(HipOptimizedOpsTest, HipblasLtMatmulMatchesReadableReference) {
    require_gpu();
    std::vector<float> left_values(64 * 64);
    std::vector<float> right_values(64 * 64);
    for (std::size_t index = 0; index < left_values.size(); ++index) {
        left_values[index] = static_cast<float>(index % 13) / 13.0F;
        right_values[index] = static_cast<float>(index % 7) / 7.0F;
    }
    const auto left_cpu = Tensor::from_vector(left_values, {64, 64});
    const auto right_cpu = Tensor::from_vector(right_values, {64, 64});
    const auto expected = matmul(left_cpu, right_cpu).to_vector();
    const auto actual = matmul_with_implementation(
                            left_cpu.to(Device::hip()), right_cpu.to(Device::hip()),
                            MatmulImplementation::HipBLASLt)
                            .to_vector();
    expect_near(actual, expected, 2.0e-4F);
    for (const auto dtype : {DType::Float16, DType::BFloat16}) {
        const auto low_left = Tensor::from_vector(left_values, {64, 64}, dtype);
        const auto low_right = Tensor::from_vector(right_values, {64, 64}, dtype);
        const auto low_expected = matmul(low_left, low_right).to_vector();
        runtime::reset_transfer_stats();
        const auto low_actual_tensor = matmul_with_implementation(
            low_left.to(Device::hip()), low_right.to(Device::hip()),
            MatmulImplementation::HipBLASLt);
        runtime::synchronize(Device::hip());
        const auto tolerance = dtype == DType::Float16 ? 3.0e-2F : 2.0e-1F;
        expect_near(low_actual_tensor.to_vector(), low_expected, tolerance);
        EXPECT_EQ(low_actual_tensor.dtype(), dtype);
    }
    EXPECT_EQ(choose_matmul_implementation(left_cpu.to(Device::hip()),
                                           right_cpu.to(Device::hip())),
              MatmulImplementation::Readable);
    const Tensor large_left({128, 128}, DType::Float32, Device::hip());
    const Tensor large_right({128, 128}, DType::Float32, Device::hip());
    EXPECT_EQ(choose_matmul_implementation(large_left, large_right),
              MatmulImplementation::HipBLASLt);
    register_matmul_implementation(64, 64, 64, MatmulImplementation::HipBLASLt);
    EXPECT_EQ(choose_matmul_implementation(left_cpu.to(Device::hip()),
                                           right_cpu.to(Device::hip())),
              MatmulImplementation::HipBLASLt);
    clear_matmul_implementation_registry();
    EXPECT_EQ(choose_matmul_implementation(left_cpu.to(Device::hip()),
                                           right_cpu.to(Device::hip())),
              MatmulImplementation::Readable);
}


TEST(HipOptimizedOpsTest, TransposeAwareGemmCoversAllLayoutsAndLowPrecisions) {
    require_gpu();
    const auto gpu = Device::hip();
    const std::vector<float> left_values{1, 2, 3, 4, 5, 6};
    const std::vector<float> right_values{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
    const std::vector<float> left_transposed_values{1, 4, 2, 5, 3, 6};
    const std::vector<float> right_transposed_values{
        1, 5, 9, 2, 6, 10, 3, 7, 11, 4, 8, 12};
    const std::vector<float> expected{38, 44, 50, 56, 83, 98, 113, 128};
    for (const auto dtype : {DType::Float32, DType::Float16, DType::BFloat16}) {
        const auto left = Tensor::from_vector(left_values, {2, 3}, dtype).to(gpu);
        const auto right = Tensor::from_vector(right_values, {3, 4}, dtype).to(gpu);
        const auto left_t = Tensor::from_vector(left_transposed_values, {3, 2}, dtype).to(gpu);
        const auto right_t = Tensor::from_vector(right_transposed_values, {4, 3}, dtype).to(gpu);
        for (const auto implementation : {MatmulImplementation::Readable,
                                          MatmulImplementation::HipBLASLt}) {
            for (const auto& test : {
                     std::tuple<const Tensor*, const Tensor*, bool, bool>{
                         &left, &right, false, false},
                     std::tuple<const Tensor*, const Tensor*, bool, bool>{
                         &left, &right_t, false, true},
                     std::tuple<const Tensor*, const Tensor*, bool, bool>{
                         &left_t, &right, true, false},
                     std::tuple<const Tensor*, const Tensor*, bool, bool>{
                         &left_t, &right_t, true, true}}) {
                runtime::reset_transfer_stats();
                const auto actual = matmul_with_implementation(
                    *std::get<0>(test), *std::get<1>(test), implementation,
                    std::get<2>(test), std::get<3>(test));
                runtime::synchronize(gpu);
                const auto transfers = runtime::transfer_stats();
                EXPECT_EQ(transfers.host_to_device_calls, 0U);
                EXPECT_EQ(transfers.device_to_host_calls, 0U);
                const auto tolerance = dtype == DType::BFloat16 ? 0.55F
                                      : dtype == DType::Float16 ? 0.08F : 3.0e-4F;
                expect_near(actual.to_vector(), expected, tolerance);
                EXPECT_EQ(actual.dtype(), dtype);
            }
        }
    }
}
#endif

}  // namespace microllm::ops
