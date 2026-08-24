#include <algorithm>
#include <cmath>
#include <filesystem>
#include <limits>
#include <tuple>
#include <utility>
#include <vector>

#include <gtest/gtest.h>
#include <microllm/ops/ops.h>
#include <microllm/ops/low_level.h>
#include <microllm/ops/tuning.h>
#include <microllm/runtime/runtime.h>
#include <microllm/runtime/memory.h>
#include <microllm/inference/generator.h>
#include <microllm/inference/scheduler.h>
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

TEST(HipOpsTest, InPlaceAddPreservesStorageWithoutPayloadTransfers) {
    require_gpu();
    const auto gpu = Device::hip(0);
    auto destination = Tensor::from_vector({1, -2, 3, 4}, {2, 2}).to(gpu);
    const auto source = Tensor::from_vector({4, 5, -6, 2}, {2, 2}).to(gpu);
    const auto* address = destination.storage().data();
    runtime::reset_transfer_stats();
    add_in_place_(destination, source);
    runtime::synchronize(gpu);
    const auto transfers = runtime::transfer_stats();
    EXPECT_EQ(destination.storage().data(), address);
    EXPECT_EQ(transfers.host_to_device_calls, 0U);
    EXPECT_EQ(transfers.device_to_host_calls, 0U);
    EXPECT_EQ(transfers.device_to_device_calls, 0U);
    EXPECT_EQ(destination.to_vector(), (std::vector<float>{5, 3, -3, 6}));
}

TEST(HipOpsTest, MatmulOutUsesCallerStorageWithoutPayloadTransfers) {
    require_gpu();
    const auto gpu = Device::hip(0);
    const auto left_cpu = Tensor::from_vector({1, 2, 3, 4, 5, 6}, {2, 3});
    const auto right_cpu = Tensor::from_vector({7, 8, 9, 10, 11, 12}, {3, 2});
    const auto left = left_cpu.to(gpu);
    const auto right = right_cpu.to(gpu);
    Tensor output({2, 2}, DType::Float32, gpu);
    const auto* address = output.storage().data();
    runtime::reset_transfer_stats();
    matmul_out_(output, left, right, MatmulImplementation::HipBLASLt);
    runtime::synchronize(gpu);
    const auto transfers = runtime::transfer_stats();
    EXPECT_EQ(output.storage().data(), address);
    EXPECT_EQ(transfers.host_to_device_calls, 0U);
    EXPECT_EQ(transfers.device_to_host_calls, 0U);
    EXPECT_EQ(transfers.device_to_device_calls, 0U);
    expect_near(output.to_vector(), matmul(left_cpu, right_cpu).to_vector());
}

TEST(HipGraphMatmulTest, CallerOwnedHipblasLtOutputCapturesAndReplays) {
    require_gpu();
    const auto gpu = Device::hip(0);
    constexpr std::int64_t kSize = 32;
    std::vector<float> left_values(static_cast<std::size_t>(kSize * kSize));
    std::vector<float> right_values(static_cast<std::size_t>(kSize * kSize));
    for (std::size_t index = 0; index < left_values.size(); ++index) {
        left_values[index] = static_cast<float>(static_cast<int>(index % 17) - 8) / 17.0F;
        right_values[index] = static_cast<float>(static_cast<int>(index % 13) - 6) / 13.0F;
    }
    const auto expected = matmul(
        Tensor::from_vector(left_values, {kSize, kSize}),
        Tensor::from_vector(right_values, {kSize, kSize})).to_vector();
    const auto left = Tensor::from_vector(left_values, {kSize, kSize}).to(gpu);
    const auto right = Tensor::from_vector(right_values, {kSize, kSize}).to(gpu);
    Tensor output({kSize, kSize}, DType::Float32, gpu);
    const auto* address = output.storage().data();
    runtime::Stream stream(gpu);
    OpContext context;
    context.stream = &stream;
    matmul_out_(output, left, right, MatmulImplementation::HipBLASLt,
                false, false, context);
    stream.synchronize();

    const auto work = [&] {
        matmul_out_(output, left, right, MatmulImplementation::HipBLASLt,
                    false, false, context);
    };
    auto graph = runtime::HipGraphExecutable::capture(stream, work);
    EXPECT_GT(graph.node_count(), 0U);
    runtime::reset_transfer_stats();
    graph.launch(stream);
    graph.launch(stream);
    stream.synchronize();
    const auto transfers = runtime::transfer_stats();
    EXPECT_EQ(output.storage().data(), address);
    EXPECT_EQ(transfers.host_to_device_calls, 0U);
    EXPECT_EQ(transfers.device_to_host_calls, 0U);
    EXPECT_EQ(transfers.device_to_device_calls, 0U);
    expect_near(output.to_vector(), expected, 2.0e-4F);
}

TEST(HipOpsTest, CastOutAndTransposeWriteCallerStorageWithoutPayloadTransfers) {
    require_gpu();
    const auto gpu = Device::hip(0);
    const auto input = Tensor::from_vector({1, 2, 3, 4, 5, 6}, {2, 3},
                                           DType::BFloat16).to(gpu);
    Tensor casted({2, 3}, DType::Float32, gpu);
    Tensor transposed({3, 2}, DType::Float32, gpu);
    const auto* cast_address = casted.storage().data();
    const auto* transpose_address = transposed.storage().data();
    runtime::reset_transfer_stats();
    cast_out_(input, casted);
    cast_transpose_2d_out_(input, transposed);
    runtime::synchronize(gpu);
    const auto transfers = runtime::transfer_stats();
    EXPECT_EQ(transfers.host_to_device_calls, 0U);
    EXPECT_EQ(transfers.device_to_host_calls, 0U);
    EXPECT_EQ(casted.storage().data(), cast_address);
    EXPECT_EQ(transposed.storage().data(), transpose_address);
    EXPECT_EQ(casted.to_vector(), (std::vector<float>{1, 2, 3, 4, 5, 6}));
    EXPECT_EQ(transposed.to_vector(), (std::vector<float>{1, 4, 2, 5, 3, 6}));
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

TEST(HipOpsTest, CooperativeBiasGradientCoversThresholdTailsAndModelWidths) {
    require_gpu();
    const auto gpu = Device::hip(0);
    for (const auto& [rows, width] : {
             std::pair<std::int64_t, std::int64_t>{1, 16},
             {3, 128}, {31, 127}, {32, 127}, {255, 127}, {256, 128},
             {511, 896}, {512, 1536}, {1024, 256}}) {
        std::vector<float> values(static_cast<std::size_t>(rows * width));
        for (std::size_t index = 0; index < values.size(); ++index) {
            values[index] =
                static_cast<float>(static_cast<int>(index % 29) - 14) / 31.0F;
        }
        const auto cpu = Tensor::from_vector(values, {rows, width});
        const auto expected = bias_gradient(cpu).to_vector();
        const auto input = cpu.to(gpu);
        runtime::reset_transfer_stats();
        const auto scalar = bias_gradient_with_implementation(
            input, BiasGradientImplementation::ScalarColumns);
        const auto cooperative = bias_gradient_with_implementation(
            input, BiasGradientImplementation::CooperativeRows);
        const auto automatic = bias_gradient(input);
        runtime::synchronize(gpu);
        const auto transfers = runtime::transfer_stats();
        EXPECT_EQ(transfers.host_to_device_calls, 0U);
        EXPECT_EQ(transfers.device_to_host_calls, 0U);
        expect_near(scalar.to_vector(), expected, 2.0e-5F);
        expect_near(cooperative.to_vector(), expected, 3.0e-5F);
        expect_near(automatic.to_vector(), expected, 3.0e-5F);
        if (rows < 32) {
            EXPECT_EQ(automatic.to_vector(), scalar.to_vector());
        } else {
            EXPECT_EQ(automatic.to_vector(), cooperative.to_vector());
        }
    }
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
        Tensor caller_gated(left.shape(), dtype, gpu);
        swiglu_out_(caller_gated, left, right);
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
        expect_near(caller_gated.to_vector(), gated.to_vector(), tolerance);
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
    const auto bf16_output = bf16_matmul_output(
        cast(device_left, DType::BFloat16), device_right_bf16, DType::BFloat16);
    EXPECT_EQ(bf16_output.dtype(), DType::BFloat16);
    expect_near(bf16_output.to_vector(),
                bf16_matmul_output(cast(left, DType::BFloat16), rounded_right,
                                   DType::BFloat16).to_vector(), 2.0e-2F);

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

TEST(HipBf16FfnTest, ContinuousIslandMatchesCpuAndAvoidsHostTransfers) {
    require_gpu();
    if (!hipblaslt_available()) GTEST_SKIP() << "hipBLASLt is unavailable";
    constexpr std::int64_t tokens = 3;
    constexpr std::int64_t hidden = 128;
    constexpr std::int64_t intermediate = 256;
    std::vector<float> input_values(tokens * hidden);
    std::vector<float> gate_values(hidden * intermediate);
    std::vector<float> up_values(hidden * intermediate);
    std::vector<float> down_values(intermediate * hidden);
    for (std::size_t index = 0; index < input_values.size(); ++index) {
        input_values[index] = static_cast<float>(static_cast<int>(index % 29) - 14) / 32.0F;
    }
    for (std::size_t index = 0; index < gate_values.size(); ++index) {
        gate_values[index] = static_cast<float>(static_cast<int>(index % 31) - 15) / 128.0F;
        up_values[index] = static_cast<float>(static_cast<int>(index % 37) - 18) / 128.0F;
    }
    for (std::size_t index = 0; index < down_values.size(); ++index) {
        down_values[index] = static_cast<float>(static_cast<int>(index % 41) - 20) / 256.0F;
    }
    const auto input_cpu = Tensor::from_vector(input_values, {tokens, hidden});
    const auto gate_cpu = Tensor::from_vector(
        gate_values, {hidden, intermediate}, DType::BFloat16);
    const auto up_cpu = Tensor::from_vector(
        up_values, {hidden, intermediate}, DType::BFloat16);
    const auto down_cpu = Tensor::from_vector(
        down_values, {intermediate, hidden}, DType::BFloat16);
    const auto expected = bf16_ffn(input_cpu, gate_cpu, up_cpu, down_cpu);
    const auto gpu = Device::hip(0);
    const auto input = input_cpu.to(gpu);
    const auto gate = gate_cpu.to(gpu);
    const auto up = up_cpu.to(gpu);
    const auto down = down_cpu.to(gpu);

    runtime::reset_transfer_stats();
    const auto diagnostics = bf16_ffn_diagnostics(input, gate, up, down);
    const auto actual = diagnostics.output;
    runtime::synchronize(gpu);
    const auto transfers = runtime::transfer_stats();
    EXPECT_EQ(transfers.host_to_device_calls, 0U);
    EXPECT_EQ(transfers.device_to_host_calls, 0U);
    EXPECT_EQ(actual.dtype(), DType::Float32);
    EXPECT_EQ(diagnostics.input_bf16.dtype(), DType::BFloat16);
    EXPECT_EQ(diagnostics.gate.dtype(), DType::BFloat16);
    EXPECT_EQ(diagnostics.up.dtype(), DType::BFloat16);
    EXPECT_EQ(diagnostics.activated.dtype(), DType::BFloat16);
    EXPECT_EQ(actual.shape(), (Shape{tokens, hidden}));
    expect_near(actual.to_vector(), expected.to_vector(), 7.5e-2F);

    Bf16FfnWorkspace workspace{
        .input_bf16 = Tensor({tokens, hidden}, DType::BFloat16, gpu),
        .gate = Tensor({tokens, intermediate}, DType::BFloat16, gpu),
        .up = Tensor({tokens, intermediate}, DType::BFloat16, gpu),
        .activated = Tensor({tokens, intermediate}, DType::BFloat16, gpu),
        .output_fallback_bf16 =
            Tensor({tokens, hidden}, DType::BFloat16, gpu)};
    Tensor caller_output({tokens, hidden}, DType::Float32, gpu);
    runtime::reset_transfer_stats();
    bf16_ffn_out_(caller_output, workspace, input, gate, up, down);
    runtime::synchronize(gpu);
    const auto caller_transfers = runtime::transfer_stats();
    EXPECT_EQ(caller_transfers.host_to_device_calls, 0U);
    EXPECT_EQ(caller_transfers.device_to_host_calls, 0U);
    expect_near(caller_output.to_vector(), actual.to_vector(), 0.0F);
}

TEST(HipBf16FfnTest, QwenDecodeShapeFallsBackToDeviceCastAndRemainsReusable) {
    require_gpu();
    if (!hipblaslt_available()) GTEST_SKIP() << "hipBLASLt is unavailable";
    constexpr std::int64_t hidden = 896;
    constexpr std::int64_t intermediate = 4864;
    const auto gpu = Device::hip(0);
    auto input = Tensor({1, hidden}, DType::Float32, gpu);
    auto gate = Tensor({hidden, intermediate}, DType::BFloat16, gpu);
    auto up = Tensor({hidden, intermediate}, DType::BFloat16, gpu);
    auto down = Tensor({intermediate, hidden}, DType::BFloat16, gpu);
    fill_(input, 0.0F);
    fill_(gate, 0.0F);
    fill_(up, 0.0F);
    fill_(down, 0.0F);
    runtime::reset_transfer_stats();
    const auto first = bf16_ffn(input, gate, up, down);
    const auto second = bf16_ffn(input, gate, up, down);
    Bf16FfnWorkspace workspace{
        .input_bf16 = Tensor({1, hidden}, DType::BFloat16, gpu),
        .gate = Tensor({1, intermediate}, DType::BFloat16, gpu),
        .up = Tensor({1, intermediate}, DType::BFloat16, gpu),
        .activated = Tensor({1, intermediate}, DType::BFloat16, gpu),
        .output_fallback_bf16 = Tensor({1, hidden}, DType::BFloat16, gpu)};
    Tensor caller_output({1, hidden}, DType::Float32, gpu);
    bf16_ffn_out_(caller_output, workspace, input, gate, up, down);
    runtime::synchronize(gpu);
    const auto transfers = runtime::transfer_stats();
    EXPECT_EQ(transfers.host_to_device_calls, 0U);
    EXPECT_EQ(transfers.device_to_host_calls, 0U);
    EXPECT_EQ(first.dtype(), DType::Float32);
    EXPECT_EQ(second.dtype(), DType::Float32);
    expect_near(first.to_vector(), std::vector<float>(hidden, 0.0F), 0.0F);
    expect_near(second.to_vector(), first.to_vector(), 0.0F);
    expect_near(caller_output.to_vector(), first.to_vector(), 0.0F);
}

TEST(HipBf16ProjectionTest, SharedQkvCastMatchesThreeCpuMixedGemms) {
    require_gpu();
    if (!hipblaslt_available()) GTEST_SKIP() << "hipBLASLt is unavailable";
    constexpr std::int64_t hidden = 128;
    constexpr std::int64_t kv = 32;
    const auto gpu = Device::hip(0);
    std::vector<float> input_values(hidden);
    std::vector<float> query_values(hidden * hidden);
    std::vector<float> key_values(hidden * kv);
    std::vector<float> value_values(hidden * kv);
    for (std::size_t index = 0; index < query_values.size(); ++index) {
        query_values[index] = static_cast<float>(static_cast<int>(index % 29) - 14) / 128.0F;
    }
    for (std::size_t index = 0; index < key_values.size(); ++index) {
        key_values[index] = static_cast<float>(static_cast<int>(index % 31) - 15) / 128.0F;
        value_values[index] = static_cast<float>(static_cast<int>(index % 37) - 18) / 128.0F;
    }
    for (std::size_t index = 0; index < input_values.size(); ++index) {
        input_values[index] = static_cast<float>(static_cast<int>(index % 23) - 11) / 32.0F;
    }
    const auto input_cpu = Tensor::from_vector(input_values, {1, hidden});
    const auto query_cpu = Tensor::from_vector(query_values, {hidden, hidden},
                                                DType::BFloat16);
    const auto key_cpu = Tensor::from_vector(key_values, {hidden, kv}, DType::BFloat16);
    const auto value_cpu = Tensor::from_vector(value_values, {hidden, kv}, DType::BFloat16);
    const auto expected = bf16_qkv_projection(input_cpu, query_cpu, key_cpu, value_cpu);
    const auto input = input_cpu.to(gpu);
    const auto query = query_cpu.to(gpu);
    const auto key = key_cpu.to(gpu);
    const auto value = value_cpu.to(gpu);
    runtime::reset_transfer_stats();
    const auto actual = bf16_qkv_projection(input, query, key, value);
    runtime::synchronize(gpu);
    const auto transfers = runtime::transfer_stats();
    EXPECT_EQ(transfers.host_to_device_calls, 0U);
    EXPECT_EQ(transfers.device_to_host_calls, 0U);
    expect_near(actual.first.to_vector(), expected.first.to_vector(), 3.0e-2F);
    expect_near(actual.second.to_vector(), expected.second.to_vector(), 3.0e-2F);
    expect_near(actual.third.to_vector(), expected.third.to_vector(), 3.0e-2F);
    Bf16QkvWorkspace workspace{
        .input_bf16 = Tensor({1, hidden}, DType::BFloat16, gpu),
        .query_fallback_bf16 = Tensor({1, hidden}, DType::BFloat16, gpu),
        .key_fallback_bf16 = Tensor({1, kv}, DType::BFloat16, gpu),
        .value_fallback_bf16 = Tensor({1, kv}, DType::BFloat16, gpu)};
    Tensor query_output({1, hidden}, DType::Float32, gpu);
    Tensor key_output({1, kv}, DType::Float32, gpu);
    Tensor value_output({1, kv}, DType::Float32, gpu);
    runtime::reset_transfer_stats();
    bf16_qkv_projection_out_(
        query_output, key_output, value_output, workspace,
        input, query, key, value);
    runtime::synchronize(gpu);
    const auto caller_transfers = runtime::transfer_stats();
    EXPECT_EQ(caller_transfers.host_to_device_calls, 0U);
    EXPECT_EQ(caller_transfers.device_to_host_calls, 0U);
    expect_near(query_output.to_vector(), actual.first.to_vector(), 0.0F);
    expect_near(key_output.to_vector(), actual.second.to_vector(), 0.0F);
    expect_near(value_output.to_vector(), actual.third.to_vector(), 0.0F);
}

TEST(HipBf16ProjectionTest, GroupedQkvCachesExactPointerStablePlan) {
    require_gpu();
    if (!hipblaslt_available()) GTEST_SKIP() << "hipBLASLt is unavailable";
    const auto gpu = Device::hip(0);
    if (!runtime::device_info(gpu).architecture.starts_with("gfx942") ||
        hipblaslt_version() != 10300) {
        GTEST_SKIP() << "recorded grouped solution is local to gfx942 hipBLASLt 1.3.0";
    }
    constexpr std::int64_t rows = 512;
    constexpr std::int64_t hidden = 896;
    constexpr std::int64_t kv = 128;
    Tensor input({rows, hidden}, DType::Float32, gpu);
    Tensor query_weight({hidden, hidden}, DType::BFloat16, gpu);
    Tensor key_weight({hidden, kv}, DType::BFloat16, gpu);
    Tensor value_weight({hidden, kv}, DType::BFloat16, gpu);
    fill_(input, 0.01F);
    fill_(query_weight, 0.01F);
    fill_(key_weight, 0.02F);
    fill_(value_weight, -0.01F);
    const auto workspace = [&] {
        return Bf16QkvWorkspace{
            .input_bf16 = Tensor({rows, hidden}, DType::BFloat16, gpu),
            .query_fallback_bf16 = Tensor({rows, hidden}, DType::BFloat16, gpu),
            .key_fallback_bf16 = Tensor({rows, kv}, DType::BFloat16, gpu),
            .value_fallback_bf16 = Tensor({rows, kv}, DType::BFloat16, gpu)};
    };
    auto baseline_workspace = workspace();
    Tensor baseline_query({rows, hidden}, DType::Float32, gpu);
    Tensor baseline_key({rows, kv}, DType::Float32, gpu);
    Tensor baseline_value({rows, kv}, DType::Float32, gpu);
    clear_bf16_grouped_qkv_registry();
    bf16_qkv_projection_out_(
        baseline_query, baseline_key, baseline_value, baseline_workspace,
        input, query_weight, key_weight, value_weight);

    auto key = make_bf16_grouped_qkv_key(
        rows, hidden, hidden, kv, kv, gpu);
    auto stale = key;
    ++stale.hip_driver_version;
    EXPECT_THROW(register_bf16_grouped_qkv_algorithm(stale, 64699),
                 std::invalid_argument);
    register_bf16_grouped_qkv_algorithm(key, 64699);
    auto candidate_workspace = workspace();
    Tensor candidate_query({rows, hidden}, DType::Float32, gpu);
    Tensor candidate_key({rows, kv}, DType::Float32, gpu);
    Tensor candidate_value({rows, kv}, DType::Float32, gpu);
    runtime::reset_transfer_stats();
    for (int iteration = 0; iteration < 2; ++iteration) {
        bf16_qkv_projection_out_(
            candidate_query, candidate_key, candidate_value,
            candidate_workspace, input, query_weight, key_weight,
            value_weight);
    }
    runtime::synchronize(gpu);
    const auto stats = bf16_grouped_qkv_stats();
    EXPECT_EQ(stats.registered_entries, 1U);
    EXPECT_EQ(stats.plan_entries, 1U);
    EXPECT_EQ(stats.plan_misses, 1U);
    EXPECT_EQ(stats.plan_hits, 1U);
    EXPECT_EQ(stats.dispatches, 2U);
    const auto transfers = runtime::transfer_stats();
    EXPECT_EQ(transfers.host_to_device_calls, 0U);
    EXPECT_EQ(transfers.device_to_host_calls, 0U);
    expect_near(candidate_query.to_vector(), baseline_query.to_vector(), 1.0e-2F);
    expect_near(candidate_key.to_vector(), baseline_key.to_vector(), 1.0e-2F);
    expect_near(candidate_value.to_vector(), baseline_value.to_vector(), 1.0e-2F);
    clear_bf16_grouped_qkv_registry();
    EXPECT_EQ(bf16_grouped_qkv_stats().registered_entries, 0U);
}

TEST(HipBf16PlanCacheTest, ExactShapeMissesOnceThenReusesImmutableDescriptors) {
    require_gpu();
    if (!hipblaslt_available()) GTEST_SKIP() << "hipBLASLt is unavailable";
    const auto gpu = Device::hip(0);
    auto left = Tensor({64, 128}, DType::BFloat16, gpu);
    auto right = Tensor({128, 256}, DType::BFloat16, gpu);
    fill_(left, 0.25F);
    fill_(right, -0.125F);
    clear_bf16_plan_cache();
    clear_bf16_algorithm_registry();
    EXPECT_EQ(bf16_registered_algorithm_count(), 0U);
    EXPECT_THROW(register_bf16_algorithm(
                     0, 128, 256, DType::BFloat16, 1),
                 std::invalid_argument);
    EXPECT_EQ(bf16_plan_cache_stats().entries, 0U);
    (void)bf16_matmul_output(left, right, DType::BFloat16);
    runtime::synchronize(gpu);
    const auto first = bf16_plan_cache_stats();
    EXPECT_EQ(first.entries, 1U);
    EXPECT_EQ(first.misses, 1U);
    EXPECT_EQ(first.hits, 0U);
    (void)bf16_matmul_output(left, right, DType::BFloat16);
    runtime::synchronize(gpu);
    const auto second = bf16_plan_cache_stats();
    EXPECT_EQ(second.entries, 1U);
    EXPECT_EQ(second.misses, 1U);
    EXPECT_EQ(second.hits, 1U);
    clear_bf16_plan_cache();
    EXPECT_EQ(bf16_plan_cache_stats().entries, 0U);
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

TEST(HipFp8OpsTest, MixedE5ActivationAndE4WeightExecuteWithExplicitDispatch) {
    require_gpu();
    if (!hipblaslt_available()) GTEST_SKIP() << "hipBLASLt is unavailable";
    constexpr std::int64_t size = 128;
    std::vector<float> activation_values(static_cast<std::size_t>(size * size));
    std::vector<float> weight_values(activation_values.size());
    for (std::size_t index = 0; index < activation_values.size(); ++index) {
        activation_values[index] =
            static_cast<float>(static_cast<int>(index % 29) - 14) * 50.0F;
        weight_values[index] =
            static_cast<float>(static_cast<int>(index % 17) - 8) / 17.0F;
    }
    const auto activation_cpu = Tensor::from_vector(
        activation_values, {size, size});
    const auto weight_cpu = Tensor::from_vector(weight_values, {size, size});
    const auto gpu = Device::hip(0);
    const auto activation = quantize_fp8_dynamic(
        activation_cpu.to(gpu), DType::Float8E5M2FNUZ, 1.0e-4F);
    const auto weight = quantize_fp8_dynamic(
        weight_cpu.to(gpu), DType::Float8E4M3FNUZ, 1.0e-4F);
    clear_fp8_dispatch_registry();
    runtime::reset_transfer_stats();
    const auto output = fp8_matmul(activation, weight, DType::Float32);
    runtime::synchronize(gpu);
    const auto transfers = runtime::transfer_stats();
    const auto dispatch = fp8_dispatch_stats();
    RecordProperty("mixed_e5e4_native_shapes", dispatch.native_shapes);
    RecordProperty("mixed_e5e4_fallback_calls",
                   dispatch.software_fallback_calls);
    EXPECT_EQ(transfers.host_to_device_calls, 0U);
    EXPECT_EQ(transfers.device_to_host_calls, 0U);
    EXPECT_EQ(dispatch.native_shapes + dispatch.software_fallback_shapes, 1U);
    EXPECT_TRUE(dispatch.native_shapes == 1U ||
                dispatch.software_fallback_calls == 1U);
    const auto actual = output.to_vector();
    const auto reference = matmul(
        dequantize_fp8(activation, DType::Float32).to(Device::cpu()),
        dequantize_fp8(weight, DType::Float32).to(Device::cpu())).to_vector();
    float maximum = 0.0F;
    for (std::size_t index = 0; index < actual.size(); ++index) {
        maximum = std::max(maximum, std::abs(actual[index] - reference[index]));
    }
    EXPECT_LT(maximum, 400.0F);
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

TEST(HipFp8OpsTest, DynamicTensorScaleQuantizeAndDequantizeStayOnDevice) {
    require_gpu();
    const auto gpu = Device::hip(0);
    const auto cpu = Tensor::from_vector(
        {-3.0F, -0.5F, 0.0F, 2.0F, 9.0F, -12.0F}, {2, 3});
    const auto input = cpu.to(gpu);
    clear_fp8_dynamic_quant_stats();
    runtime::reset_transfer_stats();
    const auto dynamic = quantize_fp8_dynamic(
        input, DType::Float8E4M3FNUZ, 0.001F);
    EXPECT_FALSE(dynamic.host_scale_available);
    const auto restored = dequantize_fp8(dynamic, DType::Float32);
    runtime::synchronize(gpu);
    const auto transfers = runtime::transfer_stats();
    EXPECT_EQ(transfers.host_to_device_calls, 0U);
    EXPECT_EQ(transfers.device_to_host_calls, 0U);
    EXPECT_EQ(fp8_dynamic_quant_stats().tensor_calls, 1U);
    EXPECT_EQ(fp8_dynamic_quant_stats().tensor_elements, 6U);
    const auto scale = dynamic.scale.to_vector()[0];
    EXPECT_NEAR(scale, 12.0F / 240.0F, 1.0e-7F);
    expect_near(restored.to_vector(), cpu.to_vector(), 0.5F);

    clear_fp8_dynamic_quant_stats();
    const auto clipped_input = Tensor::from_vector(
        {1.0F, 100.0F}, {1, 2}).to(gpu);
    const auto clipped = quantize_fp8_dynamic(
        clipped_input, DType::Float8E4M3FNUZ, 1.0e-4F, {}, 0.5F);
    const auto clipped_values = dequantize_fp8(
        clipped, DType::Float32).to_vector();
    EXPECT_NEAR(clipped_values[1], 50.0F, 0.5F);
    EXPECT_EQ(fp8_dynamic_quant_stats().clipped_tensor_calls, 1U);
}

TEST(HipFp8OpsTest, MultiBlockDynamicScaleFindsMaximumInLastPartition) {
    require_gpu();
    constexpr std::int64_t elements = 262144;
    std::vector<float> values(static_cast<std::size_t>(elements), 0.25F);
    values.back() = -123.0F;
    const auto input = Tensor::from_vector(values, {elements}).to(Device::hip(0));
    runtime::reset_transfer_stats();
    const auto dynamic = quantize_fp8_dynamic(
        input, DType::Float8E4M3FNUZ, 1.0e-4F);
    runtime::synchronize(Device::hip(0));
    const auto transfers = runtime::transfer_stats();
    EXPECT_EQ(transfers.host_to_device_calls, 0U);
    EXPECT_EQ(transfers.device_to_host_calls, 0U);
    EXPECT_NEAR(dynamic.scale.to_vector()[0], 123.0F / 240.0F, 1.0e-6F);
    const auto restored = dequantize_fp8(dynamic, DType::Float32).to_vector();
    EXPECT_NEAR(restored.back(), -123.0F, 4.0F);
}

TEST(HipFp8OpsTest, OuterRowScaleGemmMatchesIndependentRangeReference) {
    require_gpu();
    if (!hipblaslt_available()) GTEST_SKIP() << "hipBLASLt is unavailable";
    constexpr std::int64_t size = 128;
    std::vector<float> left_values(static_cast<std::size_t>(size * size));
    std::vector<float> right_values(left_values.size());
    for (std::int64_t row = 0; row < size; ++row) {
        const auto factor = row == 0 ? 100.0F : 0.1F + static_cast<float>(row) / 128.0F;
        for (std::int64_t column = 0; column < size; ++column) {
            left_values[static_cast<std::size_t>(row * size + column)] =
                factor * static_cast<float>((column % 7) - 3) / 3.0F;
            right_values[static_cast<std::size_t>(row * size + column)] =
                static_cast<float>((row + column) % 5 - 2) / 16.0F;
        }
    }
    const auto left_cpu = Tensor::from_vector(left_values, {size, size});
    const auto right_cpu = Tensor::from_vector(right_values, {size, size});
    const auto gpu = Device::hip(0);
    const auto left = left_cpu.to(gpu);
    const auto right = right_cpu.to(gpu);
    clear_fp8_dispatch_registry();
    runtime::reset_transfer_stats();
    const auto left_fp8 = quantize_fp8_rows_dynamic(
        left, DType::Float8E4M3FNUZ, 1.0e-4F);
    const auto right_fp8 = quantize_fp8(
        right, DType::Float8E4M3FNUZ, 1.0F / 240.0F);
    const auto output = fp8_matmul(left_fp8, right_fp8, DType::Float32);
    runtime::synchronize(gpu);
    const auto transfers = runtime::transfer_stats();
    EXPECT_EQ(transfers.host_to_device_calls, 1U);  // fixed right scalar only
    EXPECT_EQ(transfers.device_to_host_calls, 0U);
    EXPECT_EQ(fp8_dispatch_stats().software_fallback_calls, 1U);
    EXPECT_EQ(fp8_dispatch_stats().outer_row_fallback_calls, 1U);
    EXPECT_EQ(fp8_dispatch_stats().outer_row_native_status, 0);
    const auto actual = output.to_vector();
    const auto expected = matmul(left_cpu, right_cpu).to_vector();
    float maximum = 0.0F;
    for (std::size_t index = 0; index < actual.size(); ++index) {
        maximum = std::max(maximum, std::abs(actual[index] - expected[index]));
    }
    EXPECT_LT(maximum, 8.0F);
}

TEST(HipFp8OpsTest, OutputColumnScaleUsesNativeGemmAndDevicePostScale) {
    require_gpu();
    if (!hipblaslt_available()) GTEST_SKIP() << "hipBLASLt is unavailable";
    constexpr std::int64_t size = 128;
    std::vector<float> left_values(static_cast<std::size_t>(size * size));
    std::vector<float> weight_values(left_values.size());
    for (std::int64_t row = 0; row < size; ++row) {
        for (std::int64_t column = 0; column < size; ++column) {
            const auto index = static_cast<std::size_t>(row * size + column);
            left_values[index] =
                static_cast<float>((row + column) % 17 - 8) / 17.0F;
            const auto amplitude = column == 0
                                       ? 0.01F
                                       : static_cast<float>(column % 7 + 1) / 7.0F;
            weight_values[index] =
                static_cast<float>((row * 3 + column) % 19 - 9) /
                19.0F * amplitude;
        }
    }
    const auto left_cpu = Tensor::from_vector(left_values, {size, size});
    const auto weight_cpu = Tensor::from_vector(weight_values, {size, size});
    const auto gpu = Device::hip(0);
    const auto left = quantize_fp8_dynamic(
        left_cpu.to(gpu), DType::Float8E4M3FNUZ, 1.0e-4F);
    clear_fp8_dynamic_quant_stats();
    const auto weight = quantize_fp8_columns_dynamic(
        weight_cpu.to(gpu), DType::Float8E4M3FNUZ, 1.0e-4F);
    EXPECT_EQ(fp8_dynamic_quant_stats().column_calls, 1U);
    EXPECT_EQ(fp8_dynamic_quant_stats().column_elements,
              static_cast<std::uint64_t>(size * size));
    EXPECT_EQ(weight.scale_mode, Fp8ScaleMode::OuterColumn);

    clear_fp8_dispatch_registry();
    runtime::reset_transfer_stats();
    const auto output = fp8_matmul(left, weight, DType::Float32);
    runtime::synchronize(gpu);
    const auto transfers = runtime::transfer_stats();
    const auto dispatch = fp8_dispatch_stats();
    RecordProperty("output_column_native_status",
                   dispatch.output_column_native_status);
    RecordProperty("output_column_scale_calls",
                   dispatch.output_column_scale_calls);
    EXPECT_EQ(transfers.host_to_device_calls, 0U);
    EXPECT_EQ(transfers.device_to_host_calls, 0U);
    EXPECT_EQ(dispatch.native_shapes, 1U);
    EXPECT_EQ(dispatch.software_fallback_calls, 0U);
    ASSERT_TRUE(dispatch.output_column_native_status == 0 ||
                dispatch.output_column_native_status == 1);
    EXPECT_EQ(dispatch.output_column_scale_calls,
              dispatch.output_column_native_status == 1 ? 0U : 1U);

    const auto actual = output.to_vector();
    const auto expected = matmul(left_cpu, weight_cpu).to_vector();
    float maximum = 0.0F;
    for (std::size_t index = 0; index < actual.size(); ++index) {
        maximum = std::max(maximum, std::abs(actual[index] - expected[index]));
    }
    EXPECT_LT(maximum, 1.0F);
}

TEST(HipFp8OpsTest, PreparedTransformerWeightsMatchLazyInferenceWithoutPayloadTransfers) {
    require_gpu();
    if (!hipblaslt_available()) GTEST_SKIP() << "hipBLASLt is unavailable";
    model::ModelConfig config{.vocabulary_size = 16,
                              .dimension = 8,
                              .layers = 1,
                              .heads = 2,
                              .kv_heads = 1,
                              .ffn_dimension = 16,
                              .max_sequence_length = 8,
                              .rope_base = 10000.0F,
                              .tie_embeddings = false,
                              .linear_precision = model::LinearPrecision::Float8E4M3FNUZ,
                              .fp8_activation_scale = 0.025F,
                              .fp8_weight_scale = 0.005F};
    model::TransformerModel model(config, 223);
    model.to(Device::hip(0));
    const auto tokens = Tensor::from_int32_vector({1, 2, 3, 4}, {1, 4})
                            .to(Device::hip(0));
    const auto before = model.forward_inference(tokens).to_vector();
    const auto report = model.prepare_fp8_inference_weights();
    runtime::reset_transfer_stats();
    clear_fp8_dynamic_quant_stats();
    const auto after_tensor = model.forward_inference(tokens);
    runtime::synchronize(Device::hip(0));
    const auto transfers = runtime::transfer_stats();
    EXPECT_EQ(report.converted_tensors, 8U);
    EXPECT_EQ(report.fp32_bytes_released, report.fp8_bytes_retained * 4U);
    EXPECT_EQ(transfers.host_to_device_calls, 0U);
    EXPECT_EQ(transfers.device_to_host_calls, 0U);
    EXPECT_EQ(after_tensor.to_vector(), before);
}

TEST(HipFp8OpsTest, TensorAmaxPreparedWeightsScanOnceAndLeaveHotPathTransferFree) {
    require_gpu();
    if (!hipblaslt_available()) GTEST_SKIP() << "hipBLASLt is unavailable";
    model::ModelConfig config{.vocabulary_size = 16,
                              .dimension = 8,
                              .layers = 1,
                              .heads = 2,
                              .kv_heads = 1,
                              .ffn_dimension = 16,
                              .max_sequence_length = 8,
                              .rope_base = 10000.0F,
                              .tie_embeddings = false,
                              .linear_precision = model::LinearPrecision::Float8E4M3FNUZ,
                              .fp8_activation_scale = 0.2F,
                              .fp8_activation_minimum_scale = 1.0e-4F,
                              .fp8_weight_scale = 0.005F,
                              .fp8_weight_scale_mode =
                                  model::Fp8WeightScaleMode::TensorAmax,
                              .fp8_activation_scale_mode =
                                  model::Fp8ActivationScaleMode::TensorAmax};
    model::TransformerModel model(config, 227);
    model.to(Device::hip(0));
    const auto tokens = Tensor::from_int32_vector({1, 2, 3, 4}, {1, 4})
                            .to(Device::hip(0));
    const auto before = model.forward_inference(tokens).to_vector();
    runtime::reset_transfer_stats();
    const auto report = model.prepare_fp8_inference_weights();
    const auto preparation_transfers = runtime::transfer_stats();
    EXPECT_EQ(report.converted_tensors, 8U);
    EXPECT_EQ(report.weight_bytes_scanned, report.fp32_bytes_released);
    EXPECT_EQ(report.scale_bytes_retained,
              report.converted_tensors * sizeof(float));
    EXPECT_GE(preparation_transfers.device_to_host_calls,
              report.converted_tensors);
    EXPECT_GT(report.maximum_weight_scale, report.minimum_weight_scale);
    clear_fp8_dynamic_quant_stats();
    runtime::reset_transfer_stats();
    const auto after_tensor = model.forward_inference(tokens);
    runtime::synchronize(Device::hip(0));
    const auto hot_path_transfers = runtime::transfer_stats();
    EXPECT_EQ(hot_path_transfers.host_to_device_calls, 0U);
    EXPECT_EQ(hot_path_transfers.device_to_host_calls, 0U);
    EXPECT_EQ(fp8_dynamic_quant_stats().tensor_calls, 5U);
    EXPECT_EQ(fp8_dynamic_quant_stats().row_calls, 0U);
    EXPECT_EQ(after_tensor.to_vector(), before);
}

TEST(HipFp8OpsTest, FfnOuterRowRoutesOnlyThreeLinearsWithoutPayloadTransfers) {
    require_gpu();
    if (!hipblaslt_available()) GTEST_SKIP() << "hipBLASLt is unavailable";
    model::ModelConfig config{.vocabulary_size = 16,
                              .dimension = 8,
                              .layers = 1,
                              .heads = 2,
                              .kv_heads = 1,
                              .ffn_dimension = 16,
                              .max_sequence_length = 8,
                              .rope_base = 10000.0F,
                              .tie_embeddings = false,
                              .linear_precision = model::LinearPrecision::Float8E4M3FNUZ,
                              .fp8_activation_scale = 0.2F,
                              .fp8_activation_minimum_scale = 1.0e-4F,
                              .fp8_weight_scale = 0.005F,
                              .fp8_activation_scale_mode =
                                  model::Fp8ActivationScaleMode::FfnOuterRow};
    model::TransformerModel model(config, 229);
    model.to(Device::hip(0));
    const auto tokens = Tensor::from_int32_vector({1, 2, 3, 4}, {1, 4})
                            .to(Device::hip(0));
    const auto report = model.prepare_fp8_inference_weights();
    EXPECT_EQ(report.scale_bytes_retained, 13U * sizeof(float));
    clear_fp8_dispatch_registry();
    clear_fp8_dynamic_quant_stats();
    runtime::reset_transfer_stats();
    const auto output = model.forward_inference(tokens);
    runtime::synchronize(Device::hip(0));
    const auto transfers = runtime::transfer_stats();
    const auto stats = fp8_dispatch_stats();
    EXPECT_EQ(transfers.host_to_device_calls, 0U);
    EXPECT_EQ(transfers.device_to_host_calls, 0U);
    EXPECT_EQ(stats.outer_row_fallback_calls, 3U);
    EXPECT_EQ(stats.outer_row_native_status, 0);
    EXPECT_EQ(fp8_dynamic_quant_stats().tensor_calls, 0U);
    EXPECT_EQ(fp8_dynamic_quant_stats().row_calls, 2U);
    for (const auto value : output.to_vector()) EXPECT_TRUE(std::isfinite(value));
}

TEST(HipFp8OpsTest, DeviceWeightAmaxPreparationAvoidsWeightPayloadD2H) {
    require_gpu();
    if (!hipblaslt_available()) GTEST_SKIP() << "hipBLASLt is unavailable";
    model::ModelConfig config{.vocabulary_size = 16,
                              .dimension = 8,
                              .layers = 1,
                              .heads = 2,
                              .kv_heads = 1,
                              .ffn_dimension = 16,
                              .max_sequence_length = 8,
                              .rope_base = 10000.0F,
                              .tie_embeddings = false,
                              .linear_precision = model::LinearPrecision::Float8E4M3FNUZ,
                              .fp8_activation_scale = 0.2F,
                              .fp8_weight_scale = 1.0e-4F,
                              .fp8_weight_scale_mode =
                                  model::Fp8WeightScaleMode::DeviceTensorAmax};
    model::TransformerModel model(config, 233);
    model.to(Device::hip(0));
    const auto tokens = Tensor::from_int32_vector({1, 2, 3, 4}, {1, 4})
                            .to(Device::hip(0));
    const auto before = model.forward_inference(tokens).to_vector();
    runtime::reset_transfer_stats();
    const auto report = model.prepare_fp8_inference_weights();
    runtime::synchronize(Device::hip(0));
    const auto preparation_transfers = runtime::transfer_stats();
    EXPECT_EQ(preparation_transfers.device_to_host_calls, 0U);
    EXPECT_EQ(report.device_amax_tensors, 8U);
    EXPECT_EQ(report.device_weight_bytes_scanned, report.fp32_bytes_released);
    EXPECT_EQ(report.weight_bytes_scanned, 0U);
    EXPECT_FALSE(report.host_scale_summary_available);
    EXPECT_FLOAT_EQ(report.minimum_weight_scale, 0.0F);
    EXPECT_FLOAT_EQ(report.maximum_weight_scale, 0.0F);
    runtime::reset_transfer_stats();
    const auto after = model.forward_inference(tokens);
    runtime::synchronize(Device::hip(0));
    const auto hot = runtime::transfer_stats();
    EXPECT_EQ(hot.host_to_device_calls, 0U);
    EXPECT_EQ(hot.device_to_host_calls, 0U);
    EXPECT_EQ(after.to_vector(), before);
}

TEST(HipFp8OpsTest, OutputChannelWeightPreparationIsDeviceOnlyAndReusable) {
    require_gpu();
    model::ModelConfig config{.vocabulary_size = 16,
                              .dimension = 8,
                              .layers = 1,
                              .heads = 2,
                              .kv_heads = 1,
                              .ffn_dimension = 16,
                              .max_sequence_length = 8,
                              .rope_base = 10000.0F,
                              .tie_embeddings = false,
                              .linear_precision =
                                  model::LinearPrecision::Float8E4M3FNUZ,
                              .fp8_activation_scale = 0.2F,
                              .fp8_activation_minimum_scale = 1.0e-4F,
                              .fp8_weight_scale = 1.0e-4F,
                              .fp8_weight_scale_mode =
                                  model::Fp8WeightScaleMode::OutputChannelAmax,
                              .fp8_activation_scale_mode =
                                  model::Fp8ActivationScaleMode::TensorAmax};
    model::TransformerModel transformer(config, 237);
    const auto gpu = Device::hip(0);
    transformer.to(gpu);
    clear_fp8_dynamic_quant_stats();
    runtime::reset_transfer_stats();
    const auto report = transformer.prepare_fp8_inference_weights();
    runtime::synchronize(gpu);
    const auto preparation_transfers = runtime::transfer_stats();
    EXPECT_EQ(report.linears_covered, 8U);
    EXPECT_EQ(report.converted_tensors, 8U);
    EXPECT_EQ(report.scale_bytes_retained, 80U * sizeof(float));
    EXPECT_EQ(report.device_amax_tensors, 8U);
    EXPECT_EQ(report.device_weight_bytes_scanned, report.fp32_bytes_released);
    EXPECT_EQ(fp8_dynamic_quant_stats().column_calls, 8U);
    EXPECT_EQ(preparation_transfers.device_to_host_calls, 0U);

    const auto tokens = Tensor::from_int32_vector({1, 2, 3, 4}, {1, 4})
                            .to(gpu);
    clear_fp8_dynamic_quant_stats();
    runtime::reset_transfer_stats();
    const auto output = transformer.forward_inference(tokens);
    runtime::synchronize(gpu);
    const auto hot = runtime::transfer_stats();
    EXPECT_EQ(hot.host_to_device_calls, 0U);
    EXPECT_EQ(hot.device_to_host_calls, 0U);
    EXPECT_EQ(fp8_dynamic_quant_stats().column_calls, 0U);
    EXPECT_EQ(fp8_dynamic_quant_stats().tensor_calls, 5U);
    for (const auto value : output.to_vector()) EXPECT_TRUE(std::isfinite(value));
}

TEST(HipFp8OpsTest, AttentionOutputScopePreparesOneColumnVectorOnDevice) {
    require_gpu();
    model::ModelConfig config{.vocabulary_size = 16,
                              .dimension = 8,
                              .layers = 1,
                              .heads = 2,
                              .kv_heads = 1,
                              .ffn_dimension = 16,
                              .max_sequence_length = 8,
                              .rope_base = 10000.0F,
                              .tie_embeddings = false,
                              .linear_precision =
                                  model::LinearPrecision::Float8E4M3FNUZ,
                              .fp8_activation_scale = 0.2F,
                              .fp8_activation_minimum_scale = 1.0e-4F,
                              .fp8_weight_scale = 1.0e-4F,
                              .fp8_weight_scale_mode =
                                  model::Fp8WeightScaleMode::OutputChannelAmax,
                              .fp8_weight_scale_scope =
                                  model::Fp8WeightScaleScope::AttentionOutputOnly,
                              .fp8_activation_scale_mode =
                                  model::Fp8ActivationScaleMode::TensorAmax};
    model::TransformerModel transformer(config, 249);
    const auto gpu = Device::hip(0);
    transformer.to(gpu);
    clear_fp8_dynamic_quant_stats();
    runtime::reset_transfer_stats();
    const auto report = transformer.prepare_fp8_inference_weights();
    runtime::synchronize(gpu);
    const auto preparation = runtime::transfer_stats();
    EXPECT_EQ(report.converted_tensors, 8U);
    EXPECT_EQ(report.scale_bytes_retained, 15U * sizeof(float));
    EXPECT_EQ(report.device_amax_tensors, 8U);
    EXPECT_EQ(fp8_dynamic_quant_stats().column_calls, 1U);
    EXPECT_EQ(fp8_dynamic_quant_stats().tensor_calls, 7U);
    EXPECT_EQ(preparation.device_to_host_calls, 0U);

    const auto tokens = Tensor::from_int32_vector({1, 2, 3, 4}, {1, 4})
                            .to(gpu);
    clear_fp8_dynamic_quant_stats();
    runtime::reset_transfer_stats();
    const auto output = transformer.forward_inference(tokens);
    runtime::synchronize(gpu);
    const auto hot = runtime::transfer_stats();
    EXPECT_EQ(hot.host_to_device_calls, 0U);
    EXPECT_EQ(hot.device_to_host_calls, 0U);
    EXPECT_EQ(fp8_dynamic_quant_stats().column_calls, 0U);
    EXPECT_EQ(fp8_dynamic_quant_stats().tensor_calls, 5U);
    for (const auto value : output.to_vector()) EXPECT_TRUE(std::isfinite(value));
}

TEST(HipFp8OpsTest, SelectedFp32BlockRunsInsidePreparedHipModel) {
    require_gpu();
    if (!hipblaslt_available()) GTEST_SKIP() << "hipBLASLt is unavailable";
    model::ModelConfig config{.vocabulary_size = 16,
                              .dimension = 8,
                              .layers = 2,
                              .heads = 2,
                              .kv_heads = 1,
                              .ffn_dimension = 16,
                              .max_sequence_length = 8,
                              .rope_base = 10000.0F,
                              .tie_embeddings = false,
                              .linear_precision = model::LinearPrecision::Float8E4M3FNUZ,
                              .fp8_activation_scale = 0.2F,
                              .fp8_activation_minimum_scale = 1.0e-4F,
                              .fp8_weight_scale = 0.005F,
                              .fp8_weight_scale_mode =
                                  model::Fp8WeightScaleMode::TensorAmax,
                              .fp8_activation_scale_mode =
                                  model::Fp8ActivationScaleMode::TensorAmax,
                              .fp8_fp32_layers = {1}};
    model::TransformerModel model(config, 239);
    model.to(Device::hip(0));
    const auto report = model.prepare_fp8_inference_weights();
    EXPECT_EQ(report.converted_tensors, 8U);
    const auto tokens = Tensor::from_int32_vector({1, 2, 3, 4}, {1, 4})
                            .to(Device::hip(0));
    clear_fp8_dynamic_quant_stats();
    runtime::reset_transfer_stats();
    const auto output = model.forward_inference(tokens);
    runtime::synchronize(Device::hip(0));
    const auto transfers = runtime::transfer_stats();
    EXPECT_EQ(transfers.host_to_device_calls, 0U);
    EXPECT_EQ(transfers.device_to_host_calls, 0U);
    EXPECT_EQ(fp8_dynamic_quant_stats().tensor_calls, 5U);
    for (const auto value : output.to_vector()) EXPECT_TRUE(std::isfinite(value));
}

TEST(HipFp8OpsTest, DiagnosticModesStayOnDeviceAndUseOnlySelectedRounding) {
    require_gpu();
    const auto gpu = Device::hip(0);
    for (const auto mode : {model::Fp8DiagnosticMode::WeightOnly,
                            model::Fp8DiagnosticMode::ActivationOnly,
                            model::Fp8DiagnosticMode::BothRoundtrip}) {
        const auto rounds_weight =
            mode != model::Fp8DiagnosticMode::ActivationOnly;
        const auto rounds_activation =
            mode != model::Fp8DiagnosticMode::WeightOnly;
        model::ModelConfig config{.vocabulary_size = 16,
                                  .dimension = 8,
                                  .layers = 1,
                                  .heads = 2,
                                  .kv_heads = 1,
                                  .ffn_dimension = 16,
                                  .max_sequence_length = 8,
                                  .rope_base = 10000.0F,
                                  .tie_embeddings = false,
                                  .linear_precision =
                                      model::LinearPrecision::Float8E4M3FNUZ,
                                  .fp8_activation_scale = 0.2F,
                                  .fp8_activation_minimum_scale = 1.0e-4F,
                                  .fp8_weight_scale = 1.0e-4F,
                                  .fp8_weight_scale_mode =
                                      model::Fp8WeightScaleMode::DeviceTensorAmax,
                                  .fp8_activation_scale_mode =
                                      model::Fp8ActivationScaleMode::TensorAmax,
                                  .fp8_diagnostic_mode = mode};
        model::TransformerModel transformer(config, 241);
        transformer.to(gpu);
        const auto report = transformer.prepare_fp8_inference_weights();
        EXPECT_EQ(report.linears_covered, 8U);
        EXPECT_EQ(report.converted_tensors, rounds_weight ? 8U : 0U);
        const auto tokens = Tensor::from_int32_vector({1, 2, 3, 4}, {1, 4})
                                .to(gpu);
        clear_fp8_dispatch_registry();
        clear_fp8_dynamic_quant_stats();
        runtime::reset_transfer_stats();
        const auto output = transformer.forward_inference(tokens);
        runtime::synchronize(gpu);
        const auto transfers = runtime::transfer_stats();
        EXPECT_EQ(transfers.host_to_device_calls, 0U);
        EXPECT_EQ(transfers.device_to_host_calls, 0U);
        EXPECT_EQ(fp8_dynamic_quant_stats().tensor_calls,
                  rounds_activation ? 5U : 0U);
        EXPECT_EQ(fp8_dispatch_stats().native_shapes, 0U);
        EXPECT_EQ(fp8_dispatch_stats().software_fallback_calls, 0U);
        for (const auto value : output.to_vector()) EXPECT_TRUE(std::isfinite(value));
    }
}

TEST(HipFp8OpsTest, DeepSeekLinearShapesAcceptFp32OutputOrBf16Fallback) {
    require_gpu();
    if (!hipblaslt_available()) GTEST_SKIP() << "hipBLASLt is unavailable";
    const auto gpu = Device::hip(0);
    const auto scale = 0.005F;
    clear_fp8_dispatch_registry();
    for (const auto& [inner, columns] : {
             std::pair<std::int64_t, std::int64_t>{896, 896},
             {896, 128}, {896, 4864}, {4864, 896},
             {1536, 1536}, {1536, 256}, {1536, 8960}, {8960, 1536}}) {
        SCOPED_TRACE("inner=" + std::to_string(inner) +
                     " columns=" + std::to_string(columns));
        std::vector<float> left_values(static_cast<std::size_t>(8 * inner));
        std::vector<float> right_values(
            static_cast<std::size_t>(inner * columns));
        for (std::size_t index = 0; index < left_values.size(); ++index) {
            left_values[index] = static_cast<float>(
                static_cast<int>(index % 11) - 5) * 0.01F;
        }
        for (std::size_t index = 0; index < right_values.size(); ++index) {
            right_values[index] = static_cast<float>(
                static_cast<int>(index % 7) - 3) * 0.005F;
        }
        const auto left = Tensor::from_vector(
                              left_values, {8, inner}).to(gpu);
        const auto right = Tensor::from_vector(
                               right_values, {inner, columns}).to(gpu);
        const auto output = fp8_matmul(
            quantize_fp8(left, DType::Float8E4M3FNUZ, 0.025F),
            quantize_fp8(right, DType::Float8E4M3FNUZ, scale),
            DType::Float32);
        ASSERT_EQ(output.shape(), (Shape{8, columns}));
        for (const auto value : output.to_vector()) EXPECT_TRUE(std::isfinite(value));
    }
    const auto stats = fp8_dispatch_stats();
    EXPECT_GE(stats.native_shapes, 1U);
    EXPECT_GE(stats.software_fallback_shapes, 1U);
    EXPECT_GE(stats.software_fallback_calls, 1U);
}

TEST(HipBf16TrainingTest, TransformerPolicyKeepsFp32MastersAndGradients) {
    require_gpu();
    if (!hipblaslt_available()) GTEST_SKIP() << "hipBLASLt is unavailable";
    const model::ModelConfig config{.vocabulary_size = 16,
                                    .dimension = 128,
                                    .layers = 1,
                                    .heads = 4,
                                    .kv_heads = 2,
                                    .ffn_dimension = 256,
                                    .max_sequence_length = 4,
                                    .linear_precision = model::LinearPrecision::BFloat16};
    model::TransformerModel transformer(config, 59);
    transformer.to(Device::hip(0));
    const auto mirrors = transformer.prepare_bf16_training_mirrors();
    EXPECT_EQ(mirrors.size(), 8U);
    const auto tokens = Tensor::from_int32_vector({1, 2, 3, 4}, {1, 4}).to(Device::hip(0));
    const auto targets = Tensor::from_int32_vector({2, 3, 4, 5}, {1, 4}).to(Device::hip(0));
    runtime::reset_transfer_stats();
    const auto loss = transformer.loss(tokens, targets);
    loss.backward();
    runtime::synchronize(Device::hip(0));
    const auto transfers = runtime::transfer_stats();
    EXPECT_EQ(transfers.host_to_device_calls, 0U);
    EXPECT_EQ(transfers.device_to_host_calls, 0U);
    EXPECT_TRUE(std::isfinite(loss.data().to_vector()[0]));
    for (const auto& [name, parameter] : transformer.named_parameters()) {
        ASSERT_TRUE(parameter->has_grad()) << name;
        EXPECT_EQ(parameter->data().dtype(), DType::Float32) << name;
        EXPECT_EQ(parameter->grad().dtype(), DType::Float32) << name;
    }
}

TEST(HipBf16TrainingTest, AdamwUpdatesMasterAndMirrorInOneDeviceLaunch) {
    require_gpu();
    const auto gpu = Device::hip(0);
    autograd::Value parameter(
        Tensor::from_vector({1.0F, -2.0F, 3.0F}, {3}).to(gpu), true);
    auto mirror = ops::cast(parameter.data(), DType::BFloat16);
    training::AdamW optimizer(
        {&parameter}, {.learning_rate = 0.01F, .weight_decay = 0.0F},
        {{&parameter, &mirror}});
    autograd::sum(autograd::scale(parameter, 2.0F)).backward();
    runtime::reset_transfer_stats();
    optimizer.step();
    runtime::synchronize(gpu);
    const auto transfers = runtime::transfer_stats();
    EXPECT_EQ(transfers.host_to_device_calls, 0U);
    EXPECT_EQ(transfers.device_to_host_calls, 0U);
    EXPECT_EQ(mirror.cast(DType::Float32).to_vector(),
              parameter.data().cast(DType::BFloat16).cast(DType::Float32).to_vector());
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

TEST(HipCausalSoftmaxTest, RegisterBoundaryT2048MatchesCpuAndZerosMask) {
    require_gpu();
    constexpr std::int64_t sequence = 2048;
    std::vector<float> values(static_cast<std::size_t>(sequence * sequence));
    for (std::size_t index = 0; index < values.size(); ++index) {
        values[index] = static_cast<float>(static_cast<int>(index % 101U) - 50) *
                        0.01F;
    }
    const auto scores = Tensor::from_vector(values, {1, 1, sequence, sequence});
    const auto expected = causal_softmax(scores).to_vector();
    const auto actual = causal_softmax(scores.to(Device::hip(0))).to_vector();
    ASSERT_EQ(actual.size(), expected.size());
    float maximum_error = 0.0F;
    for (std::size_t index = 0; index < actual.size(); ++index) {
        maximum_error = std::max(maximum_error,
                                 std::abs(actual[index] - expected[index]));
    }
    EXPECT_LT(maximum_error, 2.0e-6F);
    for (const auto row : {0LL, 1LL, 1023LL, 2047LL}) {
        float total = 0.0F;
        for (std::int64_t column = 0; column < sequence; ++column) {
            const auto value = actual[static_cast<std::size_t>(row * sequence + column)];
            if (column <= row) total += value;
            else EXPECT_EQ(value, 0.0F) << "row=" << row << " column=" << column;
        }
        EXPECT_NEAR(total, 1.0F, 2.0e-5F) << "row=" << row;
    }
}

TEST(HipRmsNormTest, BlockParallelForwardBackwardCoverModelWidthsWithoutHostTransfers) {
    require_gpu();
    const auto gpu = Device::hip(0);
    for (const auto width : {16LL, 384LL, 512LL, 896LL, 1536LL}) {
        for (const auto rows : {1LL, 3LL, 32LL, 256LL}) {
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

TEST(HipCachedAttentionTest, Bf16BatchFusedAndLongFallbackMatchRoundedCpu) {
    require_gpu();
    const auto gpu = Device::hip(0);
    constexpr std::int64_t batches = 2;
    constexpr std::int64_t heads = 2;
    constexpr std::int64_t kv_heads = 1;
    constexpr std::int64_t width = 8;
    constexpr std::int64_t repeats = heads / kv_heads;
    std::vector<float> query_values(
        static_cast<std::size_t>(batches * heads * width));
    for (std::size_t index = 0; index < query_values.size(); ++index) {
        query_values[index] =
            static_cast<float>(static_cast<int>(index % 13) - 6) * 0.03125F;
    }
    const auto query = Tensor::from_vector(
        query_values, {batches, heads, 1, width});
    for (const auto sequence : {32LL, 4097LL}) {
        std::vector<float> key_values(
            static_cast<std::size_t>(batches * kv_heads * sequence * width));
        std::vector<float> value_values(key_values.size());
        for (std::size_t index = 0; index < key_values.size(); ++index) {
            const auto batch = static_cast<std::int64_t>(index) /
                               (kv_heads * sequence * width);
            key_values[index] =
                static_cast<float>(static_cast<int>(index % 23) - 11) * 0.01953125F +
                static_cast<float>(batch) * 0.25F;
            value_values[index] =
                static_cast<float>(static_cast<int>(index % 19) - 9) * 0.0234375F +
                static_cast<float>(batch) * 0.5F;
        }
        const auto key = Tensor::from_vector(
            key_values, {batches, kv_heads, sequence, width}, DType::BFloat16);
        const auto value = Tensor::from_vector(
            value_values, {batches, kv_heads, sequence, width}, DType::BFloat16);
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
        expect_near(actual.to_vector(), expected, 8.0e-4F);
    }
}

TEST(HipFullAttentionTest, CausalMhaGqaForwardBackwardMatchCpuWithoutTransfers) {
    require_gpu();
    const auto gpu = Device::hip(0);
    constexpr std::int64_t heads = 4;
    constexpr std::int64_t width = 16;
    for (const auto gqa : {false, true}) {
        const auto kv_heads = gqa ? 2LL : heads;
        const auto repeats = heads / kv_heads;
        for (const auto sequence : {1LL, 3LL, 32LL, 128LL, 256LL}) {
            std::vector<float> query_values(
                static_cast<std::size_t>(heads * sequence * width));
            std::vector<float> key_values(
                static_cast<std::size_t>(kv_heads * sequence * width));
            std::vector<float> value_values(key_values.size());
            std::vector<float> gradient_values(query_values.size());
            for (std::size_t index = 0; index < query_values.size(); ++index) {
                query_values[index] =
                    static_cast<float>(static_cast<int>(index % 29) - 14) * 0.015625F;
                gradient_values[index] =
                    static_cast<float>(static_cast<int>(index % 31) - 15) * 0.0078125F;
            }
            for (std::size_t index = 0; index < key_values.size(); ++index) {
                key_values[index] =
                    static_cast<float>(static_cast<int>(index % 23) - 11) * 0.01953125F;
                value_values[index] =
                    static_cast<float>(static_cast<int>(index % 19) - 9) * 0.0234375F;
            }
            const auto query = Tensor::from_vector(
                query_values, {1, heads, sequence, width});
            const auto key = Tensor::from_vector(
                key_values, {1, kv_heads, sequence, width});
            const auto value = Tensor::from_vector(
                value_values, {1, kv_heads, sequence, width});
            const auto gradient = Tensor::from_vector(
                gradient_values, {1, heads, sequence, width});
            const auto expected = causal_gqa_attention(
                query, key, value, repeats, 0.25F);
            const auto expected_backward = causal_gqa_attention_backward(
                query, key, value, gradient, repeats, 0.25F);
            const auto device_query = query.to(gpu);
            const auto device_key = key.to(gpu);
            const auto device_value = value.to(gpu);
            const auto device_gradient = gradient.to(gpu);
            runtime::reset_transfer_stats();
            Tensor actual;
            TensorTriple actual_backward;
            Tensor plain_long_forward;
            if (sequence >= 256) {
                plain_long_forward = causal_gqa_attention(
                    device_query, device_key, device_value, repeats, 0.25F);
                auto saved = causal_gqa_attention_saved(
                    device_query, device_key, device_value, repeats, 0.25F);
                actual = std::move(saved.first);
                actual_backward = causal_gqa_attention_backward_saved(
                    device_query, device_key, device_value, saved.second,
                    device_gradient, repeats, 0.25F);
            } else {
                actual = causal_gqa_attention(
                    device_query, device_key, device_value, repeats, 0.25F);
                actual_backward = causal_gqa_attention_backward(
                    device_query, device_key, device_value, device_gradient,
                    repeats, 0.25F);
            }
            runtime::synchronize(gpu);
            const auto transfers = runtime::transfer_stats();
            EXPECT_EQ(transfers.host_to_device_calls, 0U);
            EXPECT_EQ(transfers.device_to_host_calls, 0U);
            expect_near(actual.to_vector(), expected.to_vector(), 8.0e-4F);
            if (plain_long_forward.defined()) {
                expect_near(plain_long_forward.to_vector(), expected.to_vector(), 8.0e-4F);
            }
            expect_near(actual_backward.first.to_vector(),
                        expected_backward.first.to_vector(), 1.5e-3F);
            expect_near(actual_backward.second.to_vector(),
                        expected_backward.second.to_vector(), 2.0e-3F);
            expect_near(actual_backward.third.to_vector(),
                        expected_backward.third.to_vector(), 2.0e-3F);
            if (sequence == 1 || sequence == 256) {
                CausalGqaAttentionWorkspace workspace{
                    .scaled_query = Tensor(
                        device_query.shape(), DType::Float32, gpu),
                    .expanded_kv = Tensor(
                        device_query.shape(), DType::Float32, gpu),
                    .probabilities = Tensor(
                        {1, heads, sequence, sequence},
                        DType::Float32, gpu)};
                Tensor caller_output(
                    device_query.shape(), DType::Float32, gpu);
                runtime::reset_transfer_stats();
                causal_gqa_attention_out_(
                    caller_output, workspace, device_query, device_key,
                    device_value, repeats, 0.25F);
                runtime::synchronize(gpu);
                const auto caller_transfers = runtime::transfer_stats();
                EXPECT_EQ(caller_transfers.host_to_device_calls, 0U);
                EXPECT_EQ(caller_transfers.device_to_host_calls, 0U);
                expect_near(caller_output.to_vector(),
                            expected.to_vector(), 8.0e-4F);
            }
        }
    }
}

TEST(HipFullAttentionTest, LongBatchFusedForwardIsDeterministic) {
    require_gpu();
    constexpr std::int64_t batches = 8;
    constexpr std::int64_t heads = 4;
    constexpr std::int64_t kv_heads = 2;
    constexpr std::int64_t sequence = 128;
    constexpr std::int64_t width = 4;
    std::vector<float> query_values(
        static_cast<std::size_t>(batches * heads * sequence * width));
    std::vector<float> key_values(
        static_cast<std::size_t>(batches * kv_heads * sequence * width));
    std::vector<float> value_values(key_values.size());
    for (std::size_t index = 0; index < query_values.size(); ++index) {
        query_values[index] = static_cast<float>(static_cast<int>(index % 29) - 14) / 17.0F;
    }
    for (std::size_t index = 0; index < key_values.size(); ++index) {
        key_values[index] = static_cast<float>(static_cast<int>(index % 23) - 11) / 13.0F;
        value_values[index] = static_cast<float>(static_cast<int>(index % 19) - 9) / 11.0F;
    }
    const auto gpu = Device::hip(0);
    const auto query_cpu = Tensor::from_vector(
        query_values, {batches, heads, sequence, width});
    const auto key_cpu = Tensor::from_vector(
        key_values, {batches, kv_heads, sequence, width});
    const auto value_cpu = Tensor::from_vector(
        value_values, {batches, kv_heads, sequence, width});
    const auto cpu_reference = causal_gqa_attention(
        query_cpu, key_cpu, value_cpu, heads / kv_heads, 0.5F).to_vector();
    const auto query = query_cpu.to(gpu);
    const auto key = key_cpu.to(gpu);
    const auto value = value_cpu.to(gpu);
    const auto expected = causal_gqa_attention(
        query, key, value, heads / kv_heads, 0.5F).to_vector();
    expect_near(expected, cpu_reference, 3.0e-4F);
    for (int repetition = 0; repetition < 20; ++repetition) {
        const auto actual = causal_gqa_attention(
            query, key, value, heads / kv_heads, 0.5F).to_vector();
        float maximum = 0.0F;
        for (std::size_t index = 0; index < actual.size(); ++index) {
            maximum = std::max(maximum, std::abs(actual[index] - expected[index]));
        }
        EXPECT_EQ(maximum, 0.0F) << "repetition=" << repetition;
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

TEST(HipArgmaxTest, LastDimensionReducesEveryRowOnDevice) {
    require_gpu();
    const auto gpu = Device::hip(0);
    const auto input = Tensor::from_vector(
        {1.0F, 7.0F, 7.0F, 2.0F,
         -3.0F, -2.0F, -1.0F, -4.0F,
         9.0F, 8.0F, 7.0F, 6.0F}, {3, 1, 4}).to(gpu);
    runtime::reset_transfer_stats();
    const auto selected = argmax_last_dim(input);
    runtime::synchronize(gpu);
    auto transfers = runtime::transfer_stats();
    EXPECT_EQ(transfers.device_to_host_calls, 0U);
    EXPECT_EQ(selected.shape(), (Shape{3, 1}));
    EXPECT_EQ(selected.to_int32_vector(),
              (std::vector<std::int32_t>{1, 2, 0}));
    transfers = runtime::transfer_stats();
    EXPECT_EQ(transfers.device_to_host_calls, 1U);
    EXPECT_EQ(transfers.device_to_host_bytes, 3U * sizeof(std::int32_t));
}

TEST(HipArgmaxTest, OutVariantsFillDeviceHistoryWithoutIntermediateD2H) {
    require_gpu();
    const auto gpu = Device::hip(0);
    const auto first_input = Tensor::from_vector(
        {1.0F, 7.0F, 2.0F, 9.0F, 3.0F, 4.0F}, {2, 1, 3}).to(gpu);
    const auto second_input = Tensor::from_vector(
        {8.0F, 7.0F, 6.0F, 1.0F, 2.0F, 5.0F}, {2, 1, 3}).to(gpu);
    Tensor history({2, 2}, DType::Int32, gpu);
    auto first = history.slice(0, 0, 1).reshape({2, 1});
    auto second = history.slice(0, 1, 2).reshape({2, 1});
    runtime::reset_transfer_stats();
    argmax_last_dim_out_(first_input, first);
    argmax_last_dim_out_(second_input, second);
    runtime::synchronize(gpu);
    EXPECT_EQ(runtime::transfer_stats().device_to_host_calls, 0U);
    EXPECT_EQ(history.to_int32_vector(),
              (std::vector<std::int32_t>{1, 0, 0, 2}));
    const auto transfers = runtime::transfer_stats();
    EXPECT_EQ(transfers.device_to_host_calls, 1U);
    EXPECT_EQ(transfers.device_to_host_bytes, 4U * sizeof(std::int32_t));
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
        model, prompt, {.max_new_tokens = 4, .temperature = 0.0F, .top_k = 1,
                        .seed = 9, .kv_cache_layer_dtypes = {},
                        .stop_tokens = {}});
    const auto transfers = runtime::transfer_stats();
    EXPECT_EQ(generated.size(), 6U);
    EXPECT_EQ(transfers.host_to_device_calls, 1U);
    EXPECT_EQ(transfers.host_to_device_bytes, prompt.size() * sizeof(std::int32_t));
    EXPECT_EQ(transfers.device_to_host_calls, 1U);
    EXPECT_EQ(transfers.device_to_host_bytes, 4U * sizeof(std::int32_t));
}

TEST(HipSchedulerTest, DelayedIndependentRequestsMatchCpuReference) {
    require_gpu();
    const model::ModelConfig config{.vocabulary_size = 16,
                                    .dimension = 8,
                                    .layers = 1,
                                    .heads = 2,
                                    .kv_heads = 1,
                                    .ffn_dimension = 16,
                                    .max_sequence_length = 12,
                                    .rope_base = 10000.0F,
                                    .tie_embeddings = false};
    model::TransformerModel cpu_model(config, 97);
    model::TransformerModel hip_model(config, 97);
    hip_model.to(Device::hip());
    inference::ReferenceScheduler cpu_scheduler(cpu_model);
    inference::ReferenceScheduler hip_scheduler(hip_model);
    const inference::GenerationConfig first_config{
        .max_new_tokens = 4, .temperature = 0.0F, .top_k = 1,
        .seed = 3, .kv_cache_layer_dtypes = {},
        .stop_tokens = {}};
    const inference::GenerationConfig second_config{
        .max_new_tokens = 2, .temperature = 0.0F, .top_k = 1,
        .seed = 5, .kv_cache_layer_dtypes = {},
        .stop_tokens = {}};
    const auto cpu_first = cpu_scheduler.submit({1, 2, 3}, first_config);
    const auto hip_first = hip_scheduler.submit({1, 2, 3}, first_config);
    cpu_scheduler.step();
    hip_scheduler.step();
    const auto cpu_second = cpu_scheduler.submit({4, 5}, second_config);
    const auto hip_second = hip_scheduler.submit({4, 5}, second_config);
    cpu_scheduler.run_until_idle();
    hip_scheduler.run_until_idle();
    EXPECT_EQ(hip_scheduler.request(hip_first).generated,
              cpu_scheduler.request(cpu_first).generated);
    EXPECT_EQ(hip_scheduler.request(hip_second).generated,
              cpu_scheduler.request(cpu_second).generated);
    EXPECT_EQ(hip_scheduler.metrics().scheduler_steps, 4);
    EXPECT_EQ(hip_scheduler.metrics().prefill_calls, 2);
    EXPECT_EQ(hip_scheduler.metrics().decode_calls, 4);
    EXPECT_GT(hip_scheduler.metrics().peak_cache_bytes, 0U);
}

TEST(HipSchedulerTest, CancellationReleasesCacheAndPreservesSurvivor) {
    require_gpu();
    const model::ModelConfig config{.vocabulary_size = 16,
                                    .dimension = 8,
                                    .layers = 1,
                                    .heads = 2,
                                    .kv_heads = 1,
                                    .ffn_dimension = 16,
                                    .max_sequence_length = 12,
                                    .rope_base = 10000.0F,
                                    .tie_embeddings = false};
    model::TransformerModel cpu_model(config, 101);
    model::TransformerModel hip_model(config, 101);
    hip_model.to(Device::hip());
    inference::ReferenceScheduler cpu(cpu_model);
    inference::ReferenceScheduler hip(hip_model);
    const inference::GenerationConfig generation{
        .max_new_tokens = 4, .temperature = 0.0F, .top_k = 1,
        .seed = 7, .kv_cache_layer_dtypes = {},
        .stop_tokens = {}};
    const auto hip_cancelled = hip.submit({1, 2, 3}, generation);
    const auto hip_survivor = hip.submit({4, 5, 6}, generation);
    const auto cpu_survivor = cpu.submit({4, 5, 6}, generation);
    hip.step();
    EXPECT_GT(hip.request(hip_cancelled).cache_bytes, 0U);
    EXPECT_TRUE(hip.cancel(hip_cancelled));
    EXPECT_EQ(hip.request(hip_cancelled).cache_bytes, 0U);
    EXPECT_EQ(hip.request(hip_cancelled).state,
              inference::RequestState::Cancelled);
    hip.run_until_idle();
    cpu.run_until_idle();
    EXPECT_EQ(hip.request(hip_survivor).generated,
              cpu.request(cpu_survivor).generated);
    EXPECT_EQ(hip.metrics().cancelled_requests, 1);
    EXPECT_EQ(hip.metrics().completed_requests, 1);
    EXPECT_EQ(hip.metrics().active_cache_bytes, 0U);
}

TEST(HipSchedulerTest, StopTokenCompletesAndReleasesCacheImmediately) {
    require_gpu();
    const model::ModelConfig config{.vocabulary_size = 16,
                                    .dimension = 8,
                                    .layers = 1,
                                    .heads = 2,
                                    .kv_heads = 1,
                                    .ffn_dimension = 16,
                                    .max_sequence_length = 12,
                                    .rope_base = 10000.0F,
                                    .tie_embeddings = false};
    const std::vector<std::int32_t> prompt{1, 2, 3};
    const inference::GenerationConfig baseline{
        .max_new_tokens = 4, .temperature = 0.0F, .top_k = 1,
        .seed = 11, .kv_cache_layer_dtypes = {},
        .stop_tokens = {}};
    model::TransformerModel oracle(config, 131);
    const auto generated = inference::generate(oracle, prompt, baseline);
    auto stopped = baseline;
    stopped.stop_tokens = {generated[prompt.size()]};
    model::TransformerModel hip_model(config, 131);
    hip_model.to(Device::hip(0));
    inference::ReferenceScheduler scheduler(hip_model);
    const auto id = scheduler.submit(prompt, stopped);
    scheduler.step();
    const auto snapshot = scheduler.request(id);
    EXPECT_EQ(snapshot.state, inference::RequestState::Completed);
    EXPECT_EQ(snapshot.completion_reason,
              inference::CompletionReason::StopToken);
    EXPECT_EQ(snapshot.generated.size(), 1U);
    EXPECT_EQ(snapshot.cache_bytes, 0U);
    EXPECT_EQ(scheduler.metrics().stop_completed_requests, 1);
}

TEST(HipSchedulerTest, ContinuousSlotsRefillAndMatchCpuWithOneSelectionCopyPerStep) {
    require_gpu();
    const model::ModelConfig config{.vocabulary_size = 16,
                                    .dimension = 8,
                                    .layers = 1,
                                    .heads = 2,
                                    .kv_heads = 1,
                                    .ffn_dimension = 16,
                                    .max_sequence_length = 16,
                                    .rope_base = 10000.0F,
                                    .tie_embeddings = false};
    for (const auto dtype : {DType::Float32, DType::BFloat16}) {
        model::TransformerModel cpu_model(config, 163);
        model::TransformerModel hip_model(config, 163);
        hip_model.to(Device::hip(0));
        const inference::ContinuousBatchConfig scheduler_config{
            .max_slots = 2, .kv_cache_dtype = dtype,
            .kv_cache_layer_dtypes = {}};
        inference::ContinuousBatchScheduler cpu(cpu_model, scheduler_config);
        inference::ContinuousBatchScheduler hip(hip_model, scheduler_config);
        const inference::GenerationConfig short_generation{
            .max_new_tokens = 2, .temperature = 0.0F, .top_k = 1,
            .seed = 3, .kv_cache_dtype = dtype,
            .kv_cache_layer_dtypes = {}, .stop_tokens = {}};
        const inference::GenerationConfig long_generation{
            .max_new_tokens = 4, .temperature = 0.0F, .top_k = 1,
            .seed = 5, .kv_cache_dtype = dtype,
            .kv_cache_layer_dtypes = {}, .stop_tokens = {}};
        const auto cpu_first = cpu.submit({1, 2, 3}, short_generation);
        const auto cpu_second = cpu.submit({4, 5, 6}, long_generation);
        const auto hip_first = hip.submit({1, 2, 3}, short_generation);
        const auto hip_second = hip.submit({4, 5, 6}, long_generation);
        runtime::reset_transfer_stats();
        cpu.step();
        hip.step();
        const auto cpu_late = cpu.submit({7, 8}, short_generation);
        const auto hip_late = hip.submit({7, 8}, short_generation);
        cpu.step();
        hip.step();
        EXPECT_EQ(hip.request(hip_late).state,
                  inference::RequestState::PendingPrefill);
        cpu.step();
        hip.step();
        EXPECT_EQ(hip.request(hip_late).slot, 0);
        cpu.run_until_idle();
        hip.run_until_idle();
        for (const auto& [cpu_id, hip_id] : {
                 std::pair{cpu_first, hip_first},
                 std::pair{cpu_second, hip_second},
                 std::pair{cpu_late, hip_late}}) {
            EXPECT_EQ(hip.request(hip_id).generated,
                      cpu.request(cpu_id).generated);
            EXPECT_EQ(hip.request(hip_id).completion_reason,
                      cpu.request(cpu_id).completion_reason);
        }
        const auto cpu_metrics = cpu.metrics();
        const auto hip_metrics = hip.metrics();
        EXPECT_EQ(hip_metrics.scheduler_steps, 4);
        EXPECT_EQ(hip_metrics.slot_admissions, 3);
        EXPECT_EQ(hip_metrics.slot_refills, 1);
        EXPECT_EQ(hip_metrics.prefill_batch_calls, 2);
        EXPECT_EQ(hip_metrics.batched_prefill_calls, 1);
        EXPECT_EQ(hip_metrics.batched_prefill_rows, 2);
        EXPECT_EQ(hip_metrics.batch_decode_calls, 3);
        EXPECT_EQ(hip_metrics.divergent_batch_decode_calls, 1);
        EXPECT_EQ(hip_metrics.compacted_batch_decode_calls, 2);
        EXPECT_EQ(hip_metrics.positions_aware_batch_decode_calls, 2);
        EXPECT_EQ(hip_metrics.dummy_decode_rows, 0);
        EXPECT_EQ(hip_metrics.inactive_rows_skipped, 1);
        EXPECT_EQ(hip_metrics.logical_decode_rows,
                  cpu_metrics.logical_decode_rows);
        EXPECT_EQ(hip_metrics.allocated_cache_bytes,
                  cpu_metrics.allocated_cache_bytes);
        EXPECT_EQ(hip_metrics.active_cache_bytes, 0U);
        EXPECT_DOUBLE_EQ(hip_metrics.slot_utilization, 1.0);
        const auto transfers = runtime::transfer_stats();
        EXPECT_EQ(transfers.host_to_device_calls, 5U);
        EXPECT_EQ(transfers.host_to_device_bytes, 76U);
        EXPECT_EQ(transfers.device_to_host_calls,
                  static_cast<std::size_t>(hip_metrics.selection_calls));
        EXPECT_EQ(transfers.device_to_host_bytes,
                  static_cast<std::size_t>(hip_metrics.selection_calls * 2) *
                      sizeof(std::int32_t));

        model::TransformerModel recycled_cpu_model(config, 167);
        model::TransformerModel recycled_hip_model(config, 167);
        recycled_hip_model.to(Device::hip(0));
        const inference::ContinuousBatchConfig recycled_config{
            .max_slots = 1, .max_sequence_length = 6,
            .kv_cache_dtype = dtype, .kv_cache_layer_dtypes = {},
            .capture_selection_diagnostics = true};
        inference::ContinuousBatchScheduler recycled_cpu(
            recycled_cpu_model, recycled_config);
        inference::ContinuousBatchScheduler recycled_hip(
            recycled_hip_model, recycled_config);
        std::vector<inference::RequestId> cpu_ids;
        std::vector<inference::RequestId> hip_ids;
        for (const auto& prompt : {
                 std::vector<std::int32_t>{1, 2, 3},
                 std::vector<std::int32_t>{4, 5, 6, 7}}) {
            cpu_ids.push_back(recycled_cpu.submit(prompt, short_generation));
            hip_ids.push_back(recycled_hip.submit(prompt, short_generation));
        }
        recycled_cpu.run_until_idle();
        recycled_hip.run_until_idle();
        for (std::size_t index = 0; index < cpu_ids.size(); ++index) {
            EXPECT_EQ(recycled_hip.request(hip_ids[index]).generated,
                      recycled_cpu.request(cpu_ids[index]).generated);
        }
        EXPECT_EQ(recycled_hip.metrics().slot_refills, 1);
        ASSERT_EQ(recycled_hip.selection_diagnostics().size(),
                  recycled_cpu.selection_diagnostics().size());
        for (std::size_t index = 0;
             index < recycled_hip.selection_diagnostics().size(); ++index) {
            const auto& actual = recycled_hip.selection_diagnostics()[index];
            const auto& expected = recycled_cpu.selection_diagnostics()[index];
            EXPECT_TRUE(actual.device_argmax_matches_top1);
            EXPECT_EQ(actual.device_selected_token, expected.device_selected_token);
            EXPECT_EQ(actual.request_id, expected.request_id);
            EXPECT_EQ(actual.generated_index, expected.generated_index);
            EXPECT_EQ(actual.logit_source, expected.logit_source);
        }
    }
}

TEST(HipSchedulerTest, LengthBucketsMatchCpuAndShareOneModel) {
    require_gpu();
    const model::ModelConfig config{.vocabulary_size = 16,
                                    .dimension = 8,
                                    .layers = 1,
                                    .heads = 2,
                                    .kv_heads = 1,
                                    .ffn_dimension = 16,
                                    .max_sequence_length = 16,
                                    .rope_base = 10000.0F,
                                    .tie_embeddings = false};
    model::TransformerModel cpu_model(config, 173);
    model::TransformerModel hip_model(config, 173);
    hip_model.to(Device::hip(0));
    const inference::LengthBucketedBatchConfig scheduler_config{
        .buckets = {{.max_sequence_length = 6, .max_slots = 2},
                    {.max_sequence_length = 16, .max_slots = 1}},
        .kv_cache_dtype = DType::BFloat16,
        .kv_cache_layer_dtypes = {},
        .batch_equal_length_prefill = true};
    inference::LengthBucketedBatchScheduler cpu(cpu_model, scheduler_config);
    inference::LengthBucketedBatchScheduler hip(hip_model, scheduler_config);
    const inference::GenerationConfig short_generation{
        .max_new_tokens = 2, .temperature = 0.0F, .top_k = 1,
        .seed = 3, .kv_cache_dtype = DType::BFloat16,
        .kv_cache_layer_dtypes = {}, .stop_tokens = {}};
    const inference::GenerationConfig long_generation{
        .max_new_tokens = 4, .temperature = 0.0F, .top_k = 1,
        .seed = 5, .kv_cache_dtype = DType::BFloat16,
        .kv_cache_layer_dtypes = {}, .stop_tokens = {}};
    const std::vector<std::vector<std::int32_t>> prompts{
        {1, 2, 3}, {4, 5, 6}, {7, 8, 9, 10, 11, 12, 13, 14}};
    const std::vector<inference::GenerationConfig> generations{
        short_generation, short_generation, long_generation};
    std::vector<inference::RequestId> cpu_ids;
    std::vector<inference::RequestId> hip_ids;
    for (std::size_t index = 0; index < prompts.size(); ++index) {
        cpu_ids.push_back(cpu.submit(prompts[index], generations[index]));
        hip_ids.push_back(hip.submit(prompts[index], generations[index]));
    }
    runtime::reset_transfer_stats();
    cpu.run_until_idle();
    hip.run_until_idle();
    for (std::size_t index = 0; index < prompts.size(); ++index) {
        EXPECT_EQ(hip.request(hip_ids[index]).generated,
                  cpu.request(cpu_ids[index]).generated);
        EXPECT_EQ(hip.request(hip_ids[index]).completion_reason,
                  cpu.request(cpu_ids[index]).completion_reason);
    }
    const auto cpu_metrics = cpu.metrics();
    const auto hip_metrics = hip.metrics();
    EXPECT_EQ(hip_metrics.total_slots, 3);
    EXPECT_EQ(hip_metrics.buckets.size(), 2U);
    EXPECT_EQ(hip_metrics.allocated_cache_bytes,
              cpu_metrics.allocated_cache_bytes);
    EXPECT_LT(hip_metrics.allocated_cache_bytes,
              static_cast<std::size_t>(
                  2 * config.layers * config.kv_heads *
                  config.head_dimension() * 16 * 3 *
                  dtype_size(DType::BFloat16)));
    EXPECT_EQ(hip_metrics.active_cache_bytes, 0U);
    EXPECT_GT(hip_metrics.peak_active_cache_bytes, 0U);
    const auto transfers = runtime::transfer_stats();
    EXPECT_EQ(transfers.device_to_host_calls,
              static_cast<std::size_t>(
                  hip_metrics.buckets[0].selection_calls +
                  hip_metrics.buckets[1].selection_calls));
}

TEST(HipGenerationTest, StaticBatchDifferentRowsMatchCpuReference) {
    require_gpu();
    const model::ModelConfig config{.vocabulary_size = 16,
                                    .dimension = 8,
                                    .layers = 2,
                                    .heads = 2,
                                    .kv_heads = 1,
                                    .ffn_dimension = 16,
                                    .max_sequence_length = 12,
                                    .rope_base = 10000.0F,
                                    .tie_embeddings = false};
    model::TransformerModel cpu_model(config, 103);
    model::TransformerModel hip_model(config, 103);
    hip_model.to(Device::hip());
    const std::vector<std::vector<std::int32_t>> prompts{
        {1, 2, 3}, {4, 5, 6}, {7, 8, 9}};
    const inference::GenerationConfig generation{
        .max_new_tokens = 4, .temperature = 0.0F, .top_k = 1,
        .seed = 7, .kv_cache_dtype = DType::BFloat16,
        .kv_cache_layer_dtypes = {},
        .stop_tokens = {}};
    const auto expected = inference::generate_batch(cpu_model, prompts, generation);
    runtime::reset_transfer_stats();
    const auto actual = inference::generate_batch(hip_model, prompts, generation);
    EXPECT_EQ(actual, expected);
    EXPECT_NE(actual[0], actual[1]);
    const auto transfers = runtime::transfer_stats();
    EXPECT_EQ(transfers.device_to_host_calls, 1U);
    EXPECT_EQ(transfers.device_to_host_bytes,
              prompts.size() * static_cast<std::size_t>(generation.max_new_tokens) *
                  sizeof(std::int32_t));
}

TEST(HipGenerationTest, StopTokenRowsMatchCpuAtDifferentLengths) {
    require_gpu();
    const model::ModelConfig config{.vocabulary_size = 32,
                                    .dimension = 16,
                                    .layers = 2,
                                    .heads = 4,
                                    .kv_heads = 2,
                                    .ffn_dimension = 32,
                                    .max_sequence_length = 12,
                                    .rope_base = 10000.0F,
                                    .tie_embeddings = false};
    const std::vector<std::vector<std::int32_t>> prompts{
        {1, 2, 3}, {4, 5, 6}, {7, 8, 9}, {10, 11, 12}};
    const inference::GenerationConfig baseline{
        .max_new_tokens = 6, .temperature = 0.0F, .top_k = 1,
        .seed = 23, .kv_cache_layer_dtypes = {},
        .stop_tokens = {}};
    std::vector<std::vector<std::int32_t>> independent_rows;
    for (const auto& prompt : prompts) {
        model::TransformerModel row_model(config, 139);
        independent_rows.push_back(inference::generate(row_model, prompt, baseline));
    }
    std::int32_t stop = -1;
    for (std::int32_t token = 0; token < config.vocabulary_size; ++token) {
        std::vector<std::size_t> positions;
        for (std::size_t row = 0; row < prompts.size(); ++row) {
            const auto begin = independent_rows[row].begin() +
                               static_cast<std::ptrdiff_t>(prompts[row].size());
            const auto found = std::find(begin, independent_rows[row].end(), token);
            positions.push_back(
                found == independent_rows[row].end()
                    ? static_cast<std::size_t>(baseline.max_new_tokens)
                    : static_cast<std::size_t>(std::distance(begin, found)));
        }
        if (*std::min_element(positions.begin(), positions.end()) <
                static_cast<std::size_t>(baseline.max_new_tokens) &&
            *std::min_element(positions.begin(), positions.end()) !=
                *std::max_element(positions.begin(), positions.end())) {
            stop = token;
            break;
        }
    }
    ASSERT_GE(stop, 0);
    auto generation = baseline;
    generation.stop_tokens = {stop};
    model::TransformerModel cpu_model(config, 139);
    model::TransformerModel hip_model(config, 139);
    hip_model.to(Device::hip(0));
    const auto expected = inference::generate_batch(cpu_model, prompts, generation);
    const auto actual = inference::generate_batch(hip_model, prompts, generation);
    EXPECT_EQ(actual, expected);
    std::vector<std::size_t> lengths;
    for (const auto& row : actual) lengths.push_back(row.size());
    EXPECT_NE(*std::min_element(lengths.begin(), lengths.end()),
              *std::max_element(lengths.begin(), lengths.end()));
}

TEST(HipSchedulerTest, AdmissionBucketsMatchCpuRowsAndGroups) {
    require_gpu();
    const model::ModelConfig config{.vocabulary_size = 16,
                                    .dimension = 8,
                                    .layers = 2,
                                    .heads = 2,
                                    .kv_heads = 1,
                                    .ffn_dimension = 16,
                                    .max_sequence_length = 12,
                                    .rope_base = 10000.0F,
                                    .tie_embeddings = false};
    model::TransformerModel cpu_model(config, 113);
    model::TransformerModel hip_model(config, 113);
    hip_model.to(Device::hip());
    inference::AdmissionBatchScheduler cpu(cpu_model);
    inference::AdmissionBatchScheduler hip(hip_model);
    const inference::GenerationConfig generation{
        .max_new_tokens = 3, .temperature = 0.0F, .top_k = 1,
        .seed = 13, .kv_cache_layer_dtypes = {},
        .stop_tokens = {}};
    std::vector<inference::RequestId> cpu_ids;
    std::vector<inference::RequestId> hip_ids;
    for (const auto& prompt : std::vector<std::vector<std::int32_t>>{
             {1, 2, 3}, {4, 5, 6}, {7, 8, 9}, {10, 11}}) {
        cpu_ids.push_back(cpu.submit(prompt, generation));
        hip_ids.push_back(hip.submit(prompt, generation));
    }
    cpu.drain();
    hip.drain();
    for (std::size_t index = 0; index < cpu_ids.size(); ++index) {
        EXPECT_EQ(hip.request(hip_ids[index]).generated,
                  cpu.request(cpu_ids[index]).generated);
    }
    EXPECT_EQ(hip.metrics().batch_groups, 2);
    EXPECT_EQ(hip.metrics().singleton_groups, 1);
    EXPECT_EQ(hip.metrics().batched_requests, 3);
    EXPECT_EQ(hip.metrics().maximum_batch_size, 3);
}

TEST(HipSchedulerTest, AdmissionCancellationDoesNotEnterHipBatch) {
    require_gpu();
    const model::ModelConfig config{.vocabulary_size = 16,
                                    .dimension = 8,
                                    .layers = 2,
                                    .heads = 2,
                                    .kv_heads = 1,
                                    .ffn_dimension = 16,
                                    .max_sequence_length = 12,
                                    .rope_base = 10000.0F,
                                    .tie_embeddings = false};
    model::TransformerModel cpu_model(config, 127);
    model::TransformerModel hip_model(config, 127);
    hip_model.to(Device::hip());
    inference::AdmissionBatchScheduler cpu(cpu_model);
    inference::AdmissionBatchScheduler hip(hip_model);
    const inference::GenerationConfig generation{
        .max_new_tokens = 3, .temperature = 0.0F, .top_k = 1,
        .seed = 17, .kv_cache_layer_dtypes = {},
        .stop_tokens = {}};
    std::vector<inference::RequestId> cpu_ids;
    std::vector<inference::RequestId> hip_ids;
    for (const auto& prompt : std::vector<std::vector<std::int32_t>>{
             {1, 2, 3}, {4, 5, 6}, {7, 8, 9}}) {
        cpu_ids.push_back(cpu.submit(prompt, generation));
        hip_ids.push_back(hip.submit(prompt, generation));
    }
    EXPECT_TRUE(cpu.cancel(cpu_ids[1]));
    EXPECT_TRUE(hip.cancel(hip_ids[1]));
    cpu.drain();
    hip.drain();
    EXPECT_EQ(hip.request(hip_ids[1]).state,
              inference::RequestState::Cancelled);
    EXPECT_TRUE(hip.request(hip_ids[1]).generated.empty());
    for (const auto index : {0U, 2U}) {
        EXPECT_EQ(hip.request(hip_ids[index]).generated,
                  cpu.request(cpu_ids[index]).generated);
    }
    EXPECT_EQ(hip.metrics().cancelled_requests, 1);
    EXPECT_EQ(hip.metrics().batched_requests, 2);
    EXPECT_EQ(hip.metrics().maximum_batch_size, 2);
}

TEST(HipAllocatorStressTest, ReusedDefaultStreamBlocksPreserveAsyncKernelOrder) {
    require_gpu();
    const auto gpu = Device::hip(0);
    runtime::enable_hip_caching_allocator(gpu);
    runtime::reset_allocation_peak(gpu);
    for (int iteration = 0; iteration < 256; ++iteration) {
        Tensor temporary({4096}, DType::Float32, gpu);
        fill_(temporary, static_cast<float>(iteration));
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
    const auto fused_input = Tensor::from_vector(
        {1, 2, 3, 4, 5, 6, 7, 8,
         -1, -2, -3, -4, -5, -6, -7, -8}, {1, 2, 2, 4});
    const auto fused_bias = Tensor::from_vector(
        {0.1F, 0.2F, 0.3F, 0.4F, -0.1F, -0.2F, -0.3F, -0.4F}, {8});
    expect_near(rope_split_half_bias(fused_input.to(Device::hip()),
                                     fused_bias.to(Device::hip())).to_vector(),
                rope_split_half_bias(fused_input, fused_bias).to_vector());

    const auto bthd_input = fused_input.transpose(1, 2).contiguous();
    const auto bthd_gradient = Tensor::from_vector(
        {1, -1, 2, -2, 3, -3, 4, -4,
         -1, 1, -2, 2, -3, 3, -4, 4}, {1, 2, 2, 4});
    const auto expected_bthd = rope_split_half_bias_bthd(
        bthd_input, fused_bias, 7, 5000.0F);
    const auto expected_bthd_backward = rope_split_half_bias_bthd_backward(
        bthd_gradient, 7, 5000.0F);
    const auto device_bthd = bthd_input.to(Device::hip());
    const auto device_bias = fused_bias.to(Device::hip());
    const auto device_gradient = bthd_gradient.to(Device::hip());
    runtime::reset_transfer_stats();
    const auto actual_bthd = rope_split_half_bias_bthd(
        device_bthd, device_bias, 7, 5000.0F);
    const auto actual_bthd_backward = rope_split_half_bias_bthd_backward(
        device_gradient, 7, 5000.0F);
    runtime::synchronize(Device::hip());
    const auto transfers = runtime::transfer_stats();
    EXPECT_EQ(transfers.host_to_device_calls, 0U);
    EXPECT_EQ(transfers.device_to_host_calls, 0U);
    expect_near(actual_bthd.to_vector(), expected_bthd.to_vector(), 2.0e-5F);
    expect_near(actual_bthd_backward.to_vector(),
                expected_bthd_backward.to_vector(), 2.0e-5F);

    const auto logits_cpu = Tensor::from_vector({2, 1, 0, 0, 1, 2}, {2, 3});
    const auto targets_cpu = Tensor::from_int32_vector({0, 2}, {2});
    expect_near(cross_entropy(logits_cpu.to(Device::hip()), targets_cpu.to(Device::hip())).to_vector(),
                cross_entropy(logits_cpu, targets_cpu).to_vector());
}

TEST(HipOpsTest, AttentionProbabilityValueInterleavedLayoutMatchesCpuWithoutTransfers) {
    require_gpu();
    const auto gpu = Device::hip(0);
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
    const auto expected = attention_probability_value_bthd(probabilities, value);
    const auto device_probabilities = probabilities.to(gpu);
    const auto device_value = value.to(gpu);
    const auto output_gradient = Tensor::from_vector(
        {1, -1, 0.5F, -0.5F, 2, -2, 1.5F, -1.5F, 3, -3, 2.5F, -2.5F,
         -1, 1, -0.5F, 0.5F, -2, 2, -1.5F, 1.5F, -3, 3, -2.5F, 2.5F},
        {2, 3, 2, 2});
    const auto expected_probability_gradient =
        attention_probability_gradient_bthd(output_gradient, value);
    const auto expected_value_gradient =
        attention_value_gradient_bthd(probabilities, output_gradient);
    const auto device_output_gradient = output_gradient.to(gpu);
    runtime::reset_transfer_stats();
    const auto actual = attention_probability_value_bthd(
        device_probabilities, device_value);
    const auto actual_probability_gradient =
        attention_probability_gradient_bthd(
            device_output_gradient, device_value);
    const auto actual_value_gradient = attention_value_gradient_bthd(
        device_probabilities, device_output_gradient);
    runtime::synchronize(gpu);
    const auto transfers = runtime::transfer_stats();
    EXPECT_EQ(transfers.host_to_device_calls, 0U);
    EXPECT_EQ(transfers.device_to_host_calls, 0U);
    EXPECT_EQ(actual.shape(), value.shape());
    expect_near(actual.to_vector(), expected.to_vector(), 2.0e-5F);
    expect_near(actual_probability_gradient.to_vector(),
                expected_probability_gradient.to_vector(), 2.0e-5F);
    expect_near(actual_value_gradient.to_vector(),
                expected_value_gradient.to_vector(), 2.0e-5F);
}

TEST(HipOpsTest, GqaProbabilityValueZeroStrideBroadcastMatchesCpuForBatchTwo) {
    require_gpu();
    const auto gpu = Device::hip(0);
    const auto probabilities = Tensor::from_vector(
        {1, 0, 0.25F, 0.75F, 1, 0, 0.5F, 0.5F,
         1, 0, 0.75F, 0.25F, 1, 0, 0.1F, 0.9F,
         1, 0, 0.2F, 0.8F, 1, 0, 0.4F, 0.6F,
         1, 0, 0.6F, 0.4F, 1, 0, 0.8F, 0.2F},
        {2, 4, 2, 2});
    const auto value = Tensor::from_vector(
        {1, 2, 10, 20, 3, 4, 30, 40,
         -1, -2, -10, -20, -3, -4, -30, -40},
        {2, 2, 2, 2});
    const auto expected = attention_probability_value_gqa_bthd(
        probabilities, value, 2);
    std::vector<float> gradient_values(2 * 2 * 4 * 2);
    for (std::size_t index = 0; index < gradient_values.size(); ++index) {
        gradient_values[index] =
            static_cast<float>(static_cast<int>(index % 17) - 8) / 19.0F;
    }
    const auto output_gradient = Tensor::from_vector(
        gradient_values, {2, 2, 4, 2});
    const auto expected_gradient = attention_probability_gradient_gqa_bthd(
        output_gradient, value, 2);
    const auto device_probabilities = probabilities.to(gpu);
    const auto device_value = value.to(gpu);
    const auto device_gradient = output_gradient.to(gpu);
    runtime::reset_transfer_stats();
    const auto actual = attention_probability_value_gqa_bthd(
        device_probabilities, device_value, 2);
    const auto actual_gradient = attention_probability_gradient_gqa_bthd(
        device_gradient, device_value, 2);
    runtime::synchronize(gpu);
    const auto transfers = runtime::transfer_stats();
    EXPECT_EQ(transfers.host_to_device_calls, 0U);
    EXPECT_EQ(transfers.device_to_host_calls, 0U);
    expect_near(actual.to_vector(), expected.to_vector(), 2.0e-5F);
    expect_near(actual_gradient.to_vector(),
                expected_gradient.to_vector(), 2.0e-5F);
}

TEST(HipAttentionLayoutPlanCacheTest, ExactModesAndShapesHitWithoutChangingOutputs) {
    require_gpu();
    const auto gpu = Device::hip(0);
    const auto probabilities = Tensor::from_vector(
        {1, 0, 0, 1, 1, 0, 0, 1}, {1, 2, 2, 2}).to(gpu);
    const auto value = Tensor::from_vector(
        {1, 2, 10, 20, 3, 4, 30, 40}, {1, 2, 2, 2}).to(gpu);
    const auto seed = Tensor::from_vector(
        {1, -1, 2, -2, 3, -3, 4, -4}, {1, 2, 2, 2}).to(gpu);
    clear_attention_layout_plan_cache();
    enable_attention_layout_plan_cache(true);
    const auto pv = attention_probability_value_bthd(probabilities, value);
    const auto dp = attention_probability_gradient_bthd(seed, value);
    const auto dv = attention_value_gradient_bthd(probabilities, seed);
    auto stats = attention_layout_plan_cache_stats();
    EXPECT_TRUE(attention_layout_plan_cache_enabled());
    EXPECT_EQ(stats.entries, 3U);
    EXPECT_EQ(stats.misses, 3U);
    EXPECT_EQ(stats.hits, 0U);
    expect_near(attention_probability_value_bthd(
                    probabilities, value).to_vector(),
                pv.to_vector(), 0.0F);
    expect_near(attention_probability_gradient_bthd(
                    seed, value).to_vector(),
                dp.to_vector(), 0.0F);
    expect_near(attention_value_gradient_bthd(
                    probabilities, seed).to_vector(),
                dv.to_vector(), 0.0F);
    stats = attention_layout_plan_cache_stats();
    EXPECT_EQ(stats.entries, 3U);
    EXPECT_EQ(stats.misses, 3U);
    EXPECT_EQ(stats.hits, 3U);
    const auto other_probabilities = Tensor::from_vector(
        {1, 1}, {1, 2, 1, 1}).to(gpu);
    const auto other_value = Tensor::from_vector(
        {1, 2, 3, 4}, {1, 1, 2, 2}).to(gpu);
    (void)attention_probability_value_bthd(
        other_probabilities, other_value);
    stats = attention_layout_plan_cache_stats();
    EXPECT_EQ(stats.entries, 4U);
    EXPECT_EQ(stats.misses, 4U);
    EXPECT_EQ(stats.hits, 3U);

    enable_attention_layout_plan_cache(false);
    EXPECT_FALSE(attention_layout_plan_cache_enabled());
    const auto uncached_first = attention_probability_value_bthd(
        probabilities, value);
    const auto uncached_second = attention_probability_value_bthd(
        probabilities, value);
    stats = attention_layout_plan_cache_stats();
    EXPECT_EQ(stats.entries, 0U);
    EXPECT_EQ(stats.misses, 0U);
    EXPECT_EQ(stats.hits, 0U);
    expect_near(uncached_first.to_vector(), pv.to_vector(), 0.0F);
    expect_near(uncached_second.to_vector(), pv.to_vector(), 0.0F);
    enable_attention_layout_plan_cache(false);
}

TEST(HipOpsTest, PairedGqaRepeatForwardBackwardMatchSeparateWithoutTransfers) {
    require_gpu();
    const auto gpu = Device::hip(0);
    const auto key = Tensor::from_vector(
        {1, 2, 3, 4, 10, 20, 30, 40}, {1, 2, 2, 2});
    const auto value = Tensor::from_vector(
        {5, 6, 50, 60, 7, 8, 70, 80}, {1, 2, 2, 2});
    const auto key_gradient = Tensor::from_vector(
        {1, 2, 3, 4, 5, 6, 7, 8,
         9, 10, 11, 12, 13, 14, 15, 16}, {1, 4, 2, 2});
    const auto value_gradient = Tensor::from_vector(
        {16, 15, 14, 13, 12, 11, 10, 9,
         8, 7, 6, 5, 4, 3, 2, 1}, {1, 2, 4, 2});
    const auto expected = repeat_gqa_kv_bthd(key, value, 2);
    const auto expected_gradients = repeat_gqa_kv_bthd_backward(
        key_gradient, value_gradient, 2);
    const auto device_key = key.to(gpu);
    const auto device_value = value.to(gpu);
    const auto device_key_gradient = key_gradient.to(gpu);
    const auto device_value_gradient = value_gradient.to(gpu);
    runtime::reset_transfer_stats();
    const auto actual = repeat_gqa_kv_bthd(
        device_key, device_value, 2);
    const auto actual_gradients = repeat_gqa_kv_bthd_backward(
        device_key_gradient, device_value_gradient, 2);
    runtime::synchronize(gpu);
    const auto transfers = runtime::transfer_stats();
    EXPECT_EQ(transfers.host_to_device_calls, 0U);
    EXPECT_EQ(transfers.device_to_host_calls, 0U);
    expect_near(actual.first.to_vector(), expected.first.to_vector(), 0.0F);
    expect_near(actual.second.to_vector(), expected.second.to_vector(), 0.0F);
    expect_near(actual_gradients.first.to_vector(),
                expected_gradients.first.to_vector(), 0.0F);
    expect_near(actual_gradients.second.to_vector(),
                expected_gradients.second.to_vector(), 0.0F);
    enable_attention_paired_gqa_repeat(false);
    EXPECT_FALSE(attention_paired_gqa_repeat_enabled());
    enable_attention_paired_gqa_repeat(true);
    EXPECT_TRUE(attention_paired_gqa_repeat_enabled());
    enable_attention_paired_gqa_repeat(false);
}

TEST(HipOpsTest, LongCausalGqaBthdForwardBackwardMatchCpuWithoutTransfers) {
    require_gpu();
    constexpr std::int64_t batches = 1;
    constexpr std::int64_t heads = 2;
    constexpr std::int64_t kv_heads = 1;
    constexpr std::int64_t sequence = 256;
    constexpr std::int64_t width = 4;
    constexpr std::int64_t repeats = heads / kv_heads;
    std::vector<float> query_values(
        static_cast<std::size_t>(batches * heads * sequence * width));
    std::vector<float> key_values(
        static_cast<std::size_t>(batches * kv_heads * sequence * width));
    std::vector<float> value_values(
        static_cast<std::size_t>(batches * sequence * kv_heads * width));
    std::vector<float> seed_values(query_values.size());
    for (std::size_t index = 0; index < query_values.size(); ++index) {
        query_values[index] =
            static_cast<float>(static_cast<int>(index % 23) - 11) / 29.0F;
        seed_values[index] =
            static_cast<float>(static_cast<int>(index % 17) - 8) / 19.0F;
    }
    for (std::size_t index = 0; index < key_values.size(); ++index) {
        key_values[index] =
            static_cast<float>(static_cast<int>(index % 19) - 9) / 31.0F;
        value_values[index] =
            static_cast<float>(static_cast<int>(index % 13) - 6) / 17.0F;
    }
    const auto query = Tensor::from_vector(
        query_values, {batches, heads, sequence, width});
    const auto key = Tensor::from_vector(
        key_values, {batches, kv_heads, sequence, width});
    const auto value = Tensor::from_vector(
        value_values, {batches, sequence, kv_heads, width});
    const auto seed = Tensor::from_vector(
        seed_values, {batches, sequence, heads, width});
    const auto expected_saved = causal_gqa_attention_bthd_saved(
        query, key, value, repeats, 0.5F);
    const auto expected_gradients = causal_gqa_attention_bthd_backward_saved(
        query, key, value, expected_saved.second, seed, repeats, 0.5F);

    const auto gpu = Device::hip(0);
    const auto device_query = query.to(gpu);
    const auto device_key = key.to(gpu);
    const auto device_value = value.to(gpu);
    const auto device_seed = seed.to(gpu);
    runtime::reset_transfer_stats();
    const auto actual_saved = causal_gqa_attention_bthd_saved(
        device_query, device_key, device_value, repeats, 0.5F);
    const auto actual_gradients = causal_gqa_attention_bthd_backward_saved(
        device_query, device_key, device_value, actual_saved.second,
        device_seed, repeats, 0.5F);
    runtime::synchronize(gpu);
    const auto transfers = runtime::transfer_stats();
    EXPECT_EQ(transfers.host_to_device_calls, 0U);
    EXPECT_EQ(transfers.device_to_host_calls, 0U);
    expect_near(actual_saved.first.to_vector(),
                expected_saved.first.to_vector(), 2.0e-3F);
    expect_near(actual_saved.second.to_vector(),
                expected_saved.second.to_vector(), 2.0e-3F);
    expect_near(actual_gradients.first.to_vector(),
                expected_gradients.first.to_vector(), 3.0e-3F);
    expect_near(actual_gradients.second.to_vector(),
                expected_gradients.second.to_vector(), 3.0e-3F);
    expect_near(actual_gradients.third.to_vector(),
                expected_gradients.third.to_vector(), 3.0e-3F);
}

TEST(HipOpsTest, WideGqaValueBroadcastFullSavedPathMatchesExpandedControl) {
    require_gpu();
    constexpr std::int64_t sequence = 256;
    constexpr std::int64_t heads = 2;
    constexpr std::int64_t kv_heads = 1;
    constexpr std::int64_t width = 128;
    std::vector<float> query_values(
        static_cast<std::size_t>(heads * sequence * width));
    std::vector<float> key_values(
        static_cast<std::size_t>(kv_heads * sequence * width));
    std::vector<float> value_values(key_values.size());
    std::vector<float> seed_values(query_values.size());
    for (std::size_t index = 0; index < query_values.size(); ++index) {
        query_values[index] =
            static_cast<float>(static_cast<int>(index % 29) - 14) / 31.0F;
        seed_values[index] =
            static_cast<float>(static_cast<int>(index % 19) - 9) / 23.0F;
    }
    for (std::size_t index = 0; index < key_values.size(); ++index) {
        key_values[index] =
            static_cast<float>(static_cast<int>(index % 23) - 11) / 37.0F;
        value_values[index] =
            static_cast<float>(static_cast<int>(index % 17) - 8) / 19.0F;
    }
    const auto gpu = Device::hip(0);
    const auto query = Tensor::from_vector(
        query_values, {1, heads, sequence, width}).to(gpu);
    const auto key = Tensor::from_vector(
        key_values, {1, kv_heads, sequence, width}).to(gpu);
    const auto value = Tensor::from_vector(
        value_values, {1, sequence, kv_heads, width}).to(gpu);
    const auto seed = Tensor::from_vector(
        seed_values, {1, sequence, heads, width}).to(gpu);
    enable_attention_gqa_value_broadcast(false);
    enable_attention_gqa_forward_value_broadcast(false);
    const auto control = causal_gqa_attention_bthd_saved(
        query, key, value, 2, 1.0F / std::sqrt(static_cast<float>(width)));
    const auto control_gradients = causal_gqa_attention_bthd_backward_saved(
        query, key, value, control.second, seed, 2,
        1.0F / std::sqrt(static_cast<float>(width)));
    enable_attention_gqa_forward_value_broadcast(true);
    const auto forward_only = causal_gqa_attention_bthd_saved(
        query, key, value, 2, 1.0F / std::sqrt(static_cast<float>(width)));
    const auto forward_only_gradients = causal_gqa_attention_bthd_backward_saved(
        query, key, value, forward_only.second, seed, 2,
        1.0F / std::sqrt(static_cast<float>(width)));
    enable_attention_gqa_forward_value_broadcast(false);
    enable_attention_gqa_value_broadcast(true);
    runtime::reset_transfer_stats();
    const auto candidate = causal_gqa_attention_bthd_saved(
        query, key, value, 2, 1.0F / std::sqrt(static_cast<float>(width)));
    const auto candidate_gradients = causal_gqa_attention_bthd_backward_saved(
        query, key, value, candidate.second, seed, 2,
        1.0F / std::sqrt(static_cast<float>(width)));
    runtime::synchronize(gpu);
    const auto transfers = runtime::transfer_stats();
    EXPECT_EQ(transfers.host_to_device_calls, 0U);
    EXPECT_EQ(transfers.device_to_host_calls, 0U);
    expect_near(candidate.first.to_vector(), control.first.to_vector(), 3.0e-3F);
    expect_near(candidate.second.to_vector(), control.second.to_vector(), 3.0e-3F);
    expect_near(candidate_gradients.first.to_vector(),
                control_gradients.first.to_vector(), 3.0e-3F);
    expect_near(candidate_gradients.second.to_vector(),
                control_gradients.second.to_vector(), 3.0e-3F);
    expect_near(candidate_gradients.third.to_vector(),
                control_gradients.third.to_vector(), 3.0e-3F);
    expect_near(forward_only.first.to_vector(), control.first.to_vector(), 3.0e-3F);
    expect_near(forward_only.second.to_vector(), control.second.to_vector(), 3.0e-3F);
    expect_near(forward_only_gradients.first.to_vector(),
                control_gradients.first.to_vector(), 3.0e-3F);
    expect_near(forward_only_gradients.second.to_vector(),
                control_gradients.second.to_vector(), 3.0e-3F);
    expect_near(forward_only_gradients.third.to_vector(),
                control_gradients.third.to_vector(), 3.0e-3F);
    enable_attention_gqa_value_broadcast(false);
    enable_attention_gqa_forward_value_broadcast(false);
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

TEST(HipBackwardOpsTest, EmbeddingBackwardAddMatchesDenseReferenceWithoutTransfers) {
    require_gpu();
    const auto gpu = Device::hip(0);
    const auto initial = Tensor::from_vector(
        {10, 20, 30, 40, 50, 60, 70, 80}, {4, 2});
    const auto gradient = Tensor::from_vector({1, 2, 3, 4, 5, 6}, {3, 2});
    const auto indices = Tensor::from_int32_vector({1, 3, 1}, {3});
    const auto expected = add(
        initial, embedding_backward(gradient, indices, 4));
    auto actual = initial.to(gpu);
    const auto* address = actual.storage().data();
    const auto device_gradient = gradient.to(gpu);
    const auto device_indices = indices.to(gpu);
    runtime::reset_transfer_stats();
    embedding_backward_add_(actual, device_gradient, device_indices);
    runtime::synchronize(gpu);
    const auto transfers = runtime::transfer_stats();
    EXPECT_EQ(transfers.host_to_device_calls, 0U);
    EXPECT_EQ(transfers.device_to_host_calls, 0U);
    EXPECT_EQ(actual.storage().data(), address);
    expect_near(actual.to_vector(), expected.to_vector(), 1.0e-6F);
}

TEST(HipOpsTest, FusedResidualRmsNormMatchesComposedCpuWithoutTransfers) {
    require_gpu();
    const auto gpu = Device::hip();
    const auto left = Tensor::from_vector({1, 2, 3, -1, -2, -3}, {2, 3});
    const auto right = Tensor::from_vector({0.5F, -0.5F, 1, 2, 1, 0}, {2, 3});
    const auto weight = Tensor::from_vector({1, 0.5F, 2}, {3});
    const auto expected_sum = add(left, right);
    const auto expected_norm = rms_norm(expected_sum, weight);
    const auto device_left = left.to(gpu);
    const auto device_right = right.to(gpu);
    const auto device_weight = weight.to(gpu);
    runtime::reset_transfer_stats();
    const auto actual = add_rms_norm(device_left, device_right, device_weight);
    runtime::synchronize(gpu);
    const auto transfers = runtime::transfer_stats();
    EXPECT_EQ(transfers.host_to_device_calls, 0U);
    EXPECT_EQ(transfers.device_to_host_calls, 0U);
    expect_near(actual.first.to_vector(), expected_sum.to_vector());
    expect_near(actual.second.to_vector(), expected_norm.to_vector(), 2.0e-4F);
}

TEST(HipOpsTest, PairedKvStoreMatchesTwoCpuStoresWithoutPayloadTransfers) {
    require_gpu();
    const auto gpu = Device::hip();
    Tensor key_backing({1, 2, 2, 2}, DType::Float32, gpu);
    Tensor value_backing({1, 2, 2, 2}, DType::Float32, gpu);
    auto key_cache = Tensor::from_storage(key_backing.storage(), {1, 2, 1, 2},
                                          key_backing.strides(), 0, DType::Float32);
    auto value_cache = Tensor::from_storage(value_backing.storage(), {1, 2, 1, 2},
                                            value_backing.strides(), 0, DType::Float32);
    const auto key = Tensor::from_vector({1, 2, 3, 4}, {1, 2, 1, 2});
    const auto value = Tensor::from_vector({5, 6, 7, 8}, {1, 2, 1, 2});
    const auto device_key = key.to(gpu);
    const auto device_value = value.to(gpu);
    runtime::reset_transfer_stats();
    kv_cache_store_pair_(key_cache, value_cache, device_key, device_value, 0);
    runtime::synchronize(gpu);
    const auto transfers = runtime::transfer_stats();
    EXPECT_EQ(transfers.host_to_device_calls, 0U);
    EXPECT_EQ(transfers.device_to_host_calls, 0U);
    expect_near(key_cache.to_vector(), key.to_vector());
    expect_near(value_cache.to_vector(), value.to_vector());
}

TEST(HipOpsTest, PairedBf16KvStoreRoundsOnDeviceWithoutPayloadTransfers) {
    require_gpu();
    const auto gpu = Device::hip();
    Tensor key_backing({2, 1, 2, 2}, DType::BFloat16, gpu);
    Tensor value_backing({2, 1, 2, 2}, DType::BFloat16, gpu);
    auto key_cache = Tensor::from_storage(key_backing.storage(), {2, 1, 1, 2},
                                          key_backing.strides(), 0,
                                          DType::BFloat16);
    auto value_cache = Tensor::from_storage(value_backing.storage(), {2, 1, 1, 2},
                                            value_backing.strides(), 0,
                                            DType::BFloat16);
    const auto key = Tensor::from_vector(
        {1.00390625F, 2.01171875F, 3.01953125F, 4.02734375F},
        {2, 1, 1, 2});
    const auto value = Tensor::from_vector(
        {5.03515625F, 6.04296875F, 7.05078125F, 8.05859375F},
        {2, 1, 1, 2});
    const auto device_key = key.to(gpu);
    const auto device_value = value.to(gpu);
    runtime::reset_transfer_stats();
    kv_cache_store_pair_(key_cache, value_cache, device_key, device_value, 0);
    runtime::synchronize(gpu);
    const auto transfers = runtime::transfer_stats();
    EXPECT_EQ(transfers.host_to_device_calls, 0U);
    EXPECT_EQ(transfers.device_to_host_calls, 0U);
    expect_near(key_cache.to_vector(), key.cast(DType::BFloat16).to_vector());
    expect_near(value_cache.to_vector(), value.cast(DType::BFloat16).to_vector());
    EXPECT_THROW(kv_cache_store_pair_(key_cache, value_cache, device_key,
                                      cast(device_value, DType::BFloat16), 0),
                 std::invalid_argument);
}

TEST(HipPositionedDecodeTest, RopeStoreAndAttentionMatchCpuWithoutPayloadTransfers) {
    require_gpu();
    const auto gpu = Device::hip(0);
    const auto rope_input = Tensor::from_vector(
        {1, 2, 3, 4, 5, 6, 7, 8,
         2, 3, 4, 5, 6, 7, 8, 9},
        {2, 2, 1, 4});
    const auto positions = Tensor::from_int32_vector({0, 2}, {2});
    const auto rows = Tensor::from_int32_vector({2, 0}, {2});
    const auto bias = Tensor::from_vector(
        {0.1F, 0.2F, 0.3F, 0.4F, -0.1F, -0.2F, -0.3F, -0.4F}, {8});
    const auto device_input = rope_input.to(gpu);
    const auto device_positions = positions.to(gpu);
    const auto device_rows = rows.to(gpu);
    const auto device_bias = bias.to(gpu);
    runtime::reset_transfer_stats();
    const auto interleaved = rope_positions(device_input, device_positions);
    const auto split = rope_split_half_positions(device_input, device_positions);
    const auto fused = rope_split_half_bias_positions(
        device_input, device_bias, device_positions);
    runtime::synchronize(gpu);
    EXPECT_EQ(runtime::transfer_stats().device_to_host_calls, 0U);
    expect_near(interleaved.to_vector(),
                rope_positions(rope_input, positions).to_vector(), 2.0e-5F);
    expect_near(split.to_vector(),
                rope_split_half_positions(rope_input, positions).to_vector(),
                2.0e-5F);
    expect_near(fused.to_vector(),
                rope_split_half_bias_positions(
                    rope_input, bias, positions).to_vector(),
                2.0e-5F);

    const auto current_key = Tensor::from_vector(
        {3.01953125F, 4.02734375F, 1.00390625F, 2.01171875F},
        {2, 1, 1, 2});
    const auto current_value = Tensor::from_vector(
        {7.05078125F, 8.05859375F, 5.03515625F, 6.04296875F},
        {2, 1, 1, 2});
    const auto query = Tensor::from_vector(
        {1, 0, 0, 1, 1, 0, 0, 1}, {2, 2, 1, 2});
    for (const auto dtype : {DType::Float32, DType::BFloat16}) {
        Tensor cpu_key_backing({3, 1, 4, 2}, dtype);
        Tensor cpu_value_backing({3, 1, 4, 2}, dtype);
        fill_(cpu_key_backing, 0.0F);
        fill_(cpu_value_backing, 0.0F);
        auto cpu_key = Tensor::from_storage(
            cpu_key_backing.storage(), {3, 1, 3, 2},
            cpu_key_backing.strides(), 0, dtype);
        auto cpu_value = Tensor::from_storage(
            cpu_value_backing.storage(), {3, 1, 3, 2},
            cpu_value_backing.strides(), 0, dtype);
        kv_cache_store_pair_positions_(
            cpu_key, cpu_value, current_key, current_value, positions, rows);
        const auto expected = cached_gqa_attention_positions(
            query, cpu_key, cpu_value, positions, rows, 2, 1.0F).to_vector();

        Tensor key_backing({3, 1, 4, 2}, dtype, gpu);
        Tensor value_backing({3, 1, 4, 2}, dtype, gpu);
        fill_(key_backing, 0.0F);
        fill_(value_backing, 0.0F);
        auto key_cache = Tensor::from_storage(
            key_backing.storage(), {3, 1, 3, 2}, key_backing.strides(), 0,
            dtype);
        auto value_cache = Tensor::from_storage(
            value_backing.storage(), {3, 1, 3, 2}, value_backing.strides(), 0,
            dtype);
        const auto device_key = current_key.to(gpu);
        const auto device_value = current_value.to(gpu);
        const auto device_query = query.to(gpu);
        runtime::reset_transfer_stats();
        kv_cache_store_pair_positions_(
            key_cache, value_cache, device_key, device_value,
            device_positions, device_rows);
        const auto actual = cached_gqa_attention_positions(
            device_query, key_cache, value_cache, device_positions,
            device_rows, 2, 1.0F);
        runtime::synchronize(gpu);
        const auto transfers = runtime::transfer_stats();
        EXPECT_EQ(transfers.host_to_device_calls, 0U);
        EXPECT_EQ(transfers.device_to_host_calls, 0U);
        expect_near(actual.to_vector(), expected,
                    dtype == DType::Float32 ? 2.0e-5F : 3.0e-2F);
        expect_near(key_cache.to_vector(), cpu_key.to_vector(), 3.0e-2F);
        expect_near(value_cache.to_vector(), cpu_value.to_vector(), 3.0e-2F);
    }
}

TEST(HipPositionedDecodeTest, LongFallbackMasksEachActivePrefix) {
    require_gpu();
    constexpr std::int64_t sequence = 4097;
    std::vector<float> cache_values(static_cast<std::size_t>(2 * sequence * 2));
    for (std::size_t index = 0; index < cache_values.size(); ++index) {
        cache_values[index] =
            static_cast<float>(static_cast<int>(index % 37U) - 18) * 0.03125F;
    }
    const auto cache = Tensor::from_vector(cache_values, {2, 1, sequence, 2});
    const auto query = Tensor::from_vector({1, -0.5F, -0.25F, 0.75F},
                                           {2, 1, 1, 2});
    const auto positions = Tensor::from_int32_vector({0, 4096}, {2});
    const auto rows = Tensor::from_int32_vector({0, 1}, {2});
    const auto expected = cached_gqa_attention_positions(
        query, cache, cache, positions, rows, 1, 0.70710678F).to_vector();
    const auto gpu = Device::hip(0);
    const auto device_query = query.to(gpu);
    const auto device_cache = cache.to(gpu);
    const auto device_positions = positions.to(gpu);
    const auto device_rows = rows.to(gpu);
    runtime::reset_transfer_stats();
    const auto actual = cached_gqa_attention_positions(
        device_query, device_cache, device_cache, device_positions,
        device_rows, 1, 0.70710678F);
    runtime::synchronize(gpu);
    EXPECT_EQ(runtime::transfer_stats().device_to_host_calls, 0U);
    expect_near(actual.to_vector(), expected, 4.0e-4F);
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
    cpu_cache.reset();
    hip_cache.reset();
    const auto host_prefix = Tensor::from_int32_vector({3, 4, 5}, {1, 3});
    const auto device_prefix = host_prefix.to(Device::hip());
    const auto expected_prefix =
        cpu_model.forward_prefill_cached(host_prefix, cpu_cache).to_vector();
    runtime::reset_transfer_stats();
    const auto actual_prefix =
        hip_model.forward_prefill_cached(device_prefix, hip_cache);
    runtime::synchronize(Device::hip());
    const auto prefix_transfers = runtime::transfer_stats();
    EXPECT_EQ(prefix_transfers.host_to_device_calls, 0U);
    EXPECT_EQ(prefix_transfers.device_to_host_calls, 0U);
    expect_near(actual_prefix.to_vector(), expected_prefix, 4.0e-4F);
    EXPECT_EQ(hip_cache.position(), 3);
    EXPECT_EQ(hip_cache.layer(0).key.storage().num_bytes(),
              static_cast<std::size_t>(config.kv_heads * config.max_sequence_length *
                                       config.head_dimension()) * sizeof(float));
    const auto host_next = Tensor::from_int32_vector({6}, {1, 1});
    const auto expected_next = cpu_model.forward_cached(host_next, cpu_cache).to_vector();
    const auto actual_next = hip_model.forward_cached(
        host_next.to(Device::hip()), hip_cache).to_vector();
    expect_near(actual_next, expected_next, 4.0e-4F);

    model::TransformerModel batch_cpu_model(config, 71);
    model::TransformerModel batch_hip_model(config, 71);
    batch_hip_model.to(Device::hip());
    inference::KVCache batch_cpu_cache(config.layers, config.max_sequence_length, 2);
    inference::KVCache batch_hip_cache(config.layers, config.max_sequence_length, 2);
    const auto batch_prefix = Tensor::from_int32_vector(
        {3, 4, 5, 6, 5, 4}, {2, 3});
    const auto batch_expected = batch_cpu_model.forward_prefill_cached(
        batch_prefix, batch_cpu_cache).to_vector();
    const auto batch_device_prefix = batch_prefix.to(Device::hip());
    runtime::reset_transfer_stats();
    const auto batch_actual = batch_hip_model.forward_prefill_cached(
        batch_device_prefix, batch_hip_cache);
    runtime::synchronize(Device::hip());
    const auto batch_transfers = runtime::transfer_stats();
    EXPECT_EQ(batch_transfers.host_to_device_calls, 0U);
    EXPECT_EQ(batch_transfers.device_to_host_calls, 0U);
    expect_near(batch_actual.to_vector(), batch_expected, 5.0e-4F);
    const auto batch_next = Tensor::from_int32_vector({6, 3}, {2, 1});
    const auto batch_expected_next = batch_cpu_model.forward_cached(
        batch_next, batch_cpu_cache).to_vector();
    const auto batch_actual_next = batch_hip_model.forward_cached(
        batch_next.to(Device::hip()), batch_hip_cache).to_vector();
    expect_near(batch_actual_next, batch_expected_next, 5.0e-4F);
}

TEST(HipModelTest, ClearCacheRowIsDeviceNativeAndMatchesCpuStorage) {
    require_gpu();
    const model::ModelConfig config{.vocabulary_size = 16,
                                    .dimension = 8,
                                    .layers = 2,
                                    .heads = 2,
                                    .kv_heads = 1,
                                    .ffn_dimension = 16,
                                    .max_sequence_length = 6,
                                    .rope_base = 10000.0F,
                                    .tie_embeddings = false};
    model::TransformerModel cpu_model(config, 75);
    model::TransformerModel hip_model(config, 75);
    hip_model.to(Device::hip(0));
    inference::KVCache cpu_cache(config.layers, config.max_sequence_length, 2,
                                 DType::BFloat16);
    inference::KVCache hip_cache(config.layers, config.max_sequence_length, 2,
                                 DType::BFloat16);
    const auto prefix = Tensor::from_int32_vector({1, 2, 3, 4, 5, 6}, {2, 3});
    (void)cpu_model.forward_prefill_cached(prefix, cpu_cache);
    (void)hip_model.forward_prefill_cached(prefix.to(Device::hip(0)), hip_cache);
    runtime::reset_transfer_stats();
    cpu_cache.clear_row(0);
    hip_cache.clear_row(0);
    runtime::synchronize(Device::hip(0));
    const auto transfers = runtime::transfer_stats();
    EXPECT_EQ(transfers.host_to_device_calls, 0U);
    EXPECT_EQ(transfers.device_to_host_calls, 0U);
    EXPECT_EQ(cpu_cache.position(), 3);
    EXPECT_EQ(hip_cache.position(), 3);
    for (std::size_t layer = 0; layer < hip_cache.layer_count(); ++layer) {
        expect_near(hip_cache.layer(layer).key.to_vector(),
                    cpu_cache.layer(layer).key.to_vector());
        expect_near(hip_cache.layer(layer).value.to_vector(),
                    cpu_cache.layer(layer).value.to_vector());
    }
    const auto next = Tensor::from_int32_vector({7, 8}, {2, 1});
    const auto expected = cpu_model.forward_cached(next, cpu_cache).to_vector();
    const auto actual = hip_model.forward_cached(next.to(Device::hip(0)), hip_cache)
                            .to_vector();
    expect_near(actual, expected, 8.0e-4F);
    for (std::size_t layer = 0; layer < hip_cache.layer_count(); ++layer) {
        expect_near(hip_cache.layer(layer).key.to_vector(),
                    cpu_cache.layer(layer).key.to_vector());
        expect_near(hip_cache.layer(layer).value.to_vector(),
                    cpu_cache.layer(layer).value.to_vector());
    }
    runtime::reset_transfer_stats();
    cpu_cache.reset_row(0);
    hip_cache.reset_row(0);
    runtime::synchronize(Device::hip(0));
    EXPECT_EQ(runtime::transfer_stats().host_to_device_calls, 0U);
    EXPECT_EQ(runtime::transfer_stats().device_to_host_calls, 0U);
    EXPECT_EQ(hip_cache.row_positions(), (std::vector<std::int64_t>{0, 4}));
    EXPECT_THROW((void)hip_cache.position(), std::logic_error);
    hip_cache.advance_row(0, 4);
    cpu_cache.advance_row(0, 4);
    EXPECT_TRUE(hip_cache.positions_uniform());
    EXPECT_EQ(hip_cache.position(), 4);
    EXPECT_EQ(cpu_cache.position(), 4);
}

TEST(HipModelTest, MixedLayerKvCacheMatchesCpuAndKeepsLayerDtypes) {
    require_gpu();
    const model::ModelConfig config{.vocabulary_size = 16,
                                    .dimension = 8,
                                    .layers = 2,
                                    .heads = 2,
                                    .kv_heads = 1,
                                    .ffn_dimension = 16,
                                    .max_sequence_length = 4,
                                    .rope_base = 10000.0F,
                                    .tie_embeddings = false};
    const std::vector<DType> policy{DType::BFloat16, DType::Float32};
    model::TransformerModel cpu_model(config, 73);
    model::TransformerModel hip_model(config, 73);
    hip_model.to(Device::hip());
    inference::KVCache cpu_cache(policy, config.max_sequence_length);
    inference::KVCache hip_cache(policy, config.max_sequence_length);
    const auto prefix = Tensor::from_int32_vector({1, 2, 3}, {1, 3});
    const auto expected = cpu_model.forward_prefill_cached(prefix, cpu_cache).to_vector();
    const auto device_prefix = prefix.to(Device::hip());
    runtime::reset_transfer_stats();
    const auto actual = hip_model.forward_prefill_cached(device_prefix, hip_cache);
    runtime::synchronize(Device::hip());
    const auto transfers = runtime::transfer_stats();
    EXPECT_EQ(transfers.host_to_device_calls, 0U);
    EXPECT_EQ(transfers.device_to_host_calls, 0U);
    expect_near(actual.to_vector(), expected, 6.0e-4F);
    EXPECT_EQ(hip_cache.layer(0).key.dtype(), DType::BFloat16);
    EXPECT_EQ(hip_cache.layer(1).key.dtype(), DType::Float32);
    EXPECT_EQ(hip_cache.layer(0).key.storage().num_bytes() * 2U,
              hip_cache.layer(1).key.storage().num_bytes());
    const auto next = Tensor::from_int32_vector({4}, {1, 1});
    const auto expected_next = cpu_model.forward_cached(next, cpu_cache).to_vector();
    const auto actual_next = hip_model.forward_cached(
        next.to(Device::hip()), hip_cache).to_vector();
    expect_near(actual_next, expected_next, 6.0e-4F);
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

TEST(HipTrainingTest, VectorizedAdamWMatchesScalarAcrossTailAndMirror) {
    require_gpu();
    const auto gpu = Device::hip(0);
    constexpr std::int64_t elements = 4099;
    std::vector<float> parameter_values(static_cast<std::size_t>(elements));
    std::vector<float> gradient_values(static_cast<std::size_t>(elements));
    for (std::size_t index = 0; index < parameter_values.size(); ++index) {
        parameter_values[index] = static_cast<float>(static_cast<int>(index % 31) - 15) / 17.0F;
        gradient_values[index] = static_cast<float>(static_cast<int>(index % 13) - 6) / 19.0F;
    }
    auto scalar_parameter = Tensor::from_vector(parameter_values, {elements}).to(gpu);
    auto vector_parameter = Tensor::from_vector(parameter_values, {elements}).to(gpu);
    const auto gradient = Tensor::from_vector(gradient_values, {elements}).to(gpu);
    Tensor scalar_first({elements}, DType::Float32, gpu);
    Tensor scalar_second({elements}, DType::Float32, gpu);
    Tensor vector_first({elements}, DType::Float32, gpu);
    Tensor vector_second({elements}, DType::Float32, gpu);
    fill_(scalar_first, 0.0F);
    fill_(scalar_second, 0.0F);
    fill_(vector_first, 0.0F);
    fill_(vector_second, 0.0F);
    auto scalar_mirror = scalar_parameter.cast(DType::BFloat16);
    auto vector_mirror = vector_parameter.cast(DType::BFloat16);
    runtime::reset_transfer_stats();
    for (const auto step : {1, 2}) {
        const auto first_correction = 1.0F - std::pow(0.9F, static_cast<float>(step));
        const auto second_correction = 1.0F - std::pow(0.99F, static_cast<float>(step));
        adamw_update_bf16_mirror_(
            scalar_parameter, gradient, scalar_first, scalar_second, scalar_mirror,
            0.01F, 0.9F, 0.99F, 1.0e-8F, 0.1F,
            first_correction, second_correction, {}, AdamWImplementation::Scalar);
        adamw_update_bf16_mirror_(
            vector_parameter, gradient, vector_first, vector_second, vector_mirror,
            0.01F, 0.9F, 0.99F, 1.0e-8F, 0.1F,
            first_correction, second_correction, {}, AdamWImplementation::Vectorized);
    }
    runtime::synchronize(gpu);
    const auto transfers = runtime::transfer_stats();
    EXPECT_EQ(transfers.host_to_device_calls, 0U);
    EXPECT_EQ(transfers.device_to_host_calls, 0U);
    expect_near(vector_parameter.to_vector(), scalar_parameter.to_vector(), 2.0e-6F);
    expect_near(vector_first.to_vector(), scalar_first.to_vector(), 2.0e-6F);
    expect_near(vector_second.to_vector(), scalar_second.to_vector(), 2.0e-6F);
    EXPECT_EQ(vector_mirror.to_vector(), scalar_mirror.to_vector());

    auto unaligned_parameter = Tensor({elements + 1}, DType::Float32, gpu).slice(
        0, 1, elements + 1);
    auto unaligned_gradient = Tensor({elements + 1}, DType::Float32, gpu).slice(
        0, 1, elements + 1);
    auto unaligned_first = Tensor({elements + 1}, DType::Float32, gpu).slice(
        0, 1, elements + 1);
    auto unaligned_second = Tensor({elements + 1}, DType::Float32, gpu).slice(
        0, 1, elements + 1);
    EXPECT_THROW(adamw_update_(
                     unaligned_parameter, unaligned_gradient,
                     unaligned_first, unaligned_second, 0.01F, 0.9F, 0.99F,
                     1.0e-8F, 0.1F, 0.1F, 0.01F, {},
                     AdamWImplementation::Vectorized),
                 std::invalid_argument);
}

TEST(HipTrainingTest, AdamWAutotuneChecksEveryStateBeforeTimingAndRegistration) {
    require_gpu();
    const auto gpu = Device::hip(0);
    constexpr std::int64_t elements = 4099;
    std::vector<float> parameter_values(static_cast<std::size_t>(elements));
    std::vector<float> gradient_values(static_cast<std::size_t>(elements));
    std::vector<float> first_values(static_cast<std::size_t>(elements));
    std::vector<float> second_values(static_cast<std::size_t>(elements));
    for (std::size_t index = 0; index < parameter_values.size(); ++index) {
        parameter_values[index] =
            static_cast<float>(static_cast<int>(index % 31) - 15) / 17.0F;
        gradient_values[index] =
            static_cast<float>(static_cast<int>(index % 13) - 6) / 19.0F;
        first_values[index] =
            static_cast<float>(static_cast<int>(index % 7) - 3) / 101.0F;
        second_values[index] =
            static_cast<float>(index % 11) / 97.0F;
    }
    const auto parameter = Tensor::from_vector(parameter_values, {elements}).to(gpu);
    const auto gradient = Tensor::from_vector(gradient_values, {elements}).to(gpu);
    const auto first = Tensor::from_vector(first_values, {elements}).to(gpu);
    const auto second = Tensor::from_vector(second_values, {elements}).to(gpu);
    const auto mirror = parameter.cast(DType::BFloat16);
    AdamWAutotuneOptions options;
    options.warmup = 1;
    options.repetitions = 5;
    options.mode = OpMode::Training;
    const auto before_parameter = parameter.to_vector();
    const auto before_first = first.to_vector();
    clear_adamw_implementation_registry();
    const auto report = autotune_adamw(
        parameter, gradient, first, second, &mirror, options);
    EXPECT_EQ(report.reference_elements, elements);
    EXPECT_EQ(report.key.mode, OpMode::Training);
    EXPECT_TRUE(report.key.bf16_mirror);
    EXPECT_TRUE(report.key.parameter_aligned16);
    EXPECT_EQ(parameter.to_vector(), before_parameter)
        << "screening must not mutate the caller's parameter";
    EXPECT_EQ(first.to_vector(), before_first)
        << "screening must not mutate the caller's moment";
    EXPECT_EQ(adamw_registered_implementation_count(), 0U)
        << "screening must not change live Auto dispatch";
    ASSERT_EQ(report.candidates.size(), 2U);
    for (const auto& candidate : report.candidates) {
        EXPECT_TRUE(candidate.supported) << candidate.failure;
        EXPECT_TRUE(candidate.correctness_passed) << candidate.failure;
        EXPECT_TRUE(candidate.finite);
        EXPECT_LE(candidate.parameter.maximum_absolute_error,
                  report.maximum_absolute_tolerance);
        EXPECT_LE(candidate.first_moment.maximum_absolute_error,
                  report.maximum_absolute_tolerance);
        EXPECT_LE(candidate.second_moment.maximum_absolute_error,
                  report.maximum_absolute_tolerance);
        EXPECT_LE(candidate.bf16_mirror.maximum_absolute_error,
                  report.maximum_absolute_tolerance);
        EXPECT_GT(candidate.event_ms_p50, 0.0);
        EXPECT_GE(candidate.event_ms_p95, candidate.event_ms_p50);
        EXPECT_GT(candidate.wall_ms_p50, 0.0);
        EXPECT_GE(candidate.wall_ms_p95, candidate.wall_ms_p50);
    }
    EXPECT_NE(report.recommended, AdamWImplementation::Auto);
    register_adamw_autotune_winner(report);
    EXPECT_EQ(adamw_registered_implementation_count(), 1U);
    EXPECT_EQ(choose_adamw_implementation(
                  parameter, gradient, first, second, &mirror,
                  OpContext{.mode = OpMode::Training}),
              report.recommended);
    EXPECT_EQ(choose_adamw_implementation(
                  parameter, gradient, first, second, &mirror),
              AdamWImplementation::Scalar)
        << "a training-mode result must not leak into unspecified mode";
    const auto cache = std::filesystem::temp_directory_path() /
                       "microllm-adamw-tuning-cache-hip.jsonl";
    save_adamw_tuning_cache(cache);
    clear_adamw_implementation_registry();
    const auto loaded = load_adamw_tuning_cache(cache, gpu);
    EXPECT_EQ(loaded.parsed_entries, 1U);
    EXPECT_EQ(loaded.loaded_entries, 1U);
    EXPECT_EQ(loaded.stale_entries, 0U);
    EXPECT_EQ(choose_adamw_implementation(
                  parameter, gradient, first, second, &mirror,
                  OpContext{.mode = OpMode::Training}),
              report.recommended);
    clear_adamw_implementation_registry();
    std::error_code ignored;
    std::filesystem::remove(cache, ignored);

    auto unaligned_parameter = Tensor({elements + 1}, DType::Float32, gpu).slice(
        0, 1, elements + 1);
    auto unaligned_gradient = Tensor({elements + 1}, DType::Float32, gpu).slice(
        0, 1, elements + 1);
    auto unaligned_first = Tensor({elements + 1}, DType::Float32, gpu).slice(
        0, 1, elements + 1);
    auto unaligned_second = Tensor({elements + 1}, DType::Float32, gpu).slice(
        0, 1, elements + 1);
    fill_(unaligned_parameter, 1.0F);
    fill_(unaligned_gradient, 0.5F);
    fill_(unaligned_first, 0.0F);
    fill_(unaligned_second, 0.0F);
    const auto rejected = autotune_adamw(
        unaligned_parameter, unaligned_gradient, unaligned_first,
        unaligned_second, nullptr, options);
    const auto vectorized = std::find_if(
        rejected.candidates.begin(), rejected.candidates.end(),
        [](const auto& candidate) {
            return candidate.implementation == AdamWImplementation::Vectorized;
        });
    ASSERT_NE(vectorized, rejected.candidates.end());
    EXPECT_FALSE(vectorized->supported);
    EXPECT_FALSE(vectorized->correctness_passed);
    EXPECT_EQ(vectorized->event_ms_p50, 0.0);
    EXPECT_EQ(vectorized->wall_ms_p50, 0.0);
    EXPECT_NE(vectorized->failure.find("16-byte aligned"), std::string::npos);
    EXPECT_EQ(rejected.recommended, AdamWImplementation::Scalar);
    EXPECT_THROW(register_adamw_implementation(
                     rejected.key, AdamWImplementation::Vectorized),
                 std::invalid_argument);
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
    const auto registered_left = left_cpu.to(Device::hip());
    const auto registered_right = right_cpu.to(Device::hip());
    EXPECT_EQ(choose_matmul_implementation(registered_left, registered_right),
              MatmulImplementation::Readable);
    const Tensor large_left({128, 128}, DType::Float32, Device::hip());
    const Tensor large_right({128, 128}, DType::Float32, Device::hip());
    EXPECT_EQ(choose_matmul_implementation(large_left, large_right),
              MatmulImplementation::HipBLASLt);
    const auto fp32_key = make_matmul_tuning_key(
        registered_left, registered_right);
    EXPECT_EQ(fp32_key.dtype, DType::Float32);
    EXPECT_FALSE(fp32_key.architecture.empty());
    EXPECT_GT(fp32_key.hip_runtime_version, 0);
    EXPECT_GT(fp32_key.hip_driver_version, 0);
    EXPECT_GT(fp32_key.hipblaslt_version, 0);
    EXPECT_EQ(fp32_key.left_strides, (Strides{64, 1}));
    register_matmul_implementation(fp32_key, MatmulImplementation::HipBLASLt);
    EXPECT_EQ(matmul_registered_implementation_count(), 1U);
    EXPECT_EQ(choose_matmul_implementation(registered_left, registered_right),
              MatmulImplementation::HipBLASLt);

    const auto fp16_left = registered_left.cast(DType::Float16);
    const auto fp16_right = registered_right.cast(DType::Float16);
    EXPECT_EQ(choose_matmul_implementation(fp16_left, fp16_right),
              MatmulImplementation::Readable)
        << "an FP32 tuning choice must not leak into FP16";
    EXPECT_EQ(choose_matmul_implementation(
                  registered_left, registered_right, true, true),
              MatmulImplementation::Readable)
        << "an NN tuning choice must not leak into TT";

    OpContext training_context;
    training_context.mode = OpMode::Training;
    EXPECT_EQ(choose_matmul_implementation(
                  registered_left, registered_right, false, false,
                  training_context),
              MatmulImplementation::Readable)
        << "an unspecified-mode choice must not leak into training";
    OpContext workspace_context;
    workspace_context.workspace_bytes = 4096;
    EXPECT_EQ(choose_matmul_implementation(
                  registered_left, registered_right, false, false,
                  workspace_context),
              MatmulImplementation::Readable)
        << "a zero-workspace choice must not leak into another budget";
    clear_matmul_implementation_registry();
    EXPECT_EQ(matmul_registered_implementation_count(), 0U);
    EXPECT_EQ(choose_matmul_implementation(registered_left, registered_right),
              MatmulImplementation::Readable);

    const Tensor weight_gradient_left({32, 128}, DType::Float32, Device::hip());
    const Tensor weight_gradient_right({32, 256}, DType::Float32, Device::hip());
    EXPECT_EQ(choose_matmul_implementation(
                  weight_gradient_left, weight_gradient_right, true, false),
              MatmulImplementation::HipBLASLt);
    register_matmul_implementation(
        make_matmul_tuning_key(
            weight_gradient_left, weight_gradient_right, true, false),
        MatmulImplementation::Readable);
    EXPECT_EQ(choose_matmul_implementation(
                  weight_gradient_left, weight_gradient_right, true, false),
              MatmulImplementation::Readable);
    clear_matmul_implementation_registry();
}

TEST(HipOptimizedOpsTest, ExactFp32AttentionSolutionDispatchesAndCaches) {
    require_gpu();
    const auto gpu = Device::hip(0);
    const auto info = runtime::device_info(gpu);
    if (!info.architecture.starts_with("gfx942") ||
        hipblaslt_version() != 10300) {
        GTEST_SKIP() << "recorded solution index is local to gfx942 hipBLASLt 1.3.0";
    }
    constexpr std::int64_t heads = 14;
    constexpr std::int64_t sequence = 512;
    constexpr std::int64_t width = 64;
    std::vector<float> left_values(
        static_cast<std::size_t>(heads * sequence * width));
    std::vector<float> right_values(left_values.size());
    for (std::size_t index = 0; index < left_values.size(); ++index) {
        left_values[index] =
            static_cast<float>(static_cast<int>(index % 29) - 14) / 127.0F;
        right_values[index] =
            static_cast<float>(static_cast<int>(index % 31) - 15) / 131.0F;
    }
    const auto left = Tensor::from_vector(
        left_values, {1, heads, sequence, width}).to(gpu);
    const auto right = Tensor::from_vector(
        right_values, {1, heads, sequence, width}).to(gpu);

    clear_fp32_matmul_solution_registry();
    const auto reference = matmul_with_implementation(
        left, right, MatmulImplementation::HipBLASLt, false, true).to_vector();
    const auto key = make_fp32_matmul_solution_key(
        left.shape(), right.shape(), gpu, false, true);
    EXPECT_THROW(register_fp32_matmul_solution(key, -1), std::invalid_argument);
    auto stale = key;
    ++stale.hip_runtime_version;
    EXPECT_THROW(register_fp32_matmul_solution(stale, 311017),
                 std::invalid_argument);

    register_fp32_matmul_solution(key, 311017);
    auto stats = fp32_matmul_solution_stats();
    EXPECT_EQ(stats.registered_entries, 1U);
    EXPECT_EQ(stats.cached_algorithms, 0U);
    const auto first = matmul_with_implementation(
        left, right, MatmulImplementation::HipBLASLt, false, true).to_vector();
    expect_near(first, reference, 1.0e-5F);
    stats = fp32_matmul_solution_stats();
    EXPECT_EQ(stats.registry_hits, 1U);
    EXPECT_EQ(stats.registry_misses, 0U);
    EXPECT_EQ(stats.cache_hits, 0U);
    EXPECT_EQ(stats.cache_misses, 1U);
    EXPECT_EQ(stats.cached_algorithms, 1U);
    EXPECT_EQ(stats.dispatches, 1U);

    const auto second = matmul_with_implementation(
        left, right, MatmulImplementation::HipBLASLt, false, true).to_vector();
    expect_near(second, reference, 1.0e-5F);
    stats = fp32_matmul_solution_stats();
    EXPECT_EQ(stats.registry_hits, 2U);
    EXPECT_EQ(stats.cache_hits, 1U);
    EXPECT_EQ(stats.cache_misses, 1U);
    EXPECT_EQ(stats.dispatches, 2U);
    clear_fp32_matmul_solution_registry();
    EXPECT_EQ(fp32_matmul_solution_stats().registered_entries, 0U);
}

TEST(HipOptimizedOpsTest, PerDeviceHandlesSurviveAlternatingGpuMatmuls) {
    if (runtime::hip_device_count() < 2) {
        GTEST_SKIP() << "two visible HIP devices required";
    }
    constexpr std::int64_t width = 16;
    std::vector<float> values(static_cast<std::size_t>(width * width), 1.0F);
    const auto fp32_left = Tensor::from_vector(values, {width, width});
    const auto fp32_right = Tensor::from_vector(values, {width, width});
    const auto bf16_left = Tensor::from_vector(
        values, {width, width}, DType::BFloat16);
    const auto bf16_right = Tensor::from_vector(
        values, {width, width}, DType::BFloat16);
    const std::vector<float> expected(
        static_cast<std::size_t>(width * width), static_cast<float>(width));
    clear_bf16_plan_cache();
    for (const auto index : {0, 1, 0, 1}) {
        const auto device = Device::hip(index);
        const auto fp32_output = matmul_with_implementation(
            fp32_left.to(device), fp32_right.to(device),
            MatmulImplementation::HipBLASLt);
        const auto bf16_output = bf16_matmul_output(
            bf16_left.to(device), bf16_right.to(device), DType::Float32);
        runtime::synchronize(device);
        expect_near(fp32_output.to_vector(), expected, 1.0e-5F);
        expect_near(bf16_output.to_vector(), expected, 1.0e-5F);
    }
    const auto stats = bf16_plan_cache_stats();
    EXPECT_EQ(stats.entries, 2U);
    EXPECT_EQ(stats.misses, 2U);
    EXPECT_EQ(stats.hits, 2U);
    clear_bf16_plan_cache();
}

TEST(HipOptimizedOpsTest, ScaledHipblasLtMatmulUsesAlphaWithoutPayloadTransfers) {
    require_gpu();
    const auto gpu = Device::hip(0);
    std::vector<float> left_values(64 * 32);
    std::vector<float> right_values(64 * 48);
    for (std::size_t index = 0; index < left_values.size(); ++index) {
        left_values[index] =
            static_cast<float>(static_cast<int>(index % 17) - 8) / 19.0F;
    }
    for (std::size_t index = 0; index < right_values.size(); ++index) {
        right_values[index] =
            static_cast<float>(static_cast<int>(index % 23) - 11) / 29.0F;
    }
    const auto left = Tensor::from_vector(left_values, {64, 32});
    const auto right = Tensor::from_vector(right_values, {64, 48});
    const auto expected = scale(matmul(
        left.transpose(0, 1).contiguous(), right), 0.125F);
    const auto device_left = left.to(gpu);
    const auto device_right = right.to(gpu);
    runtime::reset_transfer_stats();
    const auto actual = matmul_scaled_with_implementation(
        device_left, device_right, 0.125F,
        MatmulImplementation::HipBLASLt, true, false);
    runtime::synchronize(gpu);
    const auto transfers = runtime::transfer_stats();
    EXPECT_EQ(transfers.host_to_device_calls, 0U);
    EXPECT_EQ(transfers.device_to_host_calls, 0U);
    expect_near(actual.to_vector(), expected.to_vector(), 2.0e-4F);
    enable_attention_gemm_scale_fusion(false);
    EXPECT_FALSE(attention_gemm_scale_fusion_enabled());
    enable_attention_gemm_scale_fusion(true);
    EXPECT_TRUE(attention_gemm_scale_fusion_enabled());
    enable_attention_gemm_scale_fusion(false);
}

TEST(HipOptimizedOpsTest, MatmulTuningCacheRestoresOnlyCurrentEnvironment) {
    require_gpu();
    const auto gpu = Device::hip(0);
    const Tensor left({64, 64}, DType::Float32, gpu);
    const Tensor right({64, 64}, DType::Float32, gpu);
    const auto path = std::filesystem::temp_directory_path() /
                      "microllm-matmul-tuning-cache-hip.jsonl";
    auto key = make_matmul_tuning_key(left, right);
    clear_matmul_implementation_registry();
    register_matmul_implementation(key, MatmulImplementation::HipBLASLt);
    save_matmul_tuning_cache(path);
    clear_matmul_implementation_registry();
    const auto loaded = load_matmul_tuning_cache(path, gpu);
    EXPECT_EQ(loaded.parsed_entries, 1U);
    EXPECT_EQ(loaded.loaded_entries, 1U);
    EXPECT_EQ(loaded.stale_entries, 0U);
    EXPECT_EQ(choose_matmul_implementation(left, right),
              MatmulImplementation::HipBLASLt);

    clear_matmul_implementation_registry();
    ++key.hip_runtime_version;
    register_matmul_implementation(key, MatmulImplementation::HipBLASLt);
    save_matmul_tuning_cache(path);
    clear_matmul_implementation_registry();
    const auto stale = load_matmul_tuning_cache(path, gpu);
    EXPECT_EQ(stale.parsed_entries, 1U);
    EXPECT_EQ(stale.loaded_entries, 0U);
    EXPECT_EQ(stale.stale_entries, 1U);
    EXPECT_EQ(choose_matmul_implementation(left, right),
              MatmulImplementation::Readable);
    clear_matmul_implementation_registry();
    std::error_code ignored;
    std::filesystem::remove(path, ignored);
}

TEST(HipOptimizedOpsTest, MatmulAutotuneChecksCorrectnessBeforeTiming) {
    require_gpu();
    const auto gpu = Device::hip(0);
    std::vector<float> left_values(64 * 64);
    std::vector<float> right_values(64 * 64);
    for (std::size_t index = 0; index < left_values.size(); ++index) {
        left_values[index] =
            static_cast<float>(static_cast<int>(index % 29) - 14) / 29.0F;
        right_values[index] =
            static_cast<float>(static_cast<int>(index % 17) - 8) / 17.0F;
    }
    const auto left = Tensor::from_vector(left_values, {64, 64}).to(gpu);
    const auto right = Tensor::from_vector(right_values, {64, 64}).to(gpu);
    runtime::enable_hip_caching_allocator(gpu);
    ASSERT_TRUE(runtime::hip_caching_allocator_enabled(gpu));
    MatmulAutotuneOptions options;
    options.warmup = 1;
    options.repetitions = 5;
    options.mode = OpMode::Inference;
    const auto report = autotune_matmul(left, right, false, false, options);
    EXPECT_EQ(report.reference_elements, 4096);
    EXPECT_EQ(report.key.mode, OpMode::Inference);
    ASSERT_EQ(report.candidates.size(), 2U);
    for (const auto& candidate : report.candidates) {
        EXPECT_TRUE(candidate.supported) << candidate.failure;
        EXPECT_TRUE(candidate.correctness_passed) << candidate.failure;
        EXPECT_TRUE(candidate.finite);
        EXPECT_GT(candidate.event_ms_p50, 0.0);
        EXPECT_GE(candidate.event_ms_p95, candidate.event_ms_p50);
        EXPECT_GT(candidate.wall_ms_p50, 0.0);
        EXPECT_GE(candidate.wall_ms_p95, candidate.wall_ms_p50);
    }
    EXPECT_NE(report.recommended, MatmulImplementation::Auto);
    EXPECT_TRUE(runtime::hip_caching_allocator_enabled(gpu))
        << "autotune must not create a non-default Stream and disable the pool";
    clear_matmul_implementation_registry();
    register_matmul_autotune_winner(report);
    EXPECT_EQ(matmul_registered_implementation_count(), 1U);
    EXPECT_EQ(choose_matmul_implementation(left, right, false, false,
                                            OpContext{.mode = OpMode::Inference}),
              report.recommended);
    clear_matmul_implementation_registry();

    const auto fp16_left = Tensor::from_vector(
        left_values, {64, 64}, DType::Float16).to(gpu);
    const auto fp16_right = Tensor::from_vector(
        right_values, {64, 64}, DType::Float16).to(gpu);
    MatmulAutotuneOptions exact = options;
    exact.repetitions = 3;
    exact.maximum_absolute_tolerance = 0.0F;
    exact.rms_tolerance = 0.0F;
    const auto strict = autotune_matmul(
        fp16_left, fp16_right, false, false, exact);
    const auto hipblaslt = std::find_if(
        strict.candidates.begin(), strict.candidates.end(),
        [](const auto& candidate) {
            return candidate.implementation == MatmulImplementation::HipBLASLt;
        });
    ASSERT_NE(hipblaslt, strict.candidates.end());
    EXPECT_TRUE(hipblaslt->supported);
    EXPECT_FALSE(hipblaslt->correctness_passed);
    EXPECT_EQ(hipblaslt->event_ms_p50, 0.0)
        << "a failed correctness candidate must never enter timing";
    EXPECT_EQ(strict.recommended, MatmulImplementation::Readable);
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

TEST(HipOptimizedOpsTest, StridedBatchedGemmCoversAttentionLayouts) {
    require_gpu();
    const auto gpu = Device::hip(0);
    constexpr std::int64_t batches = 6;
    constexpr std::int64_t rows = 5;
    constexpr std::int64_t inner = 7;
    constexpr std::int64_t columns = 4;
    std::vector<float> left_values(static_cast<std::size_t>(batches * rows * inner));
    std::vector<float> right_values(static_cast<std::size_t>(batches * inner * columns));
    for (std::size_t index = 0; index < left_values.size(); ++index) {
        left_values[index] = static_cast<float>(static_cast<int>(index % 17) - 8) / 17.0F;
    }
    for (std::size_t index = 0; index < right_values.size(); ++index) {
        right_values[index] = static_cast<float>(static_cast<int>(index % 13) - 6) / 13.0F;
    }
    for (const auto dtype : {DType::Float32, DType::BFloat16}) {
        const auto left_cpu = Tensor::from_vector(
            left_values, {2, 3, rows, inner}, dtype);
        const auto right_cpu = Tensor::from_vector(
            right_values, {2, 3, inner, columns}, dtype);
        const auto expected = matmul(left_cpu, right_cpu).to_vector();
        const auto left = left_cpu.to(gpu);
        const auto right = right_cpu.to(gpu);
        const auto left_t = left_cpu.transpose(-2, -1).contiguous().to(gpu);
        const auto right_t = right_cpu.transpose(-2, -1).contiguous().to(gpu);
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
                *std::get<0>(test), *std::get<1>(test),
                MatmulImplementation::HipBLASLt,
                std::get<2>(test), std::get<3>(test));
            runtime::synchronize(gpu);
            const auto transfers = runtime::transfer_stats();
            EXPECT_EQ(transfers.host_to_device_calls, 0U);
            EXPECT_EQ(transfers.device_to_host_calls, 0U);
            expect_near(actual.to_vector(), expected,
                        dtype == DType::BFloat16 ? 0.2F : 3.0e-4F);
            EXPECT_EQ(actual.shape(), (Shape{2, 3, rows, columns}));
        }
    }
}
#endif

}  // namespace microllm::ops
