#include <vector>

#include <gtest/gtest.h>
#include <microllm/ops/ops.h>
#include <microllm/ops/low_level.h>
#include <microllm/runtime/runtime.h>
#include <microllm/model/model.h>
#include <microllm/training/trainer.h>

namespace microllm::ops {
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

TEST(HipOpsTest, RopeAndCrossEntropyMatchCpuReference) {
    require_gpu();
    const auto rope_input = Tensor::from_vector({1, 0, 0, 1, 1, 0, 0, 1}, {1, 2, 1, 4});
    expect_near(rope(rope_input.to(Device::hip())).to_vector(), rope(rope_input).to_vector());

    const auto logits_cpu = Tensor::from_vector({2, 1, 0, 0, 1, 2}, {2, 3});
    const auto targets_cpu = Tensor::from_int32_vector({0, 2}, {2});
    expect_near(cross_entropy(logits_cpu.to(Device::hip()), targets_cpu.to(Device::hip())).to_vector(),
                cross_entropy(logits_cpu, targets_cpu).to_vector());
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

TEST(HipModelTest, TinyOneTokenCachedForwardMatchesCpu) {
    require_gpu();
    const model::ModelConfig config{.vocabulary_size = 16,
                                    .dimension = 8,
                                    .layers = 1,
                                    .heads = 2,
                                    .kv_heads = 2,
                                    .ffn_dimension = 16,
                                    .max_sequence_length = 4,
                                    .rope_base = 10000.0F,
                                    .tie_embeddings = false};
    const auto token = Tensor::from_int32_vector({3}, {1, 1});
    model::TransformerModel cpu_model(config, 61);
    inference::KVCache cpu_cache(config.layers, config.max_sequence_length);
    const auto expected = cpu_model.forward_cached(token, cpu_cache).to_vector();
    model::TransformerModel hip_model(config, 61);
    hip_model.to(Device::hip());
    inference::KVCache hip_cache(config.layers, config.max_sequence_length);
    expect_near(hip_model.forward_cached(token, hip_cache).to_vector(), expected, 2.0e-4F);
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
    EXPECT_EQ(choose_matmul_implementation(left_cpu.to(Device::hip()),
                                           right_cpu.to(Device::hip())),
              MatmulImplementation::Readable);
    const Tensor large_left({128, 128}, DType::Float32, Device::hip());
    const Tensor large_right({128, 128}, DType::Float32, Device::hip());
    EXPECT_EQ(choose_matmul_implementation(large_left, large_right),
              MatmulImplementation::HipBLASLt);
}
#endif

}  // namespace microllm::ops
