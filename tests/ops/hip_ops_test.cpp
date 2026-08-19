#include <vector>

#include <gtest/gtest.h>
#include <microllm/ops/ops.h>
#include <microllm/runtime/runtime.h>

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

}  // namespace microllm::ops
