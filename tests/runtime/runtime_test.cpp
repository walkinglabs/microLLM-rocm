#include <array>
#include <stdexcept>

#include <gtest/gtest.h>
#include <microllm/core/tensor.h>
#include <microllm/runtime/memory.h>
#include <microllm/runtime/runtime.h>

namespace microllm::runtime {

TEST(RuntimeTest, CpuStreamAndEventHaveSynchronousSemantics) {
    Stream stream(Device::cpu());
    Event event(Device::cpu());
    EXPECT_FALSE(event.ready());
    event.record(stream);
    EXPECT_TRUE(event.ready());
    event.wait(stream);
    stream.synchronize();
}

TEST(RuntimeTest, CpuCopyRoundTripsBytes) {
    const std::array<int, 4> input{1, 2, 3, 4};
    std::array<int, 4> output{};
    copy_bytes(output.data(), Device::cpu(), input.data(), Device::cpu(), sizeof(input));
    EXPECT_EQ(output, input);
}

#if MICROLLM_HAS_HIP
TEST(HipRuntimeTest, ReportsDeviceAndTransfersTensor) {
    if (hip_device_count() == 0) GTEST_SKIP() << "No visible HIP device";
    const auto gpu = Device::hip(0);
    const auto info = device_info(gpu);
    EXPECT_FALSE(info.name.empty());
    EXPECT_FALSE(info.architecture.empty());
    EXPECT_GT(info.total_memory, 0U);

    const auto cpu = Tensor::from_vector({1, 2, 3, 4, 5, 6}, {2, 3});
    const auto device_tensor = cpu.to(gpu);
    EXPECT_TRUE(device_tensor.device().is_hip());
    EXPECT_EQ(device_tensor.to_vector(), cpu.to_vector());
}

TEST(HipRuntimeTest, AsyncCopyAndEventsRespectDependencies) {
    if (hip_device_count() == 0) GTEST_SKIP() << "No visible HIP device";
    const auto gpu = Device::hip(0);
    Stream stream(gpu);
    Event start(gpu);
    Event finish(gpu);
    Storage storage(sizeof(int) * 4, gpu);
    const std::array<int, 4> input{10, 20, 30, 40};
    std::array<int, 4> output{};

    start.record(stream);
    copy_bytes_async(storage.data(), gpu, input.data(), Device::cpu(), sizeof(input), stream);
    copy_bytes_async(output.data(), Device::cpu(), storage.data(), gpu, sizeof(output), stream);
    finish.record(stream);
    finish.synchronize();

    EXPECT_EQ(output, input);
    EXPECT_TRUE(finish.ready());
    EXPECT_GE(finish.elapsed_ms_since(start), 0.0F);
}
#else
TEST(HipRuntimeTest, CpuBuildRejectsHipObjects) {
    EXPECT_FALSE(hip_compiled());
    EXPECT_EQ(hip_device_count(), 0);
    EXPECT_THROW((void)Stream(Device::hip()), std::runtime_error);
    EXPECT_THROW((void)Event(Device::hip()), std::runtime_error);
}
#endif

}  // namespace microllm::runtime
