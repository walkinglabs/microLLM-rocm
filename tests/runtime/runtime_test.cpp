#include <array>
#include <stdexcept>
#include <vector>

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

TEST(RuntimeTest, TracksCurrentPeakAndTotalEngineAllocations) {
    const auto before = allocation_stats(Device::cpu()).current_bytes;
    reset_allocation_peak(Device::cpu());
    {
        Storage first(64);
        Storage second(32);
        const auto during = allocation_stats(Device::cpu());
        EXPECT_EQ(during.current_bytes, before + 96);
        EXPECT_GE(during.peak_bytes, before + 96);
        EXPECT_EQ(during.total_allocated_bytes, 96U);
        EXPECT_EQ(during.allocation_calls, 2U);
        EXPECT_EQ(during.deallocation_calls, 0U);
        EXPECT_EQ(during.backend_allocation_calls, 2U);
    }
    const auto after = allocation_stats(Device::cpu());
    EXPECT_EQ(after.current_bytes, before);
    EXPECT_EQ(after.allocation_calls, 2U);
    EXPECT_EQ(after.deallocation_calls, 2U);
    EXPECT_EQ(after.backend_deallocation_calls, 2U);
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

TEST(HipRuntimeTest, TracksMeasuredAllocationAndDeallocationCalls) {
    if (hip_device_count() == 0) GTEST_SKIP() << "No visible HIP device";
    const auto gpu = Device::hip(0);
    const auto before = allocation_stats(gpu).current_bytes;
    reset_allocation_peak(gpu);
    {
        Storage first(256, gpu);
        Storage second(512, gpu);
        const auto during = allocation_stats(gpu);
        EXPECT_EQ(during.current_bytes, before + 768);
        EXPECT_EQ(during.total_allocated_bytes, 768U);
        EXPECT_EQ(during.allocation_calls, 2U);
        EXPECT_EQ(during.deallocation_calls, 0U);
        EXPECT_EQ(during.backend_allocation_calls, 2U);
    }
    const auto after = allocation_stats(gpu);
    EXPECT_EQ(after.current_bytes, before);
    EXPECT_EQ(after.allocation_calls, 2U);
    EXPECT_EQ(after.deallocation_calls, 2U);
    EXPECT_EQ(after.backend_deallocation_calls, 2U);
    EXPECT_EQ(after.cached_bytes, 0U);
}

TEST(HipRuntimeTest, ExactSizePoolReusesCompletedDefaultStreamBlock) {
    if (hip_device_count() == 0) GTEST_SKIP() << "No visible HIP device";
    const auto gpu = Device::hip(0);
    const auto baseline = allocation_stats(gpu);
    enable_hip_caching_allocator(gpu);
    ASSERT_TRUE(hip_caching_allocator_enabled(gpu));
    reset_allocation_peak(gpu);
    {
        Storage first(4096, gpu);
    }
    {
        Storage second(4096, gpu);
    }
    const auto stats = allocation_stats(gpu);
    EXPECT_EQ(stats.allocation_calls, 2U);
    EXPECT_EQ(stats.deallocation_calls, 2U);
    EXPECT_EQ(stats.backend_allocation_calls, 1U);
    EXPECT_EQ(stats.cache_reuse_calls, 1U);
    EXPECT_EQ(stats.backend_deallocation_calls, 0U);
    EXPECT_EQ(stats.cached_bytes, baseline.cached_bytes + 4096U);
    EXPECT_EQ(stats.reserved_bytes, baseline.reserved_bytes + 4096U);
}

TEST(HipRuntimeTest, DefaultStreamPoolReusesEveryExactSizeWithoutBatchPhase) {
    if (hip_device_count() == 0) GTEST_SKIP() << "No visible HIP device";
    constexpr std::size_t kBytes = 4352;
    const auto gpu = Device::hip(0);
    enable_hip_caching_allocator(gpu);
    reset_allocation_peak(gpu);
    {
        std::vector<Storage> blocks;
        for (int index = 0; index < 16; ++index) blocks.emplace_back(kBytes, gpu);
    }
    {
        std::vector<Storage> blocks;
        for (int index = 0; index < 16; ++index) blocks.emplace_back(kBytes, gpu);
        const auto stats = allocation_stats(gpu);
        EXPECT_EQ(stats.allocation_calls, 32U);
        EXPECT_EQ(stats.backend_allocation_calls, 16U);
        EXPECT_EQ(stats.cache_reuse_calls, 16U);
    }
}

TEST(HipRuntimeTest, NonDefaultStreamPermanentlyDisablesPoolReuse) {
    if (hip_device_count() == 0) GTEST_SKIP() << "No visible HIP device";
    const auto gpu = Device::hip(0);
    const auto baseline = allocation_stats(gpu);
    enable_hip_caching_allocator(gpu);
    Stream stream(gpu);
    EXPECT_FALSE(hip_caching_allocator_enabled(gpu));
    reset_allocation_peak(gpu);
    {
        Storage first(4096, gpu);
    }
    {
        Storage second(4096, gpu);
    }
    const auto stats = allocation_stats(gpu);
    EXPECT_EQ(stats.allocation_calls, 2U);
    EXPECT_EQ(stats.deallocation_calls, 2U);
    EXPECT_EQ(stats.backend_allocation_calls, 2U);
    EXPECT_EQ(stats.backend_deallocation_calls, 2U);
    EXPECT_EQ(stats.cache_reuse_calls, 0U);
    EXPECT_EQ(stats.cached_bytes, baseline.cached_bytes);
    EXPECT_EQ(stats.reserved_bytes, baseline.reserved_bytes);
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
