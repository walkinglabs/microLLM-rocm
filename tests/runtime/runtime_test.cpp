#include <array>
#include <stdexcept>
#include <utility>
#include <vector>

#include <gtest/gtest.h>
#include <microllm/core/tensor.h>
#include <microllm/ops/low_level.h>
#include <microllm/ops/ops.h>
#include <microllm/runtime/memory.h>
#include <microllm/runtime/runtime.h>
#include <microllm/runtime/diagnostics.h>

namespace microllm::runtime {

TEST(RuntimeTest, CpuStreamAndEventHaveSynchronousSemantics) {
    Stream stream(Device::cpu());
    Event event(Device::cpu());
    EXPECT_FALSE(event.ready());
    event.record(stream);
    EXPECT_TRUE(event.ready());
    event.wait(stream);
    stream.synchronize();
    EXPECT_THROW(event.record_default_stream(), std::runtime_error);
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

TEST(RuntimeTest, StridedCopyDiagnosticsAggregateExactLayoutOnlyWhenEnabled) {
    reset_strided_copy_diagnostics();
    enable_strided_copy_diagnostics(true);
    const auto input = Tensor::from_vector({1, 2, 3, 4, 5, 6}, {2, 3});
    const auto view = input.transpose(0, 1);
    (void)view.contiguous();
    (void)view.contiguous();
    enable_strided_copy_diagnostics(false);
    const auto snapshot = strided_copy_diagnostics();
    ASSERT_EQ(snapshot.records.size(), 1U);
    EXPECT_EQ(snapshot.calls, 2U);
    EXPECT_EQ(snapshot.elements, 12U);
    EXPECT_EQ(snapshot.bytes, 48U);
    EXPECT_EQ(snapshot.records[0].shape, (std::vector<std::int64_t>{3, 2}));
    EXPECT_EQ(snapshot.records[0].strides, (std::vector<std::int64_t>{1, 3}));
    EXPECT_EQ(snapshot.records[0].device, Device::cpu());
    (void)view.contiguous();
    EXPECT_EQ(strided_copy_diagnostics().calls, 2U)
        << "disabled diagnostics must have no hot-path side effects";
    reset_strided_copy_diagnostics();
}

TEST(HipGraphTest, CpuAndUndefinedContractsAreExplicit) {
    HipGraphExecutable graph;
    EXPECT_FALSE(graph.defined());
    EXPECT_THROW((void)graph.device(), std::logic_error);
    EXPECT_THROW((void)graph.node_count(), std::logic_error);
    const Stream cpu_stream(Device::cpu());
    EXPECT_THROW(graph.launch(cpu_stream), std::logic_error);
    EXPECT_THROW(
        (void)HipGraphExecutable::capture(cpu_stream, [] {}),
        std::runtime_error);
    EXPECT_THROW(
        (void)HipGraphExecutable::capture(cpu_stream, {}),
        std::invalid_argument);
}

TEST(DeferredHipDeallocationTest, CpuStreamAndZeroCapacityAreRejected) {
    const Stream cpu_stream(Device::cpu());
    EXPECT_THROW((void)DeferredHipDeallocationScope(cpu_stream),
                 std::invalid_argument);
}

#if MICROLLM_HAS_HIP
TEST(HipGraphTest, CapturesReplaysAndMovesCallerOwnedOperatorChain) {
    if (hip_device_count() == 0) GTEST_SKIP() << "No visible HIP device";
    const auto gpu = Device::hip(0);
    auto left = Tensor::from_vector({1, 2, 3, 4}, {2, 2}).to(gpu);
    const auto right = Tensor::from_vector({5, 6, 7, 8}, {2, 2}).to(gpu);
    Tensor sum({2, 2}, DType::Float32, gpu);
    Tensor product({2, 2}, DType::Float32, gpu);
    Stream stream(gpu);
    ops::OpContext context;
    context.stream = &stream;

    EXPECT_THROW((void)HipGraphExecutable::capture(stream, [] {}),
                 std::invalid_argument);
    EXPECT_THROW(
        (void)HipGraphExecutable::capture(
            stream, [] { throw std::runtime_error("intentional capture failure"); }),
        std::runtime_error);
    EXPECT_THROW(
        (void)HipGraphExecutable::capture(stream, [&] {
            Tensor forbidden_allocation({4}, DType::Float32, gpu);
            (void)forbidden_allocation;
        }),
        std::runtime_error);

    const auto capture_work = [&] {
        ops::add_out(sum.view(), std::as_const(left).view(), right.view(), context);
        ops::multiply_out(product.view(), std::as_const(sum).view(),
                          right.view(), context);
    };
    auto graph = HipGraphExecutable::capture(stream, capture_work);
    ASSERT_TRUE(graph.defined());
    EXPECT_EQ(graph.device(), gpu);
    EXPECT_EQ(graph.node_count(), 2U);

    reset_transfer_stats();
    graph.launch(stream);
    stream.synchronize();
    auto transfers = transfer_stats();
    EXPECT_EQ(transfers.host_to_device_calls, 0U);
    EXPECT_EQ(transfers.device_to_host_calls, 0U);
    EXPECT_EQ(transfers.device_to_device_calls, 0U);
    EXPECT_EQ(product.to_vector(), (std::vector<float>{30, 48, 70, 96}));

    ops::fill_(left, 2.0F, context);
    stream.synchronize();
    auto moved = std::move(graph);
    EXPECT_FALSE(graph.defined());
    EXPECT_TRUE(moved.defined());
    if (hip_device_count() > 1) {
        Stream other_device_stream(Device::hip(1));
        EXPECT_THROW(moved.launch(other_device_stream), std::invalid_argument);
    }
    moved.launch(stream);
    stream.synchronize();
    EXPECT_EQ(product.to_vector(), (std::vector<float>{35, 48, 63, 80}));
}

TEST(DeferredHipDeallocationTest, KeepsTemporaryChainAliveUntilOneStreamSync) {
    if (hip_device_count() == 0) GTEST_SKIP() << "No visible HIP device";
    const auto gpu = Device::hip(0);
    const auto input = Tensor::from_vector({0, 0, 0, 0}, {4}).to(gpu);
    const auto source = Tensor::from_vector({1, 1, 1, 1}, {4}).to(gpu);
    Stream stream(gpu);
    ops::OpContext context;
    context.stream = &stream;
    Tensor result;
    reset_transfer_stats();
    EXPECT_THROW((void)DeferredHipDeallocationScope(stream, 0),
                 std::invalid_argument);
    DeferredHipDeallocationScope scope(stream, 16);
    EXPECT_THROW((void)DeferredHipDeallocationScope(stream, 16),
                 std::logic_error);
    auto current = input;
    for (int iteration = 0; iteration < 8; ++iteration) {
        current = ops::add(current, source, context);
    }
    result = std::move(current);
    EXPECT_EQ(scope.pending_blocks(), 7U);
    EXPECT_EQ(scope.pending_bytes(), 7U * 4U * sizeof(float));
    EXPECT_EQ(scope.total_deferred_blocks(), 7U);
    EXPECT_EQ(scope.overflow_flushes(), 0U);
    scope.finish();
    EXPECT_TRUE(scope.finished());
    EXPECT_EQ(scope.pending_blocks(), 0U);
    const auto transfers = transfer_stats();
    EXPECT_EQ(transfers.host_to_device_calls, 0U);
    EXPECT_EQ(transfers.device_to_host_calls, 0U);
    EXPECT_EQ(transfers.device_to_device_calls, 0U);
    EXPECT_EQ(result.to_vector(), (std::vector<float>{8, 8, 8, 8}));
}

TEST(DeferredHipDeallocationTest, CapacityOverflowFlushesSafelyAndContinues) {
    if (hip_device_count() == 0) GTEST_SKIP() << "No visible HIP device";
    const auto gpu = Device::hip(0);
    const auto input = Tensor::from_vector({0, 0}, {2}).to(gpu);
    const auto source = Tensor::from_vector({1, 1}, {2}).to(gpu);
    Stream stream(gpu);
    ops::OpContext context;
    context.stream = &stream;
    Tensor result;
    DeferredHipDeallocationScope scope(stream, 2);
    auto current = input;
    for (int iteration = 0; iteration < 7; ++iteration) {
        current = ops::add(current, source, context);
    }
    result = std::move(current);
    EXPECT_EQ(scope.total_deferred_blocks(), 6U);
    EXPECT_EQ(scope.overflow_flushes(), 2U);
    scope.finish();
    EXPECT_EQ(result.to_vector(), (std::vector<float>{7, 7}));
}

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
