#include <gtest/gtest.h>
#include <microllm/autograd/autograd.h>
#include <microllm/profiling/trace.h>
#include <microllm/runtime/runtime.h>

namespace microllm::profiling {

TEST(HipTraceTest, CapturesDeviceOperatorValuesAndSynchronizedDuration) {
    if (runtime::hip_device_count() == 0) GTEST_SKIP() << "No visible HIP device";
    const auto gpu = Device::hip();
    autograd::Value left(Tensor::from_vector({1, 2, 3, 4}, {2, 2}).to(gpu), true);
    autograd::Value right(Tensor::from_vector({4, 3, 2, 1}, {2, 2}).to(gpu), true);
    TraceOptions options;
    options.phase = "values";
    options.record_layers = false;
    options.record_model = false;
    TraceSession session("microllm", "hip-unit", options);
    {
        ScopedTraceSession active(session);
        const auto output = autograd::add(left, right);
        EXPECT_EQ(output.data().to_vector(), (std::vector<float>{5, 5, 5, 5}));
    }
    ASSERT_EQ(session.records().size(), 1U);
    EXPECT_EQ(session.records()[0].kind, TraceKind::Operator);
    EXPECT_EQ(session.records()[0].name, "add");
    EXPECT_EQ(session.records()[0].device, gpu);
    EXPECT_EQ(session.records()[0].values, (std::vector<double>{5, 5, 5, 5}));
    EXPECT_GE(session.records()[0].wall_ms, 0.0);
}

}  // namespace microllm::profiling
