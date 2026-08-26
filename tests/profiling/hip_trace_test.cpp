#include <filesystem>
#include <fstream>
#include <vector>

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
    options.value_name_filters = {"add"};
    const auto binary_directory = std::filesystem::temp_directory_path() /
                                  "microllm-hip-trace-binary-test";
    std::error_code ignored;
    std::filesystem::remove_all(binary_directory, ignored);
    options.binary_value_directory = binary_directory;
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
    EXPECT_EQ(session.records()[0].binary_values_dtype, "float32");
    EXPECT_EQ(session.records()[0].binary_values_byte_order, "little");
    EXPECT_EQ(session.records()[0].binary_values_count, 4U);
    const auto binary_path =
        binary_directory / session.records()[0].binary_values_file;
    std::ifstream binary(binary_path, std::ios::binary);
    std::vector<float> binary_values(4);
    binary.read(reinterpret_cast<char*>(binary_values.data()),
                static_cast<std::streamsize>(binary_values.size() * sizeof(float)));
    EXPECT_EQ(binary.gcount(),
              static_cast<std::streamsize>(binary_values.size() * sizeof(float)));
    EXPECT_EQ(binary_values, (std::vector<float>{5, 5, 5, 5}));
    std::filesystem::remove_all(binary_directory, ignored);
}

}  // namespace microllm::profiling
