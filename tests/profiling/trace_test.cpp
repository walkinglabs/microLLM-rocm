#include <filesystem>
#include <fstream>
#include <limits>
#include <string>

#include <gtest/gtest.h>
#include <microllm/profiling/trace.h>

namespace microllm::profiling {

TEST(TraceSessionTest, RecordsTensorMetadataStatisticsAndTruncatedValues) {
    TraceOptions options;
    options.max_captured_elements = 2;
    TraceSession session("microllm", "unit", options);
    session.set_iteration(7);
    session.record(TraceKind::Operator, "add",
                   Tensor::from_vector({-2, 0, 3, 4}, {2, 2}), 0.25);
    session.record(TraceKind::Layer, "bf16",
                   Tensor::from_vector({1.25F, -0.5F, 3.0F}, {1, 3},
                                       DType::BFloat16));
    ASSERT_EQ(session.records().size(), 2U);
    const auto& record = session.records()[0];
    EXPECT_EQ(record.sequence, 0U);
    EXPECT_EQ(record.iteration, 7U);
    EXPECT_EQ(record.kind, TraceKind::Operator);
    EXPECT_EQ(record.name, "add");
    EXPECT_EQ(record.shape, (Shape{2, 2}));
    EXPECT_EQ(record.dtype, DType::Float32);
    EXPECT_EQ(record.device, Device::cpu());
    EXPECT_DOUBLE_EQ(record.wall_ms, 0.25);
    EXPECT_EQ(record.statistics.numel, 4);
    EXPECT_EQ(record.statistics.finite_count, 4);
    EXPECT_DOUBLE_EQ(record.statistics.minimum, -2.0);
    EXPECT_DOUBLE_EQ(record.statistics.maximum, 4.0);
    EXPECT_DOUBLE_EQ(record.statistics.mean, 1.25);
    EXPECT_TRUE(record.values_truncated);
    EXPECT_EQ(record.values, (std::vector<double>{-2, 0}));
    const auto& low_precision = session.records()[1];
    EXPECT_EQ(low_precision.dtype, DType::BFloat16);
    EXPECT_EQ(low_precision.statistics.finite_count, 3);
    EXPECT_TRUE(low_precision.values_truncated);
    EXPECT_EQ(low_precision.values, (std::vector<double>{1.25, -0.5}));
}

TEST(TraceSessionTest, ScopedActivationRestoresPreviousSession) {
    TraceSession outer("microllm", "outer");
    TraceSession inner("microllm", "inner");
    EXPECT_EQ(TraceSession::current(), nullptr);
    {
        ScopedTraceSession outer_scope(outer);
        EXPECT_EQ(TraceSession::current(), &outer);
        {
            ScopedTraceSession inner_scope(inner);
            EXPECT_EQ(TraceSession::current(), &inner);
        }
        EXPECT_EQ(TraceSession::current(), &outer);
    }
    EXPECT_EQ(TraceSession::current(), nullptr);
}

TEST(TraceSessionTest, TimerIsZeroOverheadWhenInactiveAndRecordsWhenActive) {
    const auto tensor = Tensor::from_vector({1, 2, 3}, {3});
    TraceTimer inactive(TraceKind::Operator, "inactive", Device::cpu());
    EXPECT_FALSE(inactive.enabled());
    inactive.finish(tensor);

    TraceOptions options;
    options.capture_values = false;
    TraceSession session("microllm", "timing", options);
    {
        ScopedTraceSession active(session);
        TraceTimer timer(TraceKind::Operator, "scale", Device::cpu());
        EXPECT_TRUE(timer.enabled());
        timer.finish(tensor);
    }
    ASSERT_EQ(session.records().size(), 1U);
    EXPECT_GE(session.records()[0].wall_ms, 0.0);
    EXPECT_TRUE(session.records()[0].values.empty());
    EXPECT_EQ(session.records()[0].statistics.numel, 3);
}

TEST(TraceSessionTest, WritesSchemaVersionedJsonLines) {
    TraceSession session("microllm", "json");
    session.record(TraceKind::Input, "tokens",
                   Tensor::from_int32_vector({1, 2}, {1, 2}));
    const auto path = std::filesystem::temp_directory_path() / "microllm-trace-test.jsonl";
    session.write_jsonl(path);
    std::ifstream input(path);
    std::string line;
    std::getline(input, line);
    EXPECT_NE(line.find("\"schema_version\":1"), std::string::npos);
    EXPECT_NE(line.find("\"framework\":\"microllm\""), std::string::npos);
    EXPECT_NE(line.find("\"name\":\"tokens\""), std::string::npos);
    EXPECT_NE(line.find("\"values\":[1,2]"), std::string::npos);
    std::error_code ignored;
    std::filesystem::remove(path, ignored);
}

TEST(TraceSessionTest, RejectsInvalidIdentityAndDuration) {
    EXPECT_THROW((void)TraceSession("", "run"), std::invalid_argument);
    EXPECT_THROW((void)TraceSession("microllm", ""), std::invalid_argument);
    TraceSession session("microllm", "errors");
    const auto tensor = Tensor::from_vector({1}, {1});
    EXPECT_THROW(session.record(TraceKind::Operator, "", tensor), std::invalid_argument);
    EXPECT_THROW(session.record(TraceKind::Operator, "bad", tensor, -1.0),
                 std::invalid_argument);
}

TEST(TraceSessionTest, SerializesNonFiniteValuesAsExplicitJsonStrings) {
    TraceSession session("microllm", "non-finite");
    session.record(
        TraceKind::Output, "bad-values",
        Tensor::from_vector({std::numeric_limits<float>::quiet_NaN(),
                             std::numeric_limits<float>::infinity(),
                             -std::numeric_limits<float>::infinity(), 2.0F},
                            {4}));
    ASSERT_EQ(session.records().size(), 1U);
    EXPECT_EQ(session.records()[0].statistics.finite_count, 1);
    const auto path = std::filesystem::temp_directory_path() /
                      "microllm-trace-non-finite-test.jsonl";
    session.write_jsonl(path);
    std::ifstream input(path);
    std::string line;
    std::getline(input, line);
    EXPECT_NE(line.find("\"values\":[\"nan\",\"inf\",\"-inf\",2]"),
              std::string::npos);
    std::error_code ignored;
    std::filesystem::remove(path, ignored);
}

TEST(TraceSessionTest, ValueFiltersKeepMetadataButCaptureOnlyMatchingNames) {
    TraceOptions options;
    options.value_name_filters = {"ffn.activated"};
    TraceSession session("microllm", "filtered", options);
    const auto tensor = Tensor::from_vector({-2.0F, 3.0F}, {2});
    session.record(TraceKind::Layer, "blocks.0.attention_norm", tensor);
    session.record(TraceKind::Layer, "blocks.0.ffn.activated", tensor);
    ASSERT_EQ(session.records().size(), 2U);
    EXPECT_TRUE(session.records()[0].values.empty());
    EXPECT_EQ(session.records()[0].statistics.finite_count, 0);
    EXPECT_EQ(session.records()[0].statistics.numel, 2);
    EXPECT_EQ(session.records()[1].values.size(), 2U);
    EXPECT_EQ(session.records()[1].statistics.finite_count, 2);
    EXPECT_DOUBLE_EQ(session.records()[1].statistics.minimum, -2.0);
    EXPECT_DOUBLE_EQ(session.records()[1].statistics.maximum, 3.0);
    options.value_name_filters = {""};
    EXPECT_THROW((void)TraceSession("microllm", "bad-filter", options),
                 std::invalid_argument);
}

}  // namespace microllm::profiling
