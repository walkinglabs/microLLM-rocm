#pragma once

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

#include <microllm/core/tensor.h>

namespace microllm::profiling {

enum class TraceKind { Operator, Layer, Model, Parameter, Input, Output };

struct TraceOptions {
    std::string phase = "values";
    bool record_operators = true;
    bool record_layers = true;
    bool record_model = true;
    bool capture_values = true;
    bool synchronize_device = true;
    bool record_all_layer_details = false;
    std::vector<std::string> value_name_filters;
    std::size_t max_captured_elements = 4096;
};

struct TensorStatistics {
    std::int64_t numel = 0;
    std::int64_t finite_count = 0;
    double minimum = 0.0;
    double maximum = 0.0;
    double mean = 0.0;
    double l2_norm = 0.0;
};

struct TraceRecord {
    std::uint64_t sequence = 0;
    std::uint64_t iteration = 0;
    TraceKind kind = TraceKind::Operator;
    std::string name;
    Shape shape;
    DType dtype = DType::Float32;
    Device device = Device::cpu();
    double wall_ms = 0.0;
    TensorStatistics statistics;
    std::vector<double> values;
    bool values_truncated = false;
};

class TraceSession {
public:
    TraceSession(std::string framework, std::string run_id, TraceOptions options = {});

    void set_iteration(std::uint64_t iteration) noexcept;
    [[nodiscard]] std::uint64_t iteration() const noexcept;
    [[nodiscard]] const std::string& framework() const noexcept;
    [[nodiscard]] const std::string& run_id() const noexcept;
    [[nodiscard]] const TraceOptions& options() const noexcept;
    [[nodiscard]] const std::vector<TraceRecord>& records() const noexcept;
    [[nodiscard]] bool enabled(TraceKind kind) const noexcept;

    void record(TraceKind kind, std::string name, const Tensor& tensor,
                double wall_ms = 0.0);
    void write_jsonl(const std::filesystem::path& path) const;

    [[nodiscard]] static TraceSession* current() noexcept;

private:
    friend class ScopedTraceSession;
    static void set_current(TraceSession* session) noexcept;

    std::string framework_;
    std::string run_id_;
    TraceOptions options_;
    std::uint64_t iteration_ = 0;
    std::uint64_t next_sequence_ = 0;
    std::vector<TraceRecord> records_;
};

class ScopedTraceSession {
public:
    explicit ScopedTraceSession(TraceSession& session);
    ~ScopedTraceSession();
    ScopedTraceSession(const ScopedTraceSession&) = delete;
    ScopedTraceSession& operator=(const ScopedTraceSession&) = delete;

private:
    TraceSession* previous_ = nullptr;
};

class TraceTimer {
public:
    TraceTimer(TraceKind kind, std::string name, Device device);
    void finish(const Tensor& output);
    [[nodiscard]] bool enabled() const noexcept;

private:
    TraceSession* session_ = nullptr;
    TraceKind kind_ = TraceKind::Operator;
    std::string name_;
    Device device_ = Device::cpu();
    std::chrono::steady_clock::time_point start_{};
    bool finished_ = false;
};

[[nodiscard]] const char* trace_kind_name(TraceKind kind) noexcept;

}  // namespace microllm::profiling
