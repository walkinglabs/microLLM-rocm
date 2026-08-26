#include <microllm/profiling/trace.h>

#include <algorithm>
#include <bit>
#include <cctype>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <limits>
#include <stdexcept>
#include <string_view>

#if defined(__unix__) || defined(__APPLE__)
#include <dlfcn.h>
#endif

#include <microllm/runtime/runtime.h>

namespace microllm::profiling {
namespace {

thread_local TraceSession* active_session = nullptr;

struct RoctxFunctions {
    using Push = int (*)(const char*);
    using Pop = int (*)();
    void* library = nullptr;
    Push push = nullptr;
    Pop pop = nullptr;
};

RoctxFunctions& roctx_functions() {
    static RoctxFunctions functions = [] {
        RoctxFunctions value;
#if defined(__unix__) || defined(__APPLE__)
        for (const auto* name : {"librocprofiler-sdk-roctx.so", "libroctx64.so"}) {
            value.library = ::dlopen(name, RTLD_LAZY | RTLD_LOCAL);
            if (value.library != nullptr) break;
        }
        if (value.library != nullptr) {
            value.push = reinterpret_cast<RoctxFunctions::Push>(
                ::dlsym(value.library, "roctxRangePushA"));
            value.pop = reinterpret_cast<RoctxFunctions::Pop>(
                ::dlsym(value.library, "roctxRangePop"));
            if (value.push == nullptr || value.pop == nullptr) {
                (void)::dlclose(value.library);
                value = {};
            }
        }
#endif
        return value;
    }();
    return functions;
}

std::string escape_json(std::string_view value) {
    std::string output;
    output.reserve(value.size() + 2);
    output.push_back('"');
    for (const auto character : value) {
        switch (character) {
            case '"': output += "\\\""; break;
            case '\\': output += "\\\\"; break;
            case '\n': output += "\\n"; break;
            case '\r': output += "\\r"; break;
            case '\t': output += "\\t"; break;
            default:
                if (static_cast<unsigned char>(character) < 0x20U) {
                    output += '?';
                } else {
                    output.push_back(character);
                }
        }
    }
    output.push_back('"');
    return output;
}

template <typename T>
TensorStatistics statistics(const std::vector<T>& values,
                            std::int64_t declared_numel) {
    TensorStatistics output;
    output.numel = declared_numel;
    if (values.empty()) return output;
    output.minimum = std::numeric_limits<double>::infinity();
    output.maximum = -std::numeric_limits<double>::infinity();
    double sum = 0.0;
    double squared_sum = 0.0;
    for (const auto value : values) {
        const auto numeric = static_cast<double>(value);
        if (!std::isfinite(numeric)) continue;
        ++output.finite_count;
        output.minimum = std::min(output.minimum, numeric);
        output.maximum = std::max(output.maximum, numeric);
        sum += numeric;
        squared_sum += numeric * numeric;
    }
    if (output.finite_count == 0) {
        output.minimum = 0.0;
        output.maximum = 0.0;
        return output;
    }
    output.mean = sum / static_cast<double>(output.finite_count);
    output.l2_norm = std::sqrt(squared_sum);
    return output;
}

std::string safe_file_component(std::string_view value) {
    std::string output;
    output.reserve(value.size());
    for (const auto character : value) {
        const auto byte = static_cast<unsigned char>(character);
        output.push_back(std::isalnum(byte) || character == '-' ||
                                 character == '_' || character == '.'
                             ? character
                             : '_');
    }
    return output.empty() ? "trace" : output;
}

template <typename T>
std::string write_binary_values(
    const std::filesystem::path& directory, std::string_view run_id,
    std::uint64_t sequence, std::string_view name, std::string_view dtype,
    const std::vector<T>& values) {
    std::filesystem::create_directories(directory);
    const auto filename = safe_file_component(run_id) + "-" +
        std::to_string(sequence) + "-" + safe_file_component(name) + "." +
        std::string(dtype) + ".bin";
    const auto path = directory / filename;
    std::ofstream output(path, std::ios::binary | std::ios::trunc);
    if (!output) {
        throw std::runtime_error(
            "cannot open trace binary output: " + path.string());
    }
    if (!values.empty()) {
        output.write(
            reinterpret_cast<const char*>(values.data()),
            static_cast<std::streamsize>(values.size() * sizeof(T)));
    }
    if (!output) {
        throw std::runtime_error(
            "failed while writing trace binary output: " + path.string());
    }
    return filename;
}

}  // namespace

const char* trace_kind_name(TraceKind kind) noexcept {
    switch (kind) {
        case TraceKind::Operator: return "operator";
        case TraceKind::Layer: return "layer";
        case TraceKind::Model: return "model";
        case TraceKind::Parameter: return "parameter";
        case TraceKind::Input: return "input";
        case TraceKind::Output: return "output";
    }
    return "unknown";
}

TraceSession::TraceSession(std::string framework, std::string run_id,
                           TraceOptions options)
    : framework_(std::move(framework)),
      run_id_(std::move(run_id)),
      options_(std::move(options)) {
    if (framework_.empty()) throw std::invalid_argument("trace framework cannot be empty");
    if (run_id_.empty()) throw std::invalid_argument("trace run ID cannot be empty");
    if (options_.phase.empty()) throw std::invalid_argument("trace phase cannot be empty");
    if (std::any_of(options_.value_name_filters.begin(),
                    options_.value_name_filters.end(),
                    [](const auto& value) { return value.empty(); })) {
        throw std::invalid_argument("trace value filters cannot be empty");
    }
    if (!options_.binary_value_directory.empty() &&
        !options_.capture_values) {
        throw std::invalid_argument(
            "trace binary output requires value capture");
    }
    if (!options_.binary_value_directory.empty() &&
        std::endian::native != std::endian::little) {
        throw std::invalid_argument(
            "trace binary output currently requires a little-endian host");
    }
    if (!options_.binary_value_directory.empty() &&
        std::filesystem::exists(options_.binary_value_directory) &&
        !std::filesystem::is_directory(options_.binary_value_directory)) {
        throw std::invalid_argument(
            "trace binary output path must be a directory");
    }
}

void TraceSession::set_iteration(std::uint64_t iteration) noexcept { iteration_ = iteration; }
std::uint64_t TraceSession::iteration() const noexcept { return iteration_; }
const std::string& TraceSession::framework() const noexcept { return framework_; }
const std::string& TraceSession::run_id() const noexcept { return run_id_; }
const TraceOptions& TraceSession::options() const noexcept { return options_; }
const std::vector<TraceRecord>& TraceSession::records() const noexcept { return records_; }

bool TraceSession::enabled(TraceKind kind) const noexcept {
    switch (kind) {
        case TraceKind::Operator: return options_.record_operators;
        case TraceKind::Layer: return options_.record_layers;
        case TraceKind::Model: return options_.record_model;
        case TraceKind::Parameter:
        case TraceKind::Input:
        case TraceKind::Output: return options_.capture_values;
    }
    return false;
}

void TraceSession::record(TraceKind kind, std::string name, const Tensor& tensor,
                          double wall_ms) {
    if (!enabled(kind)) return;
    if (name.empty()) throw std::invalid_argument("trace record name cannot be empty");
    if (!tensor.defined()) throw std::invalid_argument("trace tensor must be defined");
    if (wall_ms < 0.0 || !std::isfinite(wall_ms)) {
        throw std::invalid_argument("trace duration must be finite and non-negative");
    }
    TraceRecord record;
    record.sequence = next_sequence_++;
    record.iteration = iteration_;
    record.kind = kind;
    const auto capture_record_values = options_.capture_values &&
        (options_.value_name_filters.empty() || std::any_of(
            options_.value_name_filters.begin(), options_.value_name_filters.end(),
            [&](const auto& filter) { return name.find(filter) != std::string::npos; }));
    record.name = std::move(name);
    record.shape = tensor.shape();
    record.dtype = tensor.dtype();
    record.device = tensor.device();
    record.wall_ms = wall_ms;
    if (capture_record_values) {
        if (is_floating_point(tensor.dtype())) {
            const auto values = tensor.to_vector();
            record.statistics = statistics(values, tensor.numel());
            const auto captured =
                std::min(values.size(), options_.max_captured_elements);
            record.values.reserve(captured);
            for (std::size_t index = 0; index < captured; ++index) {
                record.values.push_back(static_cast<double>(values[index]));
            }
            record.values_truncated = captured != values.size();
            if (!options_.binary_value_directory.empty()) {
                record.binary_values_file = write_binary_values(
                    options_.binary_value_directory, run_id_, record.sequence,
                    record.name, "f32", values);
                record.binary_values_dtype = "float32";
                record.binary_values_byte_order = "little";
                record.binary_values_count = values.size();
                record.binary_values_bytes = values.size() * sizeof(float);
            }
        } else if (tensor.dtype() == DType::Int32) {
            const auto values = tensor.to_int32_vector();
            record.statistics = statistics(values, tensor.numel());
            const auto captured =
                std::min(values.size(), options_.max_captured_elements);
            record.values.reserve(captured);
            for (std::size_t index = 0; index < captured; ++index) {
                record.values.push_back(static_cast<double>(values[index]));
            }
            record.values_truncated = captured != values.size();
            if (!options_.binary_value_directory.empty()) {
                record.binary_values_file = write_binary_values(
                    options_.binary_value_directory, run_id_, record.sequence,
                    record.name, "i32", values);
                record.binary_values_dtype = "int32";
                record.binary_values_byte_order = "little";
                record.binary_values_count = values.size();
                record.binary_values_bytes =
                    values.size() * sizeof(std::int32_t);
            }
        } else {
            throw std::invalid_argument(
                "trace value capture does not support this tensor dtype");
        }
    } else {
        record.statistics.numel = tensor.numel();
    }
    records_.push_back(std::move(record));
}

void TraceSession::write_jsonl(const std::filesystem::path& path) const {
    if (path.has_parent_path()) std::filesystem::create_directories(path.parent_path());
    std::ofstream output(path, std::ios::trunc);
    if (!output) throw std::runtime_error("cannot open trace output: " + path.string());
    output << std::setprecision(17);
    for (const auto& record : records_) {
        output << "{\"schema_version\":1,\"framework\":" << escape_json(framework_)
               << ",\"run_id\":" << escape_json(run_id_)
               << ",\"phase\":" << escape_json(options_.phase)
               << ",\"sequence\":" << record.sequence
               << ",\"iteration\":" << record.iteration
               << ",\"kind\":" << escape_json(trace_kind_name(record.kind))
               << ",\"name\":" << escape_json(record.name)
               << ",\"shape\":[";
        for (std::size_t index = 0; index < record.shape.size(); ++index) {
            if (index != 0) output << ',';
            output << record.shape[index];
        }
        output << "],\"dtype\":" << escape_json(dtype_name(record.dtype))
               << ",\"device\":" << escape_json(record.device.str())
               << ",\"wall_ms\":" << record.wall_ms
               << ",\"statistics\":{\"numel\":" << record.statistics.numel
               << ",\"finite_count\":" << record.statistics.finite_count
               << ",\"minimum\":" << record.statistics.minimum
               << ",\"maximum\":" << record.statistics.maximum
               << ",\"mean\":" << record.statistics.mean
               << ",\"l2_norm\":" << record.statistics.l2_norm
               << "},\"values_truncated\":"
               << (record.values_truncated ? "true" : "false")
               << ",\"binary_values\":";
        if (record.binary_values_file.empty()) {
            output << "null";
        } else {
            output << "{\"file\":" << escape_json(record.binary_values_file)
                   << ",\"dtype\":" << escape_json(record.binary_values_dtype)
                   << ",\"byte_order\":"
                   << escape_json(record.binary_values_byte_order)
                   << ",\"count\":" << record.binary_values_count
                   << ",\"bytes\":" << record.binary_values_bytes << '}';
        }
        output << ",\"values\":[";
        for (std::size_t index = 0; index < record.values.size(); ++index) {
            if (index != 0) output << ',';
            const auto value = record.values[index];
            if (std::isnan(value)) output << "\"nan\"";
            else if (value == std::numeric_limits<double>::infinity()) output << "\"inf\"";
            else if (value == -std::numeric_limits<double>::infinity()) output << "\"-inf\"";
            else output << value;
        }
        output << "]}\n";
    }
    if (!output) throw std::runtime_error("failed while writing trace output");
}

TraceSession* TraceSession::current() noexcept { return active_session; }
void TraceSession::set_current(TraceSession* session) noexcept { active_session = session; }

ScopedTraceSession::ScopedTraceSession(TraceSession& session)
    : previous_(TraceSession::current()) {
    TraceSession::set_current(&session);
}

ScopedTraceSession::~ScopedTraceSession() { TraceSession::set_current(previous_); }

TraceTimer::TraceTimer(TraceKind kind, std::string name, Device device)
    : session_(TraceSession::current()),
      kind_(kind),
      name_(std::move(name)),
      device_(device) {
    if (session_ == nullptr || !session_->enabled(kind_)) {
        session_ = nullptr;
        return;
    }
    if (session_->options().synchronize_device) runtime::synchronize(device_);
    if (session_->options().emit_roctx_ranges) {
        auto& roctx = roctx_functions();
        if (roctx.push != nullptr) {
            (void)roctx.push(name_.c_str());
            roctx_pushed_ = true;
        }
    }
    start_ = std::chrono::steady_clock::now();
}

TraceTimer::~TraceTimer() {
    if (roctx_pushed_) {
        (void)roctx_functions().pop();
        roctx_pushed_ = false;
    }
}

void TraceTimer::finish(const Tensor& output) {
    if (session_ == nullptr || finished_) return;
    if (session_->options().synchronize_device) runtime::synchronize(device_);
    const auto finish_time = std::chrono::steady_clock::now();
    const auto elapsed = std::chrono::duration<double, std::milli>(finish_time - start_).count();
    if (roctx_pushed_) {
        (void)roctx_functions().pop();
        roctx_pushed_ = false;
    }
    session_->record(kind_, std::move(name_), output, elapsed);
    finished_ = true;
}

bool TraceTimer::enabled() const noexcept { return session_ != nullptr; }

bool roctx_markers_available() noexcept {
    const auto& functions = roctx_functions();
    return functions.push != nullptr && functions.pop != nullptr;
}

}  // namespace microllm::profiling
