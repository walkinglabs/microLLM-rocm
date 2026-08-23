#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include <hipblaslt/hipblaslt-ext.hpp>
#include <hipblaslt/hipblaslt.h>
#include <microllm/ops/ops.h>
#include <microllm/runtime/runtime.h>

namespace {

void check(hipblasStatus_t status, const char* operation) {
    if (status != HIPBLAS_STATUS_SUCCESS) {
        throw std::runtime_error(std::string(operation) + " failed: " +
                                 std::to_string(static_cast<int>(status)));
    }
}

class Handle {
public:
    Handle() { check(hipblasLtCreate(&value_), "hipblasLtCreate"); }
    ~Handle() { (void)hipblasLtDestroy(value_); }
    hipblasLtHandle_t get() const noexcept { return value_; }
private:
    hipblasLtHandle_t value_ = nullptr;
};

class Description {
public:
    Description() {
        check(hipblasLtMatmulDescCreate(&value_, HIPBLAS_COMPUTE_32F, HIP_R_32F),
              "hipblasLtMatmulDescCreate");
    }
    ~Description() { (void)hipblasLtMatmulDescDestroy(value_); }
    hipblasLtMatmulDesc_t get() const noexcept { return value_; }
private:
    hipblasLtMatmulDesc_t value_ = nullptr;
};

class Layout {
public:
    Layout(hipDataType dtype, std::uint64_t rows, std::uint64_t columns,
           std::int64_t leading) {
        check(hipblasLtMatrixLayoutCreate(&value_, dtype, rows, columns, leading),
              "hipblasLtMatrixLayoutCreate");
    }
    ~Layout() { (void)hipblasLtMatrixLayoutDestroy(value_); }
    hipblasLtMatrixLayout_t get() const noexcept { return value_; }
private:
    hipblasLtMatrixLayout_t value_ = nullptr;
};

class Preference {
public:
    explicit Preference(std::uint64_t workspace_bytes) {
        check(hipblasLtMatmulPreferenceCreate(&value_),
              "hipblasLtMatmulPreferenceCreate");
        check(hipblasLtMatmulPreferenceSetAttribute(
                  value_, HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                  &workspace_bytes, sizeof(workspace_bytes)),
              "hipblasLtMatmulPreferenceSetAttribute");
    }
    ~Preference() { (void)hipblasLtMatmulPreferenceDestroy(value_); }
    hipblasLtMatmulPreference_t get() const noexcept { return value_; }
private:
    hipblasLtMatmulPreference_t value_ = nullptr;
};

struct Options {
    std::int64_t rows = 512;
    std::int64_t inner = 896;
    std::int64_t columns = 896;
    std::string output_dtype = "fp32";
    int maximum_algorithms = 64;
    std::uint64_t workspace_bytes = 32U * 1024U * 1024U;
    int warmup = 2;
    int repetitions = 5;
};

std::int64_t integer(const char* value, const char* name) {
    char* end = nullptr;
    const auto parsed = std::strtoll(value, &end, 10);
    if (end == value || *end != '\0') {
        throw std::invalid_argument(std::string("invalid ") + name);
    }
    return parsed;
}

Options options(int argc, char** argv) {
    Options result;
    for (int index = 1; index < argc; index += 2) {
        if (index + 1 >= argc) throw std::invalid_argument("missing option value");
        const std::string_view name(argv[index]);
        if (name == "--rows") result.rows = integer(argv[index + 1], "rows");
        else if (name == "--inner") result.inner = integer(argv[index + 1], "inner");
        else if (name == "--columns") result.columns = integer(argv[index + 1], "columns");
        else if (name == "--output-dtype") result.output_dtype = argv[index + 1];
        else if (name == "--maximum-algorithms") {
            result.maximum_algorithms =
                static_cast<int>(integer(argv[index + 1], "maximum-algorithms"));
        } else if (name == "--workspace-bytes") {
            const auto value = integer(argv[index + 1], "workspace-bytes");
            if (value < 0) throw std::invalid_argument("workspace must be nonnegative");
            result.workspace_bytes = static_cast<std::uint64_t>(value);
        } else if (name == "--warmup") {
            result.warmup = static_cast<int>(integer(argv[index + 1], "warmup"));
        } else if (name == "--repetitions") {
            result.repetitions = static_cast<int>(integer(argv[index + 1], "repetitions"));
        } else {
            throw std::invalid_argument("unknown option: " + std::string(name));
        }
    }
    if (result.rows <= 0 || result.inner <= 0 || result.columns <= 0 ||
        result.rows > 16384 || result.inner > 16384 || result.columns > 200000 ||
        result.maximum_algorithms <= 0 || result.maximum_algorithms > 256 ||
        result.warmup < 0 || result.repetitions <= 0 ||
        (result.output_dtype != "fp32" && result.output_dtype != "bf16")) {
        throw std::invalid_argument("BF16 tune options are outside the safe contract");
    }
    return result;
}

microllm::DType output_dtype(const std::string& name) {
    return name == "bf16" ? microllm::DType::BFloat16 : microllm::DType::Float32;
}

struct InventoryCandidate {
    int index = -1;
    std::size_t workspace_bytes = 0;
};

std::vector<InventoryCandidate> inventory(Handle& handle, const Options& options) {
    Description operation;
    Layout matrix_b(HIP_R_16BF, static_cast<std::uint64_t>(options.columns),
                    static_cast<std::uint64_t>(options.inner), options.columns);
    Layout matrix_a(HIP_R_16BF, static_cast<std::uint64_t>(options.inner),
                    static_cast<std::uint64_t>(options.rows), options.inner);
    const auto output_type = options.output_dtype == "bf16" ? HIP_R_16BF : HIP_R_32F;
    Layout matrix_c(output_type, static_cast<std::uint64_t>(options.columns),
                    static_cast<std::uint64_t>(options.rows), options.columns);
    Preference preference(options.workspace_bytes);
    std::vector<hipblasLtMatmulHeuristicResult_t> results(
        static_cast<std::size_t>(options.maximum_algorithms));
    int returned = 0;
    check(hipblasLtMatmulAlgoGetHeuristic(
              handle.get(), operation.get(), matrix_b.get(), matrix_a.get(),
              matrix_c.get(), matrix_c.get(), preference.get(),
              options.maximum_algorithms, results.data(), &returned),
          "hipblasLtMatmulAlgoGetHeuristic");
    std::vector<InventoryCandidate> candidates;
    for (int position = 0; position < returned; ++position) {
        auto& result = results[static_cast<std::size_t>(position)];
        if (result.state != HIPBLAS_STATUS_SUCCESS) continue;
        candidates.push_back({hipblaslt_ext::getIndexFromAlgo(result.algo),
                              result.workspaceSize});
    }
    return candidates;
}

double percentile(std::vector<double> values, double fraction) {
    std::sort(values.begin(), values.end());
    const auto index = static_cast<std::size_t>(
        std::ceil(fraction * static_cast<double>(values.size()))) - 1U;
    return values[std::min(index, values.size() - 1U)];
}

struct RegistryGuard {
    ~RegistryGuard() { microllm::ops::clear_bf16_algorithm_registry(); }
};

}  // namespace

int main(int argc, char** argv) {
    try {
        const auto command = options(argc, argv);
        if (microllm::runtime::hip_device_count() == 0 ||
            !microllm::ops::hipblaslt_available()) {
            throw std::runtime_error("BF16 algorithm tune requires HIP and hipBLASLt");
        }
        RegistryGuard cleanup;
        const auto device = microllm::Device::hip(0);
        std::vector<float> left_values(
            static_cast<std::size_t>(command.rows * command.inner));
        std::vector<float> right_values(
            static_cast<std::size_t>(command.inner * command.columns));
        for (std::size_t index = 0; index < left_values.size(); ++index) {
            left_values[index] =
                static_cast<float>(static_cast<int>(index % 29) - 14) / 29.0F;
        }
        for (std::size_t index = 0; index < right_values.size(); ++index) {
            right_values[index] =
                static_cast<float>(static_cast<int>(index % 17) - 8) / 17.0F;
        }
        const auto left = microllm::Tensor::from_vector(
            left_values, {command.rows, command.inner},
            microllm::DType::BFloat16).to(device);
        const auto right = microllm::Tensor::from_vector(
            right_values, {command.inner, command.columns},
            microllm::DType::BFloat16).to(device);
        const auto dtype = output_dtype(command.output_dtype);
        microllm::ops::clear_bf16_algorithm_registry();
        const auto reference_tensor = microllm::ops::bf16_matmul_output(
            left, right, dtype);
        microllm::runtime::synchronize(device);
        const auto reference = reference_tensor.to_vector();
        Handle handle;
        const auto candidates = inventory(handle, command);
        microllm::runtime::Event start(device);
        microllm::runtime::Event finish(device);
        microllm::Tensor default_output;
        for (int iteration = 0; iteration < command.warmup; ++iteration) {
            default_output = microllm::ops::bf16_matmul_output(left, right, dtype);
        }
        microllm::runtime::synchronize(device);
        std::vector<double> default_event_times;
        std::vector<double> default_wall_times;
        for (int iteration = 0; iteration < command.repetitions; ++iteration) {
            const auto wall_start = std::chrono::steady_clock::now();
            start.record_default_stream();
            default_output = microllm::ops::bf16_matmul_output(left, right, dtype);
            finish.record_default_stream();
            finish.synchronize();
            const auto wall_finish = std::chrono::steady_clock::now();
            default_event_times.push_back(finish.elapsed_ms_since(start));
            default_wall_times.push_back(std::chrono::duration<double, std::milli>(
                wall_finish - wall_start).count());
        }
        const auto default_event_p50 = percentile(default_event_times, 0.50);
        const auto default_event_p95 = percentile(default_event_times, 0.95);
        const auto default_wall_p50 = percentile(default_wall_times, 0.50);
        const auto default_wall_p95 = percentile(default_wall_times, 0.95);
        int passing = 0;
        int recommended = -1;
        double best = 0.0;
        std::cout << std::setprecision(9)
                  << "{\"schema_version\":1,\"status\":\"pass\""
                  << ",\"track\":\"bf16_solution_correctness_before_timing\""
                  << ",\"rows\":" << command.rows
                  << ",\"inner\":" << command.inner
                  << ",\"columns\":" << command.columns
                  << ",\"output_dtype\":\"" << command.output_dtype << "\""
                  << ",\"workspace_limit_bytes\":" << command.workspace_bytes
                  << ",\"candidate_count\":" << candidates.size()
                  << ",\"complete_output_elements\":" << reference.size()
                  << ",\"default_event_ms_p50\":" << default_event_p50
                  << ",\"default_event_ms_p95\":" << default_event_p95
                  << ",\"default_wall_ms_p50\":" << default_wall_p50
                  << ",\"default_wall_ms_p95\":" << default_wall_p95
                  << ",\"candidates\":[";
        bool first_record = true;
        for (const auto& candidate : candidates) {
            bool supported = false;
            bool correctness = false;
            bool finite = true;
            float maximum = 0.0F;
            double rms = 0.0;
            double event_p50 = 0.0;
            double event_p95 = 0.0;
            double wall_p50 = 0.0;
            double wall_p95 = 0.0;
            std::string failure;
            try {
                microllm::ops::clear_bf16_algorithm_registry();
                microllm::ops::register_bf16_algorithm(
                    command.rows, command.inner, command.columns, dtype,
                    candidate.index);
                auto checked = microllm::ops::bf16_matmul_output(left, right, dtype);
                microllm::runtime::synchronize(device);
                const auto actual = checked.to_vector();
                double squared = 0.0;
                for (std::size_t index = 0; index < actual.size(); ++index) {
                    finite = finite && std::isfinite(actual[index]);
                    const auto difference = std::abs(actual[index] - reference[index]);
                    maximum = std::max(maximum, difference);
                    squared += static_cast<double>(difference) * difference;
                }
                rms = std::sqrt(squared / static_cast<double>(actual.size()));
                supported = true;
                correctness = finite && maximum <= 1.0e-4F && rms <= 1.0e-5;
                if (correctness) {
                    ++passing;
                    for (int iteration = 0; iteration < command.warmup; ++iteration) {
                        checked = microllm::ops::bf16_matmul_output(left, right, dtype);
                    }
                    microllm::runtime::synchronize(device);
                    std::vector<double> times;
                    std::vector<double> wall_times;
                    for (int iteration = 0; iteration < command.repetitions; ++iteration) {
                        const auto wall_start = std::chrono::steady_clock::now();
                        start.record_default_stream();
                        checked = microllm::ops::bf16_matmul_output(left, right, dtype);
                        finish.record_default_stream();
                        finish.synchronize();
                        const auto wall_finish = std::chrono::steady_clock::now();
                        times.push_back(finish.elapsed_ms_since(start));
                        wall_times.push_back(std::chrono::duration<double, std::milli>(
                            wall_finish - wall_start).count());
                    }
                    event_p50 = percentile(times, 0.50);
                    event_p95 = percentile(times, 0.95);
                    wall_p50 = percentile(wall_times, 0.50);
                    wall_p95 = percentile(wall_times, 0.95);
                    if (recommended < 0 || event_p50 < best) {
                        recommended = candidate.index;
                        best = event_p50;
                    }
                } else {
                    failure = "complete-output correctness gate failed";
                }
            } catch (const std::exception& error) {
                failure = error.what();
            }
            if (!first_record) std::cout << ',';
            first_record = false;
            std::cout << "{\"index\":" << candidate.index
                      << ",\"workspace_bytes\":" << candidate.workspace_bytes
                      << ",\"supported\":" << (supported ? "true" : "false")
                      << ",\"correctness_passed\":"
                      << (correctness ? "true" : "false")
                      << ",\"finite\":" << (finite ? "true" : "false")
                      << ",\"maximum_absolute_error\":" << maximum
                      << ",\"rms_error\":" << rms
                      << ",\"event_ms_p50\":" << event_p50
                      << ",\"event_ms_p95\":" << event_p95
                      << ",\"wall_ms_p50\":" << wall_p50
                      << ",\"wall_ms_p95\":" << wall_p95
                      << ",\"failure\":\"" << failure << "\"}";
        }
        microllm::ops::clear_bf16_algorithm_registry();
        std::cout << "],\"passing_candidates\":" << passing
                  << ",\"recommended_index\":" << recommended
                  << ",\"recommended_event_ms_p50\":" << best
                  << ",\"registry_entries_after_screening\":0}\n";
        return passing > 0 ? 0 : 1;
    } catch (const std::exception& error) {
        std::cerr << "tune_bf16_algorithms: " << error.what() << '\n';
        return 2;
    }
}
