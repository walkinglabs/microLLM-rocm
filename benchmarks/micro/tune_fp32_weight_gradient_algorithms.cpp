#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <set>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include <hipblaslt/hipblaslt-ext.hpp>
#include <hipblaslt/hipblaslt.h>
#include <microllm/core/storage.h>
#include <microllm/core/tensor.h>
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
    [[nodiscard]] hipblasLtHandle_t get() const noexcept { return value_; }
private:
    hipblasLtHandle_t value_ = nullptr;
};

class Description {
public:
    Description() {
        check(hipblasLtMatmulDescCreate(
                  &value_, HIPBLAS_COMPUTE_32F, HIP_R_32F),
              "hipblasLtMatmulDescCreate");
        const auto operation_a = HIPBLAS_OP_N;
        const auto operation_b = HIPBLAS_OP_T;
        check(hipblasLtMatmulDescSetAttribute(
                  value_, HIPBLASLT_MATMUL_DESC_TRANSA,
                  &operation_a, sizeof(operation_a)), "set TRANSA");
        check(hipblasLtMatmulDescSetAttribute(
                  value_, HIPBLASLT_MATMUL_DESC_TRANSB,
                  &operation_b, sizeof(operation_b)), "set TRANSB");
    }
    ~Description() { (void)hipblasLtMatmulDescDestroy(value_); }
    [[nodiscard]] hipblasLtMatmulDesc_t get() const noexcept { return value_; }
private:
    hipblasLtMatmulDesc_t value_ = nullptr;
};

class Layout {
public:
    Layout(std::uint64_t rows, std::uint64_t columns,
           std::int64_t leading) {
        check(hipblasLtMatrixLayoutCreate(
                  &value_, HIP_R_32F, rows, columns, leading),
              "hipblasLtMatrixLayoutCreate");
    }
    ~Layout() { (void)hipblasLtMatrixLayoutDestroy(value_); }
    [[nodiscard]] hipblasLtMatrixLayout_t get() const noexcept { return value_; }
private:
    hipblasLtMatrixLayout_t value_ = nullptr;
};

class Preference {
public:
    explicit Preference(std::uint64_t bytes) {
        check(hipblasLtMatmulPreferenceCreate(&value_),
              "hipblasLtMatmulPreferenceCreate");
        check(hipblasLtMatmulPreferenceSetAttribute(
                  value_, HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                  &bytes, sizeof(bytes)), "set workspace preference");
    }
    ~Preference() { (void)hipblasLtMatmulPreferenceDestroy(value_); }
    [[nodiscard]] hipblasLtMatmulPreference_t get() const noexcept {
        return value_;
    }
private:
    hipblasLtMatmulPreference_t value_ = nullptr;
};

struct Options {
    std::string model = "qwen";
    std::string operation = "gate-up";
    std::int64_t rows = 512;
    int maximum_algorithms = 64;
    std::uint64_t workspace_bytes = 64U * 1024U * 1024U;
    int warmup = 3;
    int repetitions = 10;
};

std::int64_t integer(const char* text, const char* name) {
    std::size_t consumed = 0;
    const auto result = std::stoll(text, &consumed);
    if (consumed != std::string_view(text).size()) {
        throw std::invalid_argument(std::string("invalid ") + name);
    }
    return result;
}

Options options(int argc, char** argv) {
    Options result;
    for (int index = 1; index < argc; index += 2) {
        if (index + 1 >= argc) throw std::invalid_argument("missing option value");
        const std::string_view name(argv[index]);
        if (name == "--model") result.model = argv[index + 1];
        else if (name == "--operation") result.operation = argv[index + 1];
        else if (name == "--rows") result.rows = integer(argv[index + 1], "rows");
        else if (name == "--maximum-algorithms") {
            result.maximum_algorithms = static_cast<int>(
                integer(argv[index + 1], "maximum-algorithms"));
        } else if (name == "--workspace-bytes") {
            const auto bytes = integer(argv[index + 1], "workspace-bytes");
            if (bytes < 0) throw std::invalid_argument("workspace must be nonnegative");
            result.workspace_bytes = static_cast<std::uint64_t>(bytes);
        } else if (name == "--warmup") {
            result.warmup = static_cast<int>(integer(argv[index + 1], "warmup"));
        } else if (name == "--repetitions") {
            result.repetitions = static_cast<int>(
                integer(argv[index + 1], "repetitions"));
        } else {
            throw std::invalid_argument("unknown option: " + std::string(name));
        }
    }
    if ((result.model != "qwen" && result.model != "deepseek") ||
        (result.operation != "q" && result.operation != "kv" &&
         result.operation != "gate-up" && result.operation != "down") ||
        result.rows <= 0 || result.rows > 4096 ||
        result.maximum_algorithms <= 0 || result.maximum_algorithms > 256 ||
        result.warmup < 0 || result.repetitions <= 0) {
        throw std::invalid_argument(
            "FP32 weight-gradient algorithm options are invalid");
    }
    return result;
}

struct Problem {
    std::int64_t input = 0;
    std::int64_t output = 0;
};

Problem problem(const Options& options) {
    const auto hidden = options.model == "qwen" ? 896LL : 1536LL;
    const auto kv = options.model == "qwen" ? 128LL : 256LL;
    const auto intermediate = options.model == "qwen" ? 4864LL : 8960LL;
    if (options.operation == "q") return {hidden, hidden};
    if (options.operation == "kv") return {hidden, kv};
    if (options.operation == "gate-up") return {hidden, intermediate};
    return {intermediate, hidden};
}

struct Candidate {
    int index = -1;
    std::size_t workspace_bytes = 0;
    hipblasLtMatmulAlgo_t algorithm{};
};

double percentile(std::vector<double> values, double fraction) {
    std::sort(values.begin(), values.end());
    const auto position = fraction * static_cast<double>(values.size() - 1);
    const auto low = static_cast<std::size_t>(position);
    const auto high = std::min(low + 1, values.size() - 1);
    return values[low] + (values[high] - values[low]) *
                             (position - static_cast<double>(low));
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const auto command = options(argc, argv);
        if (microllm::runtime::hip_device_count() == 0) {
            throw std::runtime_error(
                "FP32 weight-gradient tuner requires a visible HIP GPU");
        }
        const auto shape = problem(command);
        const auto device = microllm::Device::hip(0);
        std::vector<float> input_values(
            static_cast<std::size_t>(command.rows * shape.input));
        std::vector<float> gradient_values(
            static_cast<std::size_t>(command.rows * shape.output));
        for (std::size_t index = 0; index < input_values.size(); ++index) {
            input_values[index] =
                static_cast<float>(static_cast<int>(index % 29) - 14) / 127.0F;
        }
        for (std::size_t index = 0; index < gradient_values.size(); ++index) {
            gradient_values[index] =
                static_cast<float>(static_cast<int>(index % 31) - 15) / 131.0F;
        }
        const auto input = microllm::Tensor::from_vector(
            input_values, {command.rows, shape.input}).to(device);
        const auto gradient = microllm::Tensor::from_vector(
            gradient_values, {command.rows, shape.output}).to(device);
        microllm::Tensor output(
            {shape.input, shape.output}, microllm::DType::Float32, device);
        microllm::Storage workspace(
            static_cast<std::size_t>(command.workspace_bytes), device);
        Handle handle;
        Description description;
        Layout matrix_b(
            static_cast<std::uint64_t>(shape.output),
            static_cast<std::uint64_t>(command.rows), shape.output);
        Layout matrix_a(
            static_cast<std::uint64_t>(shape.input),
            static_cast<std::uint64_t>(command.rows), shape.input);
        Layout matrix_c(
            static_cast<std::uint64_t>(shape.output),
            static_cast<std::uint64_t>(shape.input), shape.output);
        Preference preference(command.workspace_bytes);
        std::vector<hipblasLtMatmulHeuristicResult_t> heuristic(
            static_cast<std::size_t>(command.maximum_algorithms));
        int returned = 0;
        check(hipblasLtMatmulAlgoGetHeuristic(
                  handle.get(), description.get(), matrix_b.get(), matrix_a.get(),
                  matrix_c.get(), matrix_c.get(), preference.get(),
                  command.maximum_algorithms, heuristic.data(), &returned),
              "hipblasLtMatmulAlgoGetHeuristic(weight gradient)");
        std::set<int> seen;
        std::vector<Candidate> candidates;
        for (int position = 0; position < returned; ++position) {
            auto& result = heuristic[static_cast<std::size_t>(position)];
            const auto index = hipblaslt_ext::getIndexFromAlgo(result.algo);
            if (result.state == HIPBLAS_STATUS_SUCCESS && seen.insert(index).second) {
                candidates.push_back({index, result.workspaceSize, result.algo});
            }
        }
        const float alpha = 1.0F;
        const float beta = 0.0F;
        const auto submit = [&](const hipblasLtMatmulAlgo_t* algorithm) {
            check(hipblasLtMatmul(
                      handle.get(), description.get(), &alpha,
                      gradient.data(), matrix_b.get(), input.data(),
                      matrix_a.get(), &beta, output.data(), matrix_c.get(),
                      output.data(), matrix_c.get(), algorithm,
                      workspace.data(), workspace.num_bytes(), nullptr),
                  "hipblasLtMatmul(weight gradient)");
        };
        submit(nullptr);
        microllm::runtime::synchronize(device);
        const auto reference = output.to_vector();
        microllm::runtime::Event start(device);
        microllm::runtime::Event finish(device);
        const auto time = [&](const hipblasLtMatmulAlgo_t* algorithm) {
            for (int iteration = 0; iteration < command.warmup; ++iteration) {
                submit(algorithm);
            }
            microllm::runtime::synchronize(device);
            std::vector<double> event;
            std::vector<double> wall;
            for (int iteration = 0; iteration < command.repetitions; ++iteration) {
                const auto wall_start = std::chrono::steady_clock::now();
                start.record_default_stream();
                submit(algorithm);
                finish.record_default_stream();
                finish.synchronize();
                const auto wall_finish = std::chrono::steady_clock::now();
                event.push_back(finish.elapsed_ms_since(start));
                wall.push_back(std::chrono::duration<double, std::milli>(
                    wall_finish - wall_start).count());
            }
            return std::vector<double>{
                percentile(event, 0.5), percentile(event, 0.95),
                percentile(wall, 0.5), percentile(wall, 0.95)};
        };
        const auto default_time = time(nullptr);
        int passing = 0;
        int keep = 0;
        int recommended = -1;
        double best_event = default_time[0];
        std::cout << std::setprecision(12)
                  << "{\"schema_version\":1,\"status\":\"pass\""
                  << ",\"record_type\":\"fp32_weight_gradient_algorithm_tune\""
                  << ",\"model\":\"" << command.model << "\""
                  << ",\"operation\":\"" << command.operation << "\""
                  << ",\"rows\":" << command.rows
                  << ",\"input\":" << shape.input
                  << ",\"output\":" << shape.output
                  << ",\"m\":" << shape.input
                  << ",\"k\":" << command.rows
                  << ",\"n\":" << shape.output
                  << ",\"transpose_left\":true"
                  << ",\"workspace_limit_bytes\":" << command.workspace_bytes
                  << ",\"candidate_count\":" << candidates.size()
                  << ",\"complete_output_elements\":" << reference.size()
                  << ",\"default_event_ms_p50\":" << default_time[0]
                  << ",\"default_event_ms_p95\":" << default_time[1]
                  << ",\"default_wall_ms_p50\":" << default_time[2]
                  << ",\"default_wall_ms_p95\":" << default_time[3]
                  << ",\"candidates\":[";
        for (std::size_t position = 0; position < candidates.size(); ++position) {
            const auto& candidate = candidates[position];
            bool finite = true;
            bool correctness = false;
            float maximum = 0.0F;
            double rms = 0.0;
            std::vector<double> timing{0, 0, 0, 0};
            std::string failure;
            try {
                submit(&candidate.algorithm);
                microllm::runtime::synchronize(device);
                const auto actual = output.to_vector();
                double squared = 0.0;
                for (std::size_t index = 0; index < actual.size(); ++index) {
                    finite = finite && std::isfinite(actual[index]);
                    const auto difference = std::abs(actual[index] - reference[index]);
                    maximum = std::max(maximum, difference);
                    squared += static_cast<double>(difference) * difference;
                }
                rms = std::sqrt(squared / static_cast<double>(actual.size()));
                correctness = finite && maximum <= 1.0e-4F && rms <= 1.0e-5;
                if (correctness) {
                    ++passing;
                    timing = time(&candidate.algorithm);
                    if (default_time[0] / timing[0] >= 1.05) ++keep;
                    if (timing[0] < best_event) {
                        best_event = timing[0];
                        recommended = candidate.index;
                    }
                } else {
                    failure = "complete-output gate failed";
                }
            } catch (const std::exception& error) {
                failure = error.what();
            }
            if (position != 0) std::cout << ',';
            std::cout << "{\"index\":" << candidate.index
                      << ",\"workspace_bytes\":" << candidate.workspace_bytes
                      << ",\"finite\":" << (finite ? "true" : "false")
                      << ",\"correctness_passed\":"
                      << (correctness ? "true" : "false")
                      << ",\"maximum_absolute_error\":" << maximum
                      << ",\"rms_error\":" << rms
                      << ",\"event_ms_p50\":" << timing[0]
                      << ",\"event_ms_p95\":" << timing[1]
                      << ",\"wall_ms_p50\":" << timing[2]
                      << ",\"wall_ms_p95\":" << timing[3]
                      << ",\"event_speedup_vs_default\":"
                      << (timing[0] > 0.0 ? default_time[0] / timing[0] : 0.0)
                      << ",\"failure\":\"" << failure << "\"}";
        }
        std::cout << "],\"passing_candidates\":" << passing
                  << ",\"keep_candidates\":" << keep
                  << ",\"recommended_index\":" << recommended
                  << ",\"recommended_event_speedup\":"
                  << (recommended >= 0 ? default_time[0] / best_event : 1.0)
                  << "}\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "tune_fp32_weight_gradient_algorithms: "
                  << error.what() << '\n';
        return 1;
    }
}
