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
        throw std::runtime_error(
            std::string(operation) + " failed: " +
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
    Description(bool transpose_left, bool transpose_right) {
        check(hipblasLtMatmulDescCreate(
                  &value_, HIPBLAS_COMPUTE_32F, HIP_R_32F),
              "hipblasLtMatmulDescCreate");
        const auto operation_a = transpose_right ? HIPBLAS_OP_T : HIPBLAS_OP_N;
        const auto operation_b = transpose_left ? HIPBLAS_OP_T : HIPBLAS_OP_N;
        check(hipblasLtMatmulDescSetAttribute(
                  value_, HIPBLASLT_MATMUL_DESC_TRANSA,
                  &operation_a, sizeof(operation_a)),
              "set TRANSA");
        check(hipblasLtMatmulDescSetAttribute(
                  value_, HIPBLASLT_MATMUL_DESC_TRANSB,
                  &operation_b, sizeof(operation_b)),
              "set TRANSB");
    }
    ~Description() { (void)hipblasLtMatmulDescDestroy(value_); }
    hipblasLtMatmulDesc_t get() const noexcept { return value_; }
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
    void set_batch(std::int32_t count, std::int64_t stride) {
        check(hipblasLtMatrixLayoutSetAttribute(
                  value_, HIPBLASLT_MATRIX_LAYOUT_BATCH_COUNT,
                  &count, sizeof(count)), "set batch count");
        check(hipblasLtMatrixLayoutSetAttribute(
                  value_, HIPBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET,
                  &stride, sizeof(stride)), "set batch stride");
    }
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
              "set workspace preference");
    }
    ~Preference() { (void)hipblasLtMatmulPreferenceDestroy(value_); }
    hipblasLtMatmulPreference_t get() const noexcept { return value_; }
private:
    hipblasLtMatmulPreference_t value_ = nullptr;
};

struct Options {
    std::string model = "qwen";
    std::string operation = "qk";
    std::int64_t sequence = 512;
    int maximum_algorithms = 64;
    std::uint64_t workspace_bytes = 32U * 1024U * 1024U;
    int warmup = 2;
    int repetitions = 5;
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
        else if (name == "--sequence") {
            result.sequence = integer(argv[index + 1], "sequence");
        } else if (name == "--maximum-algorithms") {
            result.maximum_algorithms = static_cast<int>(
                integer(argv[index + 1], "maximum-algorithms"));
        } else if (name == "--workspace-bytes") {
            const auto value = integer(argv[index + 1], "workspace-bytes");
            if (value < 0) throw std::invalid_argument("workspace must be nonnegative");
            result.workspace_bytes = static_cast<std::uint64_t>(value);
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
        (result.operation != "qk" && result.operation != "pv") ||
        result.sequence <= 0 || result.sequence > 4096 ||
        result.maximum_algorithms <= 0 || result.maximum_algorithms > 256 ||
        result.warmup < 0 || result.repetitions <= 0) {
        throw std::invalid_argument("attention algorithm options are invalid");
    }
    return result;
}

struct Problem {
    std::int64_t batches = 0;
    std::int64_t left_rows = 0;
    std::int64_t left_columns = 0;
    std::int64_t right_rows = 0;
    std::int64_t right_columns = 0;
    std::int64_t output_columns = 0;
    bool transpose_right = false;
};

Problem problem(const Options& options) {
    const auto heads = options.model == "qwen" ? 14LL : 12LL;
    const auto width = options.model == "qwen" ? 64LL : 128LL;
    if (options.operation == "qk") {
        return {heads, options.sequence, width,
                options.sequence, width, options.sequence, true};
    }
    return {heads, options.sequence, options.sequence,
            options.sequence, width, width, false};
}

struct Candidate {
    int index = -1;
    std::size_t workspace_bytes = 0;
    hipblasLtMatmulAlgo_t algorithm{};
};

struct Plan {
    Plan(const Problem& problem, std::uint64_t workspace_bytes)
        : operation(false, problem.transpose_right),
          matrix_b(static_cast<std::uint64_t>(problem.right_columns),
                   static_cast<std::uint64_t>(problem.right_rows),
                   problem.right_columns),
          matrix_a(static_cast<std::uint64_t>(problem.left_columns),
                   static_cast<std::uint64_t>(problem.left_rows),
                   problem.left_columns),
          matrix_c(static_cast<std::uint64_t>(problem.output_columns),
                   static_cast<std::uint64_t>(problem.left_rows),
                   problem.output_columns),
          preference(workspace_bytes) {
        const auto count = static_cast<std::int32_t>(problem.batches);
        matrix_b.set_batch(
            count, problem.right_rows * problem.right_columns);
        matrix_a.set_batch(
            count, problem.left_rows * problem.left_columns);
        matrix_c.set_batch(
            count, problem.left_rows * problem.output_columns);
    }
    Description operation;
    Layout matrix_b;
    Layout matrix_a;
    Layout matrix_c;
    Preference preference;
};

std::vector<Candidate> inventory(
    Handle& handle, Plan& plan, int maximum) {
    std::vector<hipblasLtMatmulHeuristicResult_t> results(
        static_cast<std::size_t>(maximum));
    int returned = 0;
    check(hipblasLtMatmulAlgoGetHeuristic(
              handle.get(), plan.operation.get(), plan.matrix_b.get(),
              plan.matrix_a.get(), plan.matrix_c.get(), plan.matrix_c.get(),
              plan.preference.get(), maximum, results.data(), &returned),
          "hipblasLtMatmulAlgoGetHeuristic");
    std::set<int> seen;
    std::vector<Candidate> candidates;
    for (int position = 0; position < returned; ++position) {
        auto& result = results[static_cast<std::size_t>(position)];
        const auto index = hipblaslt_ext::getIndexFromAlgo(result.algo);
        if (result.state == HIPBLAS_STATUS_SUCCESS && seen.insert(index).second) {
            candidates.push_back(
                {index, result.workspaceSize, result.algo});
        }
    }
    return candidates;
}

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
            throw std::runtime_error("attention tune requires a visible HIP GPU");
        }
        const auto shape = problem(command);
        const auto device = microllm::Device::hip(0);
        const auto left_elements = static_cast<std::size_t>(
            shape.batches * shape.left_rows * shape.left_columns);
        const auto right_elements = static_cast<std::size_t>(
            shape.batches * shape.right_rows * shape.right_columns);
        const auto output_elements = static_cast<std::size_t>(
            shape.batches * shape.left_rows * shape.output_columns);
        std::vector<float> left_values(left_elements);
        std::vector<float> right_values(right_elements);
        for (std::size_t index = 0; index < left_values.size(); ++index) {
            left_values[index] =
                static_cast<float>(static_cast<int>(index % 29) - 14) / 127.0F;
        }
        for (std::size_t index = 0; index < right_values.size(); ++index) {
            right_values[index] =
                static_cast<float>(static_cast<int>(index % 31) - 15) / 131.0F;
        }
        const auto left = microllm::Tensor::from_vector(
            left_values,
            {shape.batches, shape.left_rows, shape.left_columns}).to(device);
        const auto right = microllm::Tensor::from_vector(
            right_values,
            {shape.batches, shape.right_rows, shape.right_columns}).to(device);
        microllm::Tensor output(
            {shape.batches, shape.left_rows, shape.output_columns},
            microllm::DType::Float32, device);
        microllm::Storage workspace(
            static_cast<std::size_t>(command.workspace_bytes), device);
        Handle handle;
        Plan plan(shape, command.workspace_bytes);
        const auto candidates = inventory(
            handle, plan, command.maximum_algorithms);
        const float alpha = 1.0F;
        const float beta = 0.0F;
        const auto submit = [&](const hipblasLtMatmulAlgo_t* algorithm) {
            check(hipblasLtMatmul(
                      handle.get(), plan.operation.get(), &alpha,
                      right.data(), plan.matrix_b.get(), left.data(),
                      plan.matrix_a.get(), &beta, output.data(),
                      plan.matrix_c.get(), output.data(), plan.matrix_c.get(),
                      algorithm, workspace.data(), workspace.num_bytes(), nullptr),
                  "hipblasLtMatmul");
        };
        submit(nullptr);
        microllm::runtime::synchronize(device);
        const auto reference = output.to_vector();
        if (reference.size() != output_elements) {
            throw std::logic_error("attention output element count changed");
        }
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
                  << ",\"record_type\":\"fp32_attention_algorithm_tune\""
                  << ",\"model\":\"" << command.model << "\""
                  << ",\"operation\":\"" << command.operation << "\""
                  << ",\"sequence\":" << command.sequence
                  << ",\"batches\":" << shape.batches
                  << ",\"m\":" << shape.left_rows
                  << ",\"k\":" << shape.left_columns
                  << ",\"n\":" << shape.output_columns
                  << ",\"transpose_right\":"
                  << (shape.transpose_right ? "true" : "false")
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
                    const auto speedup = default_time[0] / timing[0];
                    if (speedup >= 1.05) ++keep;
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
        std::cerr << "tune_fp32_attention_algorithms: " << error.what() << '\n';
        return 1;
    }
}
