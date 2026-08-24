#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include <hipblaslt/hipblaslt-ext.hpp>
#include <hipblaslt/hipblaslt.h>
#include <microllm/core/storage.h>
#include <microllm/core/tensor.h>
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
    Handle(const Handle&) = delete;
    Handle& operator=(const Handle&) = delete;
    [[nodiscard]] hipblasLtHandle_t get() const noexcept { return value_; }

private:
    hipblasLtHandle_t value_ = nullptr;
};

struct Options {
    std::string model = "qwen";
    std::string projection = "qkv";
    std::string input_layout = "direct";
    std::int64_t rows = 512;
    int warmup = 3;
    int repetitions = 10;
    int maximum_algorithms = 32;
    std::size_t workspace_limit = 64U * 1024U * 1024U;
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
        else if (name == "--projection") result.projection = argv[index + 1];
        else if (name == "--input-layout") result.input_layout = argv[index + 1];
        else if (name == "--rows") result.rows = integer(argv[index + 1], "rows");
        else if (name == "--warmup") {
            result.warmup = static_cast<int>(integer(argv[index + 1], "warmup"));
        } else if (name == "--repetitions") {
            result.repetitions = static_cast<int>(
                integer(argv[index + 1], "repetitions"));
        } else if (name == "--maximum-algorithms") {
            result.maximum_algorithms = static_cast<int>(
                integer(argv[index + 1], "maximum-algorithms"));
        } else if (name == "--workspace-bytes") {
            const auto bytes = integer(argv[index + 1], "workspace-bytes");
            if (bytes < 0) throw std::invalid_argument("workspace must be nonnegative");
            result.workspace_limit = static_cast<std::size_t>(bytes);
        } else {
            throw std::invalid_argument("unknown option: " + std::string(name));
        }
    }
    if ((result.model != "qwen" && result.model != "deepseek") ||
        (result.projection != "qkv" && result.projection != "gate-up") ||
        (result.input_layout != "direct" &&
         result.input_layout != "materialized") ||
        result.rows <= 0 || result.rows > 4096 || result.warmup < 0 ||
        result.repetitions <= 0 || result.maximum_algorithms <= 0 ||
        result.maximum_algorithms > 256) {
        throw std::invalid_argument("grouped weight-gradient options are invalid");
    }
    return result;
}

double percentile(std::vector<double> values, double fraction) {
    std::sort(values.begin(), values.end());
    const auto position = fraction * static_cast<double>(values.size() - 1);
    const auto low = static_cast<std::size_t>(position);
    const auto high = std::min(low + 1, values.size() - 1);
    return values[low] + (values[high] - values[low]) *
                             (position - static_cast<double>(low));
}

struct Timing {
    double event_p50 = 0.0;
    double event_p95 = 0.0;
    double wall_p50 = 0.0;
    double wall_p95 = 0.0;
};

struct Errors {
    float maximum = 0.0F;
    double rms = 0.0;
    bool finite = true;
};

Errors compare(const std::vector<float>& expected,
               const std::vector<float>& actual) {
    if (expected.size() != actual.size()) {
        throw std::runtime_error("grouped weight-gradient output size changed");
    }
    Errors result;
    double squared = 0.0;
    for (std::size_t index = 0; index < expected.size(); ++index) {
        result.finite = result.finite && std::isfinite(actual[index]);
        const auto difference = std::abs(expected[index] - actual[index]);
        result.maximum = std::max(result.maximum, difference);
        squared += static_cast<double>(difference) * difference;
    }
    result.rms = expected.empty()
                     ? 0.0
                     : std::sqrt(squared / static_cast<double>(expected.size()));
    return result;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const auto command = options(argc, argv);
        if (microllm::runtime::hip_device_count() == 0) {
            throw std::runtime_error(
                "grouped weight-gradient probe requires a visible HIP GPU");
        }
        const auto hidden = command.model == "qwen" ? 896LL : 1536LL;
        const auto kv_width = command.model == "qwen" ? 128LL : 256LL;
        const auto intermediate = command.model == "qwen" ? 4864LL : 8960LL;
        const std::vector<std::int64_t> widths =
            command.projection == "gate-up"
                ? std::vector<std::int64_t>{intermediate, intermediate}
                : std::vector<std::int64_t>{hidden, kv_width, kv_width};
        const auto device = microllm::Device::hip(0);
        std::vector<float> input_values(
            static_cast<std::size_t>(command.rows * hidden));
        for (std::size_t index = 0; index < input_values.size(); ++index) {
            input_values[index] =
                static_cast<float>(static_cast<int>(index % 23) - 11) / 127.0F;
        }
        const auto input = microllm::Tensor::from_vector(
            input_values, {command.rows, hidden}).to(device);
        microllm::Tensor transposed_input(
            {hidden, command.rows}, microllm::DType::Float32, device);
        std::vector<microllm::Tensor> gradients;
        std::vector<microllm::Tensor> baseline_outputs;
        std::vector<microllm::Tensor> grouped_outputs;
        for (std::size_t group = 0; group < widths.size(); ++group) {
            const auto width = widths[group];
            std::vector<float> values(
                static_cast<std::size_t>(command.rows * width));
            for (std::size_t index = 0; index < values.size(); ++index) {
                values[index] = static_cast<float>(
                    static_cast<int>((index + group * 11U) % 29U) - 14) /
                    131.0F;
            }
            gradients.push_back(microllm::Tensor::from_vector(
                values, {command.rows, width}).to(device));
            baseline_outputs.emplace_back(
                microllm::Shape{hidden, width}, microllm::DType::Float32,
                device);
            grouped_outputs.emplace_back(
                microllm::Shape{hidden, width}, microllm::DType::Float32,
                device);
        }
        const auto baseline = [&] {
            for (std::size_t group = 0; group < widths.size(); ++group) {
                microllm::ops::matmul_out_(
                    baseline_outputs[group], input, gradients[group],
                    microllm::ops::MatmulImplementation::HipBLASLt, true,
                    false, {.mode = microllm::ops::OpMode::Training});
            }
        };
        baseline();
        microllm::runtime::synchronize(device);
        std::vector<std::vector<float>> references;
        references.reserve(widths.size());
        for (const auto& output : baseline_outputs) {
            references.push_back(output.to_vector());
        }

        Handle handle;
        const auto transpose_input = command.input_layout == "materialized";
        const auto grouped_transpose_b =
            transpose_input ? HIPBLAS_OP_N : HIPBLAS_OP_T;
        hipblaslt_ext::GroupedGemm grouped(
            handle.get(), HIPBLAS_OP_N, grouped_transpose_b,
            HIP_R_32F, HIP_R_32F, HIP_R_32F, HIP_R_32F,
            HIPBLAS_COMPUTE_32F);
        grouped.setMaxWorkspaceBytes(command.workspace_limit);
        std::vector<std::int64_t> m(widths.begin(), widths.end());
        std::vector<std::int64_t> n(widths.size(), hidden);
        std::vector<std::int64_t> k(widths.size(), command.rows);
        std::vector<std::int64_t> batch(widths.size(), 1);
        std::vector<hipblaslt_ext::GemmEpilogue> epilogues(widths.size());
        std::vector<hipblaslt_ext::GemmInputs> inputs(widths.size());
        float alpha = 1.0F;
        float beta = 0.0F;
        for (std::size_t group = 0; group < widths.size(); ++group) {
            inputs[group].setA(gradients[group].data());
            inputs[group].setB(transpose_input ? transposed_input.data()
                                               : input.data());
            inputs[group].setC(grouped_outputs[group].data());
            inputs[group].setD(grouped_outputs[group].data());
            inputs[group].setAlpha(&alpha);
            inputs[group].setBeta(&beta);
        }
        check(grouped.setProblem(m, n, k, batch, epilogues, inputs),
              "GroupedGemm::setProblem(weight gradient)");
        std::vector<hipblasLtMatmulHeuristicResult_t> algorithms;
        const auto inventory_status = hipblaslt_ext::getAllAlgos(
            handle.get(), hipblaslt_ext::GemmType::HIPBLASLT_GROUPED_GEMM,
            HIPBLAS_OP_N, grouped_transpose_b, HIP_R_32F, HIP_R_32F, HIP_R_32F,
            HIP_R_32F, HIPBLAS_COMPUTE_32F, algorithms);
        if (inventory_status != HIPBLAS_STATUS_SUCCESS) algorithms.clear();

        microllm::runtime::Event start(device);
        microllm::runtime::Event finish(device);
        const auto time = [&](const auto& operation) {
            for (int iteration = 0; iteration < command.warmup; ++iteration) {
                operation();
            }
            microllm::runtime::synchronize(device);
            std::vector<double> event;
            std::vector<double> wall;
            for (int iteration = 0; iteration < command.repetitions; ++iteration) {
                const auto wall_start = std::chrono::steady_clock::now();
                start.record_default_stream();
                operation();
                finish.record_default_stream();
                finish.synchronize();
                const auto wall_finish = std::chrono::steady_clock::now();
                event.push_back(finish.elapsed_ms_since(start));
                wall.push_back(std::chrono::duration<double, std::milli>(
                    wall_finish - wall_start).count());
            }
            return Timing{percentile(event, 0.5), percentile(event, 0.95),
                          percentile(wall, 0.5), percentile(wall, 0.95)};
        };
        const auto baseline_time = time(baseline);
        microllm::Storage workspace(command.workspace_limit, device);
        int supported_candidates = 0;
        int passing_candidates = 0;
        int selected_index = -1;
        std::size_t selected_workspace = 0;
        Timing grouped_time;
        Errors selected_errors;
        hipblasLtMatmulAlgo_t selected_algorithm{};
        for (auto& result : algorithms) {
            hipblaslt_ext::GemmTuning tuning;
            std::size_t workspace_bytes = 0;
            auto algorithm = result.algo;
            if (grouped.isAlgoSupported(
                    algorithm, tuning, workspace_bytes) !=
                    HIPBLAS_STATUS_SUCCESS ||
                workspace_bytes > command.workspace_limit) {
                continue;
            }
            if (supported_candidates >= command.maximum_algorithms) break;
            ++supported_candidates;
            check(grouped.initialize(
                      algorithm, workspace.data(), false, nullptr),
                  "GroupedGemm::initialize(weight gradient)");
            const auto operation = [&] {
                if (transpose_input) {
                    microllm::ops::cast_transpose_2d_out_(
                        input, transposed_input);
                }
                check(grouped.run(nullptr),
                      "GroupedGemm::run(weight gradient)");
            };
            operation();
            microllm::runtime::synchronize(device);
            Errors errors;
            for (std::size_t group = 0; group < widths.size(); ++group) {
                const auto current = compare(
                    references[group], grouped_outputs[group].to_vector());
                errors.maximum = std::max(errors.maximum, current.maximum);
                errors.rms = std::max(errors.rms, current.rms);
                errors.finite = errors.finite && current.finite;
            }
            if (!errors.finite || errors.maximum > 2.0e-3F) continue;
            ++passing_candidates;
            const auto timing = time(operation);
            if (selected_index < 0 ||
                timing.event_p50 < grouped_time.event_p50) {
                selected_index = hipblaslt_ext::getIndexFromAlgo(algorithm);
                selected_workspace = workspace_bytes;
                selected_algorithm = algorithm;
                selected_errors = errors;
                grouped_time = timing;
            }
        }
        const auto supported = selected_index >= 0;
        if (supported) {
            check(grouped.initialize(
                      selected_algorithm, workspace.data(), false, nullptr),
                  "GroupedGemm::initialize(selected weight gradient)");
            grouped_time = time([&] {
                if (transpose_input) {
                    microllm::ops::cast_transpose_2d_out_(
                        input, transposed_input);
                }
                check(grouped.run(nullptr),
                      "GroupedGemm::run(selected weight gradient)");
            });
        }
        const auto info = microllm::runtime::device_info(device);
        std::cout << std::setprecision(12)
                  << "{\"schema_version\":1,\"status\":\"pass\""
                  << ",\"record_type\":\"grouped_weight_gradient_probe\""
                  << ",\"model\":\"" << command.model << "\""
                  << ",\"projection\":\"" << command.projection << "\""
                  << ",\"input_layout\":\"" << command.input_layout << "\""
                  << ",\"device_name\":\"" << info.name << "\""
                  << ",\"architecture\":\"" << info.architecture << "\""
                  << ",\"rows\":" << command.rows
                  << ",\"hidden\":" << hidden
                  << ",\"groups\":" << widths.size()
                  << ",\"algorithm_count\":" << algorithms.size()
                  << ",\"supported_candidates\":" << supported_candidates
                  << ",\"passing_candidates\":" << passing_candidates
                  << ",\"grouped_supported\":"
                  << (supported ? "true" : "false")
                  << ",\"selected_solution_index\":" << selected_index
                  << ",\"workspace_bytes\":" << selected_workspace
                  << ",\"warmup\":" << command.warmup
                  << ",\"repetitions\":" << command.repetitions
                  << ",\"baseline_event_ms_p50\":"
                  << baseline_time.event_p50
                  << ",\"baseline_event_ms_p95\":"
                  << baseline_time.event_p95
                  << ",\"baseline_wall_ms_p50\":"
                  << baseline_time.wall_p50
                  << ",\"baseline_wall_ms_p95\":"
                  << baseline_time.wall_p95
                  << ",\"grouped_event_ms_p50\":"
                  << grouped_time.event_p50
                  << ",\"grouped_event_ms_p95\":"
                  << grouped_time.event_p95
                  << ",\"grouped_wall_ms_p50\":"
                  << grouped_time.wall_p50
                  << ",\"grouped_wall_ms_p95\":"
                  << grouped_time.wall_p95
                  << ",\"event_speedup\":"
                  << (supported ? baseline_time.event_p50 /
                                      grouped_time.event_p50
                                : 0.0)
                  << ",\"maximum_absolute_error\":"
                  << selected_errors.maximum
                  << ",\"maximum_rms_error\":" << selected_errors.rms
                  << "}\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "microllm_bench_grouped_weight_gradient: "
                  << error.what() << '\n';
        return 1;
    }
}
