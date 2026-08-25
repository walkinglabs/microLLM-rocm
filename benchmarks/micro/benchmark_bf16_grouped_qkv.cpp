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
#include <microllm/runtime/memory.h>
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
    hipblasLtHandle_t get() const noexcept { return value_; }

private:
    hipblasLtHandle_t value_ = nullptr;
};

struct Options {
    std::string model = "qwen";
    std::string projection = "qkv";
    std::string output_dtype = "fp32";
    bool equal_width = false;
    bool gate_swish = false;
    std::int64_t rows = 512;
    int warmup = 2;
    int repetitions = 5;
    int maximum_algorithms = 16;
    std::size_t workspace_limit = 32U * 1024U * 1024U;
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
        else if (name == "--output-dtype") result.output_dtype = argv[index + 1];
        else if (name == "--equal-width") {
            const std::string value = argv[index + 1];
            if (value != "true" && value != "false") {
                throw std::invalid_argument("equal-width must be true or false");
            }
            result.equal_width = value == "true";
        }
        else if (name == "--gate-swish") {
            const std::string value = argv[index + 1];
            if (value != "true" && value != "false") {
                throw std::invalid_argument("gate-swish must be true or false");
            }
            result.gate_swish = value == "true";
        }
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
            const auto value = integer(argv[index + 1], "workspace-bytes");
            if (value < 0) throw std::invalid_argument("workspace must be nonnegative");
            result.workspace_limit = static_cast<std::size_t>(value);
        } else {
            throw std::invalid_argument("unknown option: " + std::string(name));
        }
    }
    if ((result.model != "qwen" && result.model != "deepseek") ||
        (result.projection != "qkv" && result.projection != "gate-up") ||
        (result.output_dtype != "fp32" && result.output_dtype != "bf16" &&
         result.output_dtype != "model") ||
        result.rows <= 0 || result.rows > 4096 || result.warmup < 0 ||
        result.repetitions <= 0 || result.maximum_algorithms <= 0 ||
        result.maximum_algorithms > 256) {
        throw std::invalid_argument(
            "grouped BF16 projection options are invalid");
    }
    if (result.gate_swish &&
        (result.projection != "gate-up" || result.output_dtype != "bf16")) {
        throw std::invalid_argument(
            "gate-swish requires gate-up with bf16 output");
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

struct Errors {
    float maximum = 0.0F;
    double rms = 0.0;
    bool finite = true;
};

Errors compare(const std::vector<float>& reference,
               const std::vector<float>& actual) {
    if (reference.size() != actual.size()) {
        throw std::runtime_error("grouped BF16 output size changed");
    }
    Errors result;
    double squared = 0.0;
    for (std::size_t index = 0; index < reference.size(); ++index) {
        result.finite = result.finite && std::isfinite(actual[index]);
        const auto difference = std::abs(reference[index] - actual[index]);
        result.maximum = std::max(result.maximum, difference);
        squared += static_cast<double>(difference) * difference;
    }
    result.rms = reference.empty()
                     ? 0.0 : std::sqrt(
                         squared / static_cast<double>(reference.size()));
    return result;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const auto command = options(argc, argv);
        if (microllm::runtime::hip_device_count() == 0) {
            throw std::runtime_error(
                "grouped BF16 projection requires a visible HIP GPU");
        }
        const auto hidden = command.model == "qwen" ? 896LL : 1536LL;
        const auto query_width = hidden;
        const auto kv_width = command.model == "qwen" ? 128LL : 256LL;
        const auto intermediate =
            command.model == "qwen" ? 4864LL : 8960LL;
        const std::vector<std::int64_t> widths =
            command.projection == "gate-up"
                ? std::vector<std::int64_t>{intermediate, intermediate}
                : command.equal_width
                      ? std::vector<std::int64_t>{
                            query_width, query_width, query_width}
                      : std::vector<std::int64_t>{
                            query_width, kv_width, kv_width};
        const auto record_type = command.projection == "gate-up"
                                     ? "bf16_grouped_gate_up_probe"
                                     : "bf16_grouped_qkv_probe";
        const auto model_comparison = command.output_dtype == "model";
        const auto maximum_tolerance = command.gate_swish
                                           ? 2.0e-2F
                                           : model_comparison ? 0.25F : 2.0e-4F;
        const auto grouped_output_dtype = command.output_dtype == "fp32"
                                              ? microllm::DType::Float32
                                              : microllm::DType::BFloat16;
        const auto baseline_output_dtype = model_comparison
                                               ? microllm::DType::Float32
                                               : grouped_output_dtype;
        const auto hip_output_dtype = command.output_dtype == "fp32"
                                          ? HIP_R_32F : HIP_R_16BF;
        const auto device = microllm::Device::hip(0);
        std::vector<float> input_values(static_cast<std::size_t>(
            command.rows * hidden));
        for (std::size_t index = 0; index < input_values.size(); ++index) {
            input_values[index] =
                static_cast<float>(static_cast<int>(index % 23) - 11) / 127.0F;
        }
        const auto input = microllm::Tensor::from_vector(
            input_values, {command.rows, hidden},
            microllm::DType::BFloat16).to(device);
        std::vector<microllm::Tensor> weights;
        std::vector<microllm::Tensor> baseline_outputs;
        std::vector<microllm::Tensor> grouped_outputs;
        std::vector<microllm::Tensor> candidate_outputs;
        std::vector<microllm::Tensor> fallbacks;
        for (std::size_t group = 0; group < widths.size(); ++group) {
            const auto width = widths[group];
            std::vector<float> values(static_cast<std::size_t>(hidden * width));
            for (std::size_t index = 0; index < values.size(); ++index) {
                values[index] = static_cast<float>(
                    static_cast<int>((index + group * 7U) % 29U) - 14) / 131.0F;
            }
            weights.push_back(microllm::Tensor::from_vector(
                values, {hidden, width}, microllm::DType::BFloat16).to(device));
            baseline_outputs.emplace_back(
                microllm::Shape{command.rows, width}, baseline_output_dtype,
                device);
            grouped_outputs.emplace_back(
                microllm::Shape{command.rows, width}, grouped_output_dtype,
                device);
            candidate_outputs.emplace_back(
                microllm::Shape{command.rows, width}, baseline_output_dtype,
                device);
            fallbacks.emplace_back(
                microllm::Shape{command.rows, width}, microllm::DType::BFloat16,
                device);
        }

        const auto baseline = [&] {
            for (std::size_t group = 0; group < widths.size(); ++group) {
                microllm::ops::bf16_matmul_output_out_(
                    baseline_outputs[group], input, weights[group],
                    fallbacks[group]);
            }
        };
        baseline();
        microllm::runtime::synchronize(device);
        std::vector<std::vector<float>> references;
        for (std::size_t group = 0; group < baseline_outputs.size(); ++group) {
            auto values = baseline_outputs[group].to_vector();
            if (command.gate_swish && group == 0) {
                for (auto& value : values) {
                    const auto exponential = std::exp(-std::abs(value));
                    const auto sigmoid = value >= 0.0F
                        ? 1.0F / (1.0F + exponential)
                        : exponential / (1.0F + exponential);
                    value *= sigmoid;
                }
                values = microllm::Tensor::from_vector(
                    values, baseline_outputs[group].shape(),
                    microllm::DType::BFloat16).to_vector();
            }
            references.push_back(std::move(values));
        }

        Handle handle;
        hipblaslt_ext::GroupedGemm grouped(
            handle.get(), HIPBLAS_OP_N, HIPBLAS_OP_N,
            HIP_R_16BF, HIP_R_16BF, hip_output_dtype, hip_output_dtype,
            HIPBLAS_COMPUTE_32F);
        grouped.setMaxWorkspaceBytes(command.workspace_limit);
        std::vector<std::int64_t> m(widths.begin(), widths.end());
        std::vector<std::int64_t> n(widths.size(), command.rows);
        std::vector<std::int64_t> k(widths.size(), hidden);
        std::vector<std::int64_t> batch(widths.size(), 1);
        std::vector<hipblaslt_ext::GemmEpilogue> epilogues(widths.size());
        if (command.gate_swish) {
            epilogues[0].setMode(HIPBLASLT_EPILOGUE_SWISH_EXT);
        }
        std::vector<hipblaslt_ext::GemmInputs> inputs(widths.size());
        float alpha = 1.0F;
        float beta = 0.0F;
        for (std::size_t group = 0; group < widths.size(); ++group) {
            inputs[group].setA(weights[group].data());
            inputs[group].setB(input.data());
            inputs[group].setC(grouped_outputs[group].data());
            inputs[group].setD(grouped_outputs[group].data());
            inputs[group].setAlpha(&alpha);
            inputs[group].setBeta(&beta);
        }
        check(grouped.setProblem(m, n, k, batch, epilogues, inputs),
              "GroupedGemm::setProblem");
        std::vector<hipblasLtMatmulHeuristicResult_t> algorithms;
        const auto inventory_status = hipblaslt_ext::getAllAlgos(
            handle.get(), hipblaslt_ext::GemmType::HIPBLASLT_GROUPED_GEMM,
            HIPBLAS_OP_N, HIPBLAS_OP_N, HIP_R_16BF, HIP_R_16BF,
            hip_output_dtype, hip_output_dtype, HIPBLAS_COMPUTE_32F,
            algorithms);
        if (inventory_status != HIPBLAS_STATUS_SUCCESS) {
            std::cout << "{\"schema_version\":1,\"status\":\"pass\""
                      << ",\"record_type\":\"" << record_type << "\""
                      << ",\"model\":\"" << command.model << "\""
                      << ",\"projection\":\"" << command.projection << "\""
                      << ",\"output_dtype\":\"" << command.output_dtype << "\""
                      << ",\"equal_width\":"
                      << (command.equal_width ? "true" : "false")
                      << ",\"gate_swish\":"
                      << (command.gate_swish ? "true" : "false")
                      << ",\"rows\":" << command.rows
                      << ",\"hidden\":" << hidden
                      << ",\"query_width\":" << query_width
                      << ",\"kv_width\":" << kv_width
                      << ",\"intermediate\":" << intermediate
                      << ",\"groups\":" << widths.size()
                      << ",\"algorithm_count\":" << algorithms.size()
                      << ",\"supported_candidates\":0"
                      << ",\"passing_candidates\":0"
                      << ",\"grouped_supported\":false}\n";
            return 0;
        }

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
            return std::vector<double>{
                percentile(event, 0.5), percentile(event, 0.95),
                percentile(wall, 0.5), percentile(wall, 0.95)};
        };
        const auto baseline_time = time(baseline);
        microllm::Storage workspace(command.workspace_limit, device);
        int supported_candidates = 0;
        int passing_candidates = 0;
        int solution_index = -1;
        std::size_t workspace_bytes = 0;
        hipblasLtMatmulAlgo_t best_algorithm{};
        Errors errors;
        std::vector<double> grouped_time{0.0, 0.0, 0.0, 0.0};
        for (auto& result : algorithms) {
            hipblaslt_ext::GemmTuning tuning;
            std::size_t candidate_workspace = 0;
            auto candidate = result.algo;
            if (grouped.isAlgoSupported(
                    candidate, tuning, candidate_workspace) !=
                    HIPBLAS_STATUS_SUCCESS ||
                candidate_workspace > command.workspace_limit) {
                continue;
            }
            if (supported_candidates >= command.maximum_algorithms) break;
            ++supported_candidates;
            check(grouped.initialize(
                      candidate, workspace.data(), false, nullptr),
                  "GroupedGemm::initialize(candidate)");
            const auto candidate_operation = [&] {
                check(grouped.run(nullptr), "GroupedGemm::run(candidate)");
                if (model_comparison) {
                    for (std::size_t group = 0; group < widths.size(); ++group) {
                        microllm::ops::cast_out_(
                            grouped_outputs[group], candidate_outputs[group]);
                    }
                }
            };
            candidate_operation();
            microllm::runtime::synchronize(device);
            Errors candidate_errors;
            for (std::size_t group = 0; group < widths.size(); ++group) {
                const auto current = compare(
                    references[group],
                    (model_comparison ? candidate_outputs[group]
                                      : grouped_outputs[group]).to_vector());
                candidate_errors.maximum = std::max(
                    candidate_errors.maximum, current.maximum);
                candidate_errors.rms = std::max(
                    candidate_errors.rms, current.rms);
                candidate_errors.finite = candidate_errors.finite && current.finite;
            }
            if (!candidate_errors.finite ||
                candidate_errors.maximum > maximum_tolerance) {
                continue;
            }
            ++passing_candidates;
            const auto candidate_time = time(candidate_operation);
            if (solution_index < 0 || candidate_time[0] < grouped_time[0]) {
                solution_index = hipblaslt_ext::getIndexFromAlgo(candidate);
                workspace_bytes = candidate_workspace;
                best_algorithm = candidate;
                errors = candidate_errors;
                grouped_time = candidate_time;
            }
        }
        const auto supported = solution_index >= 0;
        if (!supported) {
            std::cout << "{\"schema_version\":1,\"status\":\"pass\""
                      << ",\"record_type\":\"" << record_type << "\""
                      << ",\"model\":\"" << command.model << "\""
                      << ",\"projection\":\"" << command.projection << "\""
                      << ",\"output_dtype\":\"" << command.output_dtype << "\""
                      << ",\"equal_width\":"
                      << (command.equal_width ? "true" : "false")
                      << ",\"gate_swish\":"
                      << (command.gate_swish ? "true" : "false")
                      << ",\"rows\":" << command.rows
                      << ",\"hidden\":" << hidden
                      << ",\"query_width\":" << query_width
                      << ",\"kv_width\":" << kv_width
                      << ",\"intermediate\":" << intermediate
                      << ",\"groups\":" << widths.size()
                      << ",\"algorithm_count\":" << algorithms.size()
                      << ",\"supported_candidates\":" << supported_candidates
                      << ",\"passing_candidates\":" << passing_candidates
                      << ",\"grouped_supported\":false}\n";
            return 0;
        }
        const auto reinitialized_time = time([&] {
            check(grouped.setProblem(m, n, k, batch, epilogues, inputs),
                  "GroupedGemm::setProblem(timed)");
            check(grouped.initialize(
                      best_algorithm, workspace.data(), false, nullptr),
                  "GroupedGemm::initialize(timed)");
            check(grouped.run(nullptr), "GroupedGemm::run(reinitialized)");
            if (model_comparison) {
                for (std::size_t group = 0; group < widths.size(); ++group) {
                    microllm::ops::cast_out_(
                        grouped_outputs[group], candidate_outputs[group]);
                }
            }
        });
        const auto user_arguments_setup_start = std::chrono::steady_clock::now();
        check(grouped.setProblem(m, n, k, batch, epilogues, inputs),
              "GroupedGemm::setProblem(user arguments)");
        check(grouped.initialize(
                  best_algorithm, workspace.data(), true, nullptr),
              "GroupedGemm::initialize(user arguments)");
        std::vector<hipblaslt_ext::UserArguments> host_arguments(widths.size());
        check(grouped.getDefaultValueForDeviceUserArguments(
                  host_arguments.data()),
              "GroupedGemm::getDefaultValueForDeviceUserArguments");
        microllm::Storage device_arguments(
            host_arguments.size() * sizeof(hipblaslt_ext::UserArguments), device);
        microllm::runtime::copy_bytes(
            device_arguments.data(), device, host_arguments.data(),
            microllm::Device::cpu(), device_arguments.num_bytes());
        microllm::runtime::synchronize(device);
        const auto user_arguments_setup_finish = std::chrono::steady_clock::now();
        const auto user_arguments_time = time([&] {
            check(grouped.run(device_arguments.data(), nullptr),
                  "GroupedGemm::run(user arguments)");
            if (model_comparison) {
                for (std::size_t group = 0; group < widths.size(); ++group) {
                    microllm::ops::cast_out_(
                        grouped_outputs[group], candidate_outputs[group]);
                }
            }
        });
        const auto user_arguments_setup_ms =
            std::chrono::duration<double, std::milli>(
                user_arguments_setup_finish - user_arguments_setup_start).count();
        std::cout << std::setprecision(12)
                  << "{\"schema_version\":1,\"status\":\"pass\""
                  << ",\"record_type\":\"" << record_type << "\""
                  << ",\"model\":\"" << command.model << "\""
                  << ",\"projection\":\"" << command.projection << "\""
                  << ",\"output_dtype\":\"" << command.output_dtype << "\""
                  << ",\"equal_width\":"
                  << (command.equal_width ? "true" : "false")
                  << ",\"gate_swish\":"
                  << (command.gate_swish ? "true" : "false")
                  << ",\"rows\":" << command.rows
                  << ",\"hidden\":" << hidden
                  << ",\"query_width\":" << query_width
                  << ",\"kv_width\":" << kv_width
                  << ",\"intermediate\":" << intermediate
                  << ",\"groups\":" << widths.size()
                  << ",\"algorithm_count\":" << algorithms.size()
                  << ",\"supported_candidates\":" << supported_candidates
                  << ",\"passing_candidates\":" << passing_candidates
                  << ",\"grouped_supported\":true"
                  << ",\"solution_index\":" << solution_index
                  << ",\"workspace_bytes\":" << workspace_bytes
                  << ",\"maximum_absolute_tolerance\":" << maximum_tolerance
                  << ",\"finite\":" << (errors.finite ? "true" : "false")
                  << ",\"maximum_absolute_error\":" << errors.maximum
                  << ",\"maximum_rms_error\":" << errors.rms
                  << ",\"baseline_event_ms_p50\":" << baseline_time[0]
                  << ",\"baseline_event_ms_p95\":" << baseline_time[1]
                  << ",\"baseline_wall_ms_p50\":" << baseline_time[2]
                  << ",\"baseline_wall_ms_p95\":" << baseline_time[3]
                  << ",\"grouped_event_ms_p50\":" << grouped_time[0]
                  << ",\"grouped_event_ms_p95\":" << grouped_time[1]
                  << ",\"grouped_wall_ms_p50\":" << grouped_time[2]
                  << ",\"grouped_wall_ms_p95\":" << grouped_time[3]
                  << ",\"event_speedup\":"
                  << baseline_time[0] / grouped_time[0]
                  << ",\"wall_speedup\":"
                  << baseline_time[2] / grouped_time[2]
                  << ",\"reinitialized_event_ms_p50\":"
                  << reinitialized_time[0]
                  << ",\"reinitialized_wall_ms_p50\":"
                  << reinitialized_time[2]
                  << ",\"reinitialized_event_speedup\":"
                  << baseline_time[0] / reinitialized_time[0]
                  << ",\"reinitialized_wall_speedup\":"
                  << baseline_time[2] / reinitialized_time[2]
                  << ",\"user_arguments_setup_ms\":"
                  << user_arguments_setup_ms
                  << ",\"user_arguments_event_ms_p50\":"
                  << user_arguments_time[0]
                  << ",\"user_arguments_wall_ms_p50\":"
                  << user_arguments_time[2]
                  << ",\"user_arguments_event_speedup\":"
                  << baseline_time[0] / user_arguments_time[0]
                  << ",\"user_arguments_wall_speedup\":"
                  << baseline_time[2] / user_arguments_time[2] << "}\n";
        return errors.finite && errors.maximum <= maximum_tolerance ? 0 : 2;
    } catch (const std::exception& error) {
        std::cerr << "grouped BF16 projection probe failed: "
                  << error.what() << '\n';
        return 1;
    }
}
