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

#include <hip/hip_runtime.h>
#include <microllm/core/tensor.h>
#include <microllm/ops/ops.h>
#include <microllm/runtime/runtime.h>

namespace {

struct Options {
    std::string model = "qwen";
    std::string projection = "qkv";
    std::int64_t rows = 512;
    int warmup = 3;
    int repetitions = 20;
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
        else if (name == "--rows") result.rows = integer(argv[index + 1], "rows");
        else if (name == "--warmup") {
            result.warmup = static_cast<int>(integer(argv[index + 1], "warmup"));
        } else if (name == "--repetitions") {
            result.repetitions = static_cast<int>(
                integer(argv[index + 1], "repetitions"));
        } else {
            throw std::invalid_argument("unknown option: " + std::string(name));
        }
    }
    if ((result.model != "qwen" && result.model != "deepseek") ||
        (result.projection != "qkv" && result.projection != "gate-up") ||
        result.rows <= 0 || result.rows > 4096 || result.warmup < 0 ||
        result.repetitions <= 0) {
        throw std::invalid_argument("packed weight-gradient options are invalid");
    }
    return result;
}

void check_hip(hipError_t status, const char* operation) {
    if (status != hipSuccess) {
        throw std::runtime_error(std::string(operation) + ": " +
                                 hipGetErrorString(status));
    }
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

}  // namespace

int main(int argc, char** argv) {
    try {
        const auto command = options(argc, argv);
        if (microllm::runtime::hip_device_count() == 0) {
            throw std::runtime_error(
                "packed weight-gradient benchmark requires a visible HIP GPU");
        }
        const auto hidden = command.model == "qwen" ? 896LL : 1536LL;
        const auto kv_width = command.model == "qwen" ? 128LL : 256LL;
        const auto intermediate = command.model == "qwen" ? 4864LL : 8960LL;
        const std::vector<std::int64_t> widths =
            command.projection == "gate-up"
                ? std::vector<std::int64_t>{intermediate, intermediate}
                : std::vector<std::int64_t>{hidden, kv_width, kv_width};
        std::vector<std::int64_t> offsets(widths.size());
        std::int64_t total_width = 0;
        for (std::size_t index = 0; index < widths.size(); ++index) {
            offsets[index] = total_width;
            total_width += widths[index];
        }
        const auto device = microllm::Device::hip(0);
        std::vector<float> input_values(
            static_cast<std::size_t>(command.rows * hidden));
        for (std::size_t index = 0; index < input_values.size(); ++index) {
            input_values[index] =
                static_cast<float>(static_cast<int>(index % 23) - 11) / 127.0F;
        }
        const auto input = microllm::Tensor::from_vector(
            input_values, {command.rows, hidden}).to(device);
        std::vector<microllm::Tensor> gradients;
        std::vector<microllm::Tensor> baseline_outputs;
        for (std::size_t group = 0; group < widths.size(); ++group) {
            std::vector<float> values(static_cast<std::size_t>(
                command.rows * widths[group]));
            for (std::size_t index = 0; index < values.size(); ++index) {
                values[index] = static_cast<float>(
                    static_cast<int>((index + group * 11U) % 29U) - 14) /
                    131.0F;
            }
            gradients.push_back(microllm::Tensor::from_vector(
                values, {command.rows, widths[group]}).to(device));
            baseline_outputs.emplace_back(
                microllm::Shape{hidden, widths[group]},
                microllm::DType::Float32, device);
        }
        microllm::Tensor packed_gradient(
            {command.rows, total_width}, microllm::DType::Float32, device);
        microllm::Tensor packed_output(
            {hidden, total_width}, microllm::DType::Float32, device);
        const auto baseline = [&] {
            for (std::size_t group = 0; group < widths.size(); ++group) {
                microllm::ops::matmul_out_(
                    baseline_outputs[group], input, gradients[group],
                    microllm::ops::MatmulImplementation::HipBLASLt,
                    true, false, {.mode = microllm::ops::OpMode::Training});
            }
        };
        const auto pack = [&] {
            for (std::size_t group = 0; group < widths.size(); ++group) {
                auto* destination = static_cast<float*>(packed_gradient.data()) +
                                    offsets[group];
                check_hip(
                    hipMemcpy2DAsync(
                        destination,
                        static_cast<std::size_t>(total_width) * sizeof(float),
                        gradients[group].data(),
                        static_cast<std::size_t>(widths[group]) * sizeof(float),
                        static_cast<std::size_t>(widths[group]) * sizeof(float),
                        static_cast<std::size_t>(command.rows),
                        hipMemcpyDeviceToDevice, nullptr),
                    "hipMemcpy2DAsync(pack weight gradients)");
            }
        };
        const auto candidate = [&] {
            pack();
            microllm::ops::matmul_out_(
                packed_output, input, packed_gradient,
                microllm::ops::MatmulImplementation::HipBLASLt,
                true, false, {.mode = microllm::ops::OpMode::Training});
        };
        baseline();
        candidate();
        microllm::runtime::synchronize(device);
        const auto packed_values = packed_output.to_vector();
        float maximum_error = 0.0F;
        double squared_error = 0.0;
        std::uint64_t compared = 0;
        for (std::size_t group = 0; group < widths.size(); ++group) {
            const auto reference = baseline_outputs[group].to_vector();
            for (std::int64_t row = 0; row < hidden; ++row) {
                for (std::int64_t column = 0; column < widths[group]; ++column) {
                    const auto reference_index = static_cast<std::size_t>(
                        row * widths[group] + column);
                    const auto packed_index = static_cast<std::size_t>(
                        row * total_width + offsets[group] + column);
                    const auto difference = std::abs(
                        reference[reference_index] - packed_values[packed_index]);
                    maximum_error = std::max(maximum_error, difference);
                    squared_error +=
                        static_cast<double>(difference) * difference;
                    ++compared;
                }
            }
        }
        const auto rms_error = std::sqrt(
            squared_error / static_cast<double>(compared));
        if (!std::isfinite(maximum_error) || maximum_error > 2.0e-3F) {
            throw std::runtime_error(
                "packed weight-gradient complete-output gate failed");
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
            return Timing{percentile(event, 0.5), percentile(event, 0.95),
                          percentile(wall, 0.5), percentile(wall, 0.95)};
        };
        const auto baseline_time = time(baseline);
        const auto candidate_time = time(candidate);
        const auto info = microllm::runtime::device_info(device);
        const auto packed_gradient_bytes = static_cast<std::uint64_t>(
            packed_gradient.storage().num_bytes());
        const auto packed_output_bytes = static_cast<std::uint64_t>(
            packed_output.storage().num_bytes());
        std::cout << std::setprecision(12)
                  << "{\"schema_version\":1,\"status\":\"pass\""
                  << ",\"record_type\":\"packed_weight_gradient_probe\""
                  << ",\"model\":\"" << command.model << "\""
                  << ",\"projection\":\"" << command.projection << "\""
                  << ",\"device_name\":\"" << info.name << "\""
                  << ",\"architecture\":\"" << info.architecture << "\""
                  << ",\"rows\":" << command.rows
                  << ",\"hidden\":" << hidden
                  << ",\"groups\":" << widths.size()
                  << ",\"total_width\":" << total_width
                  << ",\"pack_copies_per_step\":" << widths.size()
                  << ",\"packed_gradient_bytes\":" << packed_gradient_bytes
                  << ",\"packed_output_bytes\":" << packed_output_bytes
                  << ",\"warmup\":" << command.warmup
                  << ",\"repetitions\":" << command.repetitions
                  << ",\"baseline_event_ms_p50\":"
                  << baseline_time.event_p50
                  << ",\"baseline_event_ms_p95\":"
                  << baseline_time.event_p95
                  << ",\"candidate_event_ms_p50\":"
                  << candidate_time.event_p50
                  << ",\"candidate_event_ms_p95\":"
                  << candidate_time.event_p95
                  << ",\"baseline_wall_ms_p50\":"
                  << baseline_time.wall_p50
                  << ",\"candidate_wall_ms_p50\":"
                  << candidate_time.wall_p50
                  << ",\"event_speedup\":"
                  << baseline_time.event_p50 / candidate_time.event_p50
                  << ",\"maximum_absolute_error\":" << maximum_error
                  << ",\"rms_error\":" << rms_error << "}\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "microllm_bench_packed_weight_gradient: "
                  << error.what() << '\n';
        return 1;
    }
}
