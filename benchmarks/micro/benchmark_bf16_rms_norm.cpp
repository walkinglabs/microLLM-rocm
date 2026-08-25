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

#include <microllm/ops/ops.h>
#include <microllm/runtime/runtime.h>

namespace {

struct Options {
    std::string model = "qwen";
    std::int64_t rows = 1024;
    int warmup = 3;
    int repetitions = 30;
};

std::int64_t integer(const char* text, const char* name) {
    std::size_t consumed = 0;
    const auto value = std::stoll(text, &consumed);
    if (consumed != std::string_view(text).size()) {
        throw std::invalid_argument(std::string("invalid ") + name);
    }
    return value;
}

Options options(int argc, char** argv) {
    Options result;
    for (int index = 1; index < argc; index += 2) {
        if (index + 1 >= argc) throw std::invalid_argument("missing option value");
        const std::string_view name(argv[index]);
        if (name == "--model") result.model = argv[index + 1];
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
        result.rows <= 0 || result.rows > 4096 || result.warmup < 0 ||
        result.repetitions <= 0) {
        throw std::invalid_argument("BF16 RMSNorm options are invalid");
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

template <typename Operation>
Timing time(Operation&& operation, int warmup, int repetitions,
            microllm::runtime::Stream& stream) {
    for (int iteration = 0; iteration < warmup; ++iteration) operation();
    stream.synchronize();
    microllm::runtime::Event start(stream.device());
    microllm::runtime::Event finish(stream.device());
    std::vector<double> event;
    std::vector<double> wall;
    for (int iteration = 0; iteration < repetitions; ++iteration) {
        const auto wall_start = std::chrono::steady_clock::now();
        start.record(stream);
        operation();
        finish.record(stream);
        finish.synchronize();
        const auto wall_finish = std::chrono::steady_clock::now();
        event.push_back(finish.elapsed_ms_since(start));
        wall.push_back(std::chrono::duration<double, std::milli>(
                           wall_finish - wall_start).count());
    }
    return {percentile(event, 0.5), percentile(event, 0.95),
            percentile(wall, 0.5), percentile(wall, 0.95)};
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const auto command = options(argc, argv);
        if (microllm::runtime::hip_device_count() == 0) {
            throw std::runtime_error("BF16 RMSNorm benchmark requires HIP");
        }
        const auto width = command.model == "qwen" ? 896LL : 1536LL;
        const auto epsilon = command.model == "qwen" ? 1.0e-6F : 1.0e-6F;
        std::vector<float> input_values(
            static_cast<std::size_t>(command.rows * width));
        std::vector<float> weight_values(static_cast<std::size_t>(width));
        for (std::size_t index = 0; index < input_values.size(); ++index) {
            input_values[index] =
                static_cast<float>(static_cast<int>(index % 67U) - 33) / 32.0F;
        }
        for (std::size_t index = 0; index < weight_values.size(); ++index) {
            weight_values[index] = 0.75F +
                static_cast<float>(index % 19U) / 64.0F;
        }
        const auto device = microllm::Device::hip(0);
        const auto input = microllm::Tensor::from_vector(
            input_values, {command.rows, width}).to(device);
        const auto weight = microllm::Tensor::from_vector(
            weight_values, {width}).to(device);
        microllm::Tensor fp32_output(
            {command.rows, width}, microllm::DType::Float32, device);
        microllm::Tensor baseline_bf16(
            {command.rows, width}, microllm::DType::BFloat16, device);
        microllm::Tensor candidate_bf16(
            {command.rows, width}, microllm::DType::BFloat16, device);
        microllm::runtime::Stream stream(device);
        const microllm::ops::OpContext context{&stream, nullptr, 0};
        const auto baseline = [&] {
            microllm::ops::rms_norm_out_(
                fp32_output, input, weight, epsilon, context);
            microllm::ops::cast_out_(fp32_output, baseline_bf16, context);
        };
        const auto candidate = [&] {
            microllm::ops::rms_norm_bf16_out_(
                candidate_bf16, input, weight, epsilon, context);
        };
        baseline();
        candidate();
        stream.synchronize();
        const auto reference = baseline_bf16.to_vector();
        const auto actual = candidate_bf16.to_vector();
        if (actual != reference) {
            throw std::runtime_error("fused BF16 RMSNorm changed complete output");
        }
        microllm::runtime::reset_transfer_stats();
        const auto baseline_time = time(
            baseline, command.warmup, command.repetitions, stream);
        const auto candidate_time = time(
            candidate, command.warmup, command.repetitions, stream);
        const auto transfers = microllm::runtime::transfer_stats();
        std::cout << std::setprecision(12)
                  << "{\"schema_version\":1,\"status\":\"pass\""
                  << ",\"record_type\":\"bf16_rms_norm_output_probe\""
                  << ",\"model\":\"" << command.model << "\""
                  << ",\"rows\":" << command.rows
                  << ",\"width\":" << width
                  << ",\"elements\":" << command.rows * width
                  << ",\"warmup\":" << command.warmup
                  << ",\"repetitions\":" << command.repetitions
                  << ",\"complete_output_equal\":true"
                  << ",\"baseline_event_ms_p50\":" << baseline_time.event_p50
                  << ",\"candidate_event_ms_p50\":" << candidate_time.event_p50
                  << ",\"event_speedup\":"
                  << baseline_time.event_p50 / candidate_time.event_p50
                  << ",\"baseline_wall_ms_p50\":" << baseline_time.wall_p50
                  << ",\"candidate_wall_ms_p50\":" << candidate_time.wall_p50
                  << ",\"wall_speedup\":"
                  << baseline_time.wall_p50 / candidate_time.wall_p50
                  << ",\"host_to_device_calls\":"
                  << transfers.host_to_device_calls
                  << ",\"device_to_host_calls\":"
                  << transfers.device_to_host_calls << "}\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "BF16 RMSNorm benchmark failed: " << error.what() << '\n';
        return 1;
    }
}
