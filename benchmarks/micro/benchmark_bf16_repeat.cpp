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

#include <microllm/ops/ops.h>
#include <microllm/runtime/runtime.h>

namespace {

struct Options {
    std::int64_t batch = 1;
    std::int64_t sequence = 512;
    std::int64_t kv_heads = 2;
    std::int64_t width = 64;
    std::int64_t repeats = 7;
    bool fused = false;
    int warmup = 3;
    int repetitions = 20;
};

std::int64_t integer(const char* value, const char* name) {
    char* end = nullptr;
    const auto result = std::strtoll(value, &end, 10);
    if (end == value || *end != '\0') {
        throw std::invalid_argument(std::string("invalid ") + name);
    }
    return result;
}

bool boolean(const char* value, const char* name) {
    const std::string_view text(value);
    if (text == "true") return true;
    if (text == "false") return false;
    throw std::invalid_argument(std::string(name) + " must be true or false");
}

Options parse(int argc, char** argv) {
    Options result;
    for (int index = 1; index < argc; index += 2) {
        if (index + 1 >= argc) {
            throw std::invalid_argument("option is missing a value");
        }
        const std::string_view name(argv[index]);
        if (name == "--batch") result.batch = integer(argv[index + 1], "batch");
        else if (name == "--sequence") {
            result.sequence = integer(argv[index + 1], "sequence");
        } else if (name == "--kv-heads") {
            result.kv_heads = integer(argv[index + 1], "kv-heads");
        } else if (name == "--width") {
            result.width = integer(argv[index + 1], "width");
        } else if (name == "--repeats") {
            result.repeats = integer(argv[index + 1], "repeats");
        } else if (name == "--fused") {
            result.fused = boolean(argv[index + 1], "fused");
        } else if (name == "--warmup") {
            result.warmup = static_cast<int>(integer(argv[index + 1], "warmup"));
        } else if (name == "--repetitions") {
            result.repetitions = static_cast<int>(
                integer(argv[index + 1], "repetitions"));
        } else {
            throw std::invalid_argument("unknown option: " + std::string(name));
        }
    }
    if (result.batch <= 0 || result.batch > 8 || result.sequence < 1 ||
        result.sequence > 2048 || result.kv_heads <= 0 ||
        result.kv_heads > 32 || result.width <= 0 || result.width > 256 ||
        result.repeats <= 0 || result.repeats > 32 || result.warmup < 0 ||
        result.repetitions <= 0) {
        throw std::invalid_argument("BF16 repeat options are outside contract");
    }
    return result;
}

double percentile(std::vector<double> values, double fraction) {
    std::sort(values.begin(), values.end());
    const auto index = static_cast<std::size_t>(
        std::ceil(fraction * static_cast<double>(values.size()))) - 1U;
    return values[std::min(index, values.size() - 1U)];
}

microllm::Tensor execute(const microllm::Tensor& input, const Options& options) {
    return options.fused
               ? microllm::ops::repeat_interleave_bf16_to_float(
                     input, 2, options.repeats)
               : microllm::ops::repeat_interleave(
                     microllm::ops::cast(
                         input, microllm::DType::Float32), 2,
                     options.repeats);
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const auto options = parse(argc, argv);
        if (microllm::runtime::hip_device_count() == 0) {
            throw std::runtime_error("BF16 repeat benchmark requires HIP");
        }
        const auto elements = options.batch * options.sequence *
                              options.kv_heads * options.width;
        std::vector<float> values(static_cast<std::size_t>(elements));
        for (std::size_t index = 0; index < values.size(); ++index) {
            values[index] =
                static_cast<float>(static_cast<int>(index % 97U) - 48) /
                64.0F;
        }
        const auto device = microllm::Device::hip(0);
        const auto input = microllm::Tensor::from_vector(
            values,
            {options.batch, options.sequence, options.kv_heads, options.width},
            microllm::DType::BFloat16).to(device);
        auto control = options;
        control.fused = false;
        const auto reference = execute(input, control).to_vector();
        const auto checked = execute(input, options).to_vector();
        if (checked != reference) {
            throw std::runtime_error("BF16 repeat complete-output gate failed");
        }
        microllm::Tensor output;
        for (int iteration = 0; iteration < options.warmup; ++iteration) {
            output = execute(input, options);
        }
        microllm::runtime::synchronize(device);
        microllm::runtime::Event start(device);
        microllm::runtime::Event finish(device);
        std::vector<double> event_times;
        std::vector<double> wall_times;
        microllm::runtime::reset_transfer_stats();
        for (int iteration = 0; iteration < options.repetitions; ++iteration) {
            const auto wall_start = std::chrono::steady_clock::now();
            start.record_default_stream();
            output = execute(input, options);
            finish.record_default_stream();
            finish.synchronize();
            const auto wall_finish = std::chrono::steady_clock::now();
            event_times.push_back(finish.elapsed_ms_since(start));
            wall_times.push_back(std::chrono::duration<double, std::milli>(
                wall_finish - wall_start).count());
        }
        const auto transfers = microllm::runtime::transfer_stats();
        if (transfers.host_to_device_calls != 0 ||
            transfers.device_to_host_calls != 0) {
            throw std::runtime_error(
                "BF16 repeat timing transferred payloads h2d=" +
                std::to_string(transfers.host_to_device_calls) +
                " d2h=" + std::to_string(transfers.device_to_host_calls));
        }
        const auto info = microllm::runtime::device_info(device);
        std::cout << std::setprecision(9)
                  << "{\"schema_version\":1,\"status\":\"pass\""
                  << ",\"op\":\"bf16_repeat_interleave\""
                  << ",\"architecture\":\"" << info.architecture << "\""
                  << ",\"batch\":" << options.batch
                  << ",\"sequence\":" << options.sequence
                  << ",\"kv_heads\":" << options.kv_heads
                  << ",\"width\":" << options.width
                  << ",\"repeats\":" << options.repeats
                  << ",\"fused\":" << (options.fused ? "true" : "false")
                  << ",\"complete_output_elements\":" << checked.size()
                  << ",\"event_ms_p50\":" << percentile(event_times, 0.50)
                  << ",\"event_ms_p95\":" << percentile(event_times, 0.95)
                  << ",\"wall_ms_p50\":" << percentile(wall_times, 0.50)
                  << ",\"wall_ms_p95\":" << percentile(wall_times, 0.95)
                  << ",\"host_to_device_calls\":"
                  << transfers.host_to_device_calls
                  << ",\"device_to_host_calls\":"
                  << transfers.device_to_host_calls << "}\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "benchmark_bf16_repeat: " << error.what() << '\n';
        return 2;
    }
}
