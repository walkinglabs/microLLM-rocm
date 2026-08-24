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
    std::int64_t heads = 14;
    std::int64_t sequence = 512;
    bool threads128 = false;
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
        if (name == "--heads") {
            result.heads = integer(argv[index + 1], "heads");
        } else if (name == "--sequence") {
            result.sequence = integer(argv[index + 1], "sequence");
        } else if (name == "--threads-128") {
            result.threads128 = boolean(argv[index + 1], "threads-128");
        } else if (name == "--warmup") {
            result.warmup = static_cast<int>(integer(argv[index + 1], "warmup"));
        } else if (name == "--repetitions") {
            result.repetitions = static_cast<int>(
                integer(argv[index + 1], "repetitions"));
        } else {
            throw std::invalid_argument("unknown option: " + std::string(name));
        }
    }
    if (result.heads <= 0 || result.heads > 256 || result.sequence < 256 ||
        result.sequence > 1024 || result.warmup < 0 ||
        result.repetitions <= 0) {
        throw std::invalid_argument(
            "causal softmax options are outside the measured contract");
    }
    return result;
}

double percentile(std::vector<double> values, double fraction) {
    std::sort(values.begin(), values.end());
    const auto index = static_cast<std::size_t>(
        std::ceil(fraction * static_cast<double>(values.size()))) - 1U;
    return values[std::min(index, values.size() - 1U)];
}

microllm::Tensor execute(const microllm::Tensor& scores, bool threads128) {
    return microllm::ops::causal_softmax_with_implementation(
        scores, threads128
                    ? microllm::ops::CausalSoftmaxImplementation::Rows128
                    : microllm::ops::CausalSoftmaxImplementation::Auto);
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const auto options = parse(argc, argv);
        if (microllm::runtime::hip_device_count() == 0) {
            throw std::runtime_error("causal softmax benchmark requires HIP");
        }
        const auto rows = options.heads * options.sequence;
        const auto elements = rows * options.sequence;
        std::vector<float> values(static_cast<std::size_t>(elements));
        for (std::size_t index = 0; index < values.size(); ++index) {
            values[index] =
                static_cast<float>(static_cast<int>(index % 113U) - 56) /
                32.0F;
        }
        const auto device = microllm::Device::hip(0);
        const auto scores = microllm::Tensor::from_vector(
            values, {1, options.heads, options.sequence, options.sequence})
                                .to(device);
        const auto reference = execute(scores, false).to_vector();
        const auto checked = execute(scores, options.threads128).to_vector();
        float maximum_error = 0.0F;
        double squared_error = 0.0;
        bool finite = true;
        for (std::size_t index = 0; index < checked.size(); ++index) {
            finite = finite && std::isfinite(checked[index]);
            const auto difference = std::abs(checked[index] - reference[index]);
            maximum_error = std::max(maximum_error, difference);
            squared_error += static_cast<double>(difference) * difference;
        }
        const auto rms_error = std::sqrt(
            squared_error / static_cast<double>(checked.size()));
        if (!finite || maximum_error > 2.0e-6F || rms_error > 1.0e-7) {
            throw std::runtime_error(
                "causal softmax complete-output gate failed");
        }
        microllm::Tensor output;
        for (int iteration = 0; iteration < options.warmup; ++iteration) {
            output = execute(scores, options.threads128);
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
            output = execute(scores, options.threads128);
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
                "causal softmax timing performed a payload transfer");
        }
        const auto info = microllm::runtime::device_info(device);
        std::cout << std::setprecision(9)
                  << "{\"schema_version\":1,\"status\":\"pass\""
                  << ",\"op\":\"causal_softmax\""
                  << ",\"architecture\":\"" << info.architecture << "\""
                  << ",\"heads\":" << options.heads
                  << ",\"sequence\":" << options.sequence
                  << ",\"rows\":" << rows
                  << ",\"threads\":" << (options.threads128 ? 128 : 256)
                  << ",\"warmup\":" << options.warmup
                  << ",\"repetitions\":" << options.repetitions
                  << ",\"complete_output_elements\":" << checked.size()
                  << ",\"maximum_absolute_error\":" << maximum_error
                  << ",\"rms_error\":" << rms_error
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
        std::cerr << "benchmark_causal_softmax: " << error.what() << '\n';
        return 2;
    }
}
