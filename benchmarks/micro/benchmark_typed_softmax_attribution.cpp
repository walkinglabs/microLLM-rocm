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

#include "kernels.h"

namespace {

struct Options {
    std::int64_t rows = 8;
    std::int64_t width = 4096;
    int warmup = 5;
    int repetitions = 25;
    int samples = 7;
    std::string order = "raw-first";
};

std::int64_t parse_integer(const char* text, const char* name) {
    char* end = nullptr;
    const auto value = std::strtoll(text, &end, 10);
    if (end == text || *end != '\0') {
        throw std::invalid_argument(std::string("invalid ") + name);
    }
    return value;
}

Options parse_options(int argc, char** argv) {
    Options options;
    for (int index = 1; index < argc; index += 2) {
        if (index + 1 >= argc) {
            throw std::invalid_argument("benchmark option is missing a value");
        }
        const std::string_view name(argv[index]);
        if (name == "--rows") {
            options.rows = parse_integer(argv[index + 1], "rows");
        } else if (name == "--width") {
            options.width = parse_integer(argv[index + 1], "width");
        } else if (name == "--warmup") {
            options.warmup =
                static_cast<int>(parse_integer(argv[index + 1], "warmup"));
        } else if (name == "--repetitions") {
            options.repetitions = static_cast<int>(
                parse_integer(argv[index + 1], "repetitions"));
        } else if (name == "--samples") {
            options.samples =
                static_cast<int>(parse_integer(argv[index + 1], "samples"));
        } else if (name == "--order") {
            options.order = argv[index + 1];
        } else {
            throw std::invalid_argument("unknown option: " + std::string(name));
        }
    }
    if (options.rows <= 0 || options.width <= 0 || options.width > 8192 ||
        options.warmup < 0 || options.repetitions <= 0 || options.samples <= 0) {
        throw std::invalid_argument("benchmark dimensions or repetitions are invalid");
    }
    if (options.order != "raw-first" && options.order != "cpp-first") {
        throw std::invalid_argument("order must be raw-first or cpp-first");
    }
    return options;
}

double median(std::vector<double> values) {
    if (values.empty()) throw std::invalid_argument("cannot summarize no samples");
    std::sort(values.begin(), values.end());
    const auto middle = values.size() / 2;
    return values.size() % 2 == 0
               ? (values[middle - 1] + values[middle]) / 2.0
               : values[middle];
}

struct Timing {
    double event_ms = 0.0;
    double wall_ms = 0.0;
};

template <typename Function>
Timing measure(Function&& function, const Options& options,
               const microllm::runtime::Stream& stream,
               const microllm::Device& device) {
    for (int iteration = 0; iteration < options.warmup; ++iteration) function();
    stream.synchronize();
    std::vector<double> event_samples;
    std::vector<double> wall_samples;
    event_samples.reserve(static_cast<std::size_t>(options.samples));
    wall_samples.reserve(static_cast<std::size_t>(options.samples));
    for (int sample = 0; sample < options.samples; ++sample) {
        microllm::runtime::Event start(device);
        microllm::runtime::Event finish(device);
        const auto wall_start = std::chrono::steady_clock::now();
        start.record(stream);
        for (int iteration = 0; iteration < options.repetitions; ++iteration) {
            function();
        }
        finish.record(stream);
        finish.synchronize();
        const auto wall_finish = std::chrono::steady_clock::now();
        event_samples.push_back(
            static_cast<double>(finish.elapsed_ms_since(start)) /
            static_cast<double>(options.repetitions));
        wall_samples.push_back(
            std::chrono::duration<double, std::milli>(wall_finish - wall_start)
                .count() /
            static_cast<double>(options.repetitions));
    }
    return {median(event_samples), median(wall_samples)};
}

float maximum_error(const std::vector<float>& actual,
                    const std::vector<float>& expected) {
    if (actual.size() != expected.size()) {
        throw std::invalid_argument("comparison size mismatch");
    }
    float result = 0.0F;
    for (std::size_t index = 0; index < actual.size(); ++index) {
        result = std::max(result, std::abs(actual[index] - expected[index]));
    }
    return result;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const auto options = parse_options(argc, argv);
        if (microllm::runtime::hip_device_count() == 0) {
            throw std::runtime_error("typed Softmax attribution requires HIP");
        }
        const auto gpu = microllm::Device::hip();
        const auto elements = options.rows * options.width;
        std::vector<float> values(static_cast<std::size_t>(elements));
        for (std::size_t index = 0; index < values.size(); ++index) {
            values[index] =
                (static_cast<float>(index % 251) * 0.03125F) - 2.0F;
        }
        const auto cpu = microllm::Tensor::from_vector(
            values, {options.rows, options.width}, microllm::DType::Float16);
        const auto expected = microllm::ops::softmax(cpu).to_vector();
        const auto input = cpu.to(gpu);
        microllm::Tensor raw_output(input.shape(), input.dtype(), gpu);
        microllm::Tensor cpp_output(input.shape(), input.dtype(), gpu);
        microllm::runtime::Stream stream(gpu);
        const microllm::ops::OpContext context{&stream, nullptr, 0};

        const auto raw = [&] {
            microllm::ops::hip::launch_softmax_typed(
                input.data(), raw_output.data(), input.dtype(), options.rows,
                options.width, stream.native_handle());
        };
        const auto cpp = [&] {
            microllm::ops::softmax_typed_out_(
                cpp_output, input, -1, context);
        };

        Timing raw_timing;
        Timing cpp_timing;
        microllm::runtime::reset_transfer_stats();
        if (options.order == "raw-first") {
            raw_timing = measure(raw, options, stream, gpu);
            cpp_timing = measure(cpp, options, stream, gpu);
        } else {
            cpp_timing = measure(cpp, options, stream, gpu);
            raw_timing = measure(raw, options, stream, gpu);
        }
        raw();
        cpp();
        stream.synchronize();
        const auto transfers = microllm::runtime::transfer_stats();
        const auto raw_error = maximum_error(raw_output.to_vector(), expected);
        const auto cpp_error = maximum_error(cpp_output.to_vector(), expected);
        const auto info = microllm::runtime::device_info(gpu);

        std::cout << std::setprecision(9)
                  << "{\"schema_version\":1"
                  << ",\"record_type\":\"typed_softmax_attribution\""
                  << ",\"engine_version\":\"" << MICROLLM_VERSION << "\""
                  << ",\"architecture\":\"" << info.architecture << "\""
                  << ",\"order\":\"" << options.order << "\""
                  << ",\"rows\":" << options.rows
                  << ",\"width\":" << options.width
                  << ",\"warmup\":" << options.warmup
                  << ",\"repetitions\":" << options.repetitions
                  << ",\"samples\":" << options.samples
                  << ",\"raw_event_ms\":" << raw_timing.event_ms
                  << ",\"raw_wall_ms\":" << raw_timing.wall_ms
                  << ",\"cpp_event_ms\":" << cpp_timing.event_ms
                  << ",\"cpp_wall_ms\":" << cpp_timing.wall_ms
                  << ",\"raw_maximum_error\":" << raw_error
                  << ",\"cpp_maximum_error\":" << cpp_error
                  << ",\"timed_h2d_calls\":" << transfers.host_to_device_calls
                  << ",\"timed_d2h_calls\":" << transfers.device_to_host_calls
                  << ",\"timed_d2d_calls\":" << transfers.device_to_device_calls
                  << "}\n";
        return raw_error <= 5.0e-4F && cpp_error <= 5.0e-4F &&
                       transfers.host_to_device_calls == 0 &&
                       transfers.device_to_host_calls == 0
                   ? 0
                   : 2;
    } catch (const std::exception& error) {
        std::cerr << "benchmark_typed_softmax_attribution: "
                  << error.what() << '\n';
        return 1;
    }
}
