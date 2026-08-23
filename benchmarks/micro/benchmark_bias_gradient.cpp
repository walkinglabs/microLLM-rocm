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
    std::int64_t rows = 512;
    std::int64_t width = 896;
    std::string implementation = "auto";
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

Options parse(int argc, char** argv) {
    Options result;
    for (int index = 1; index < argc; index += 2) {
        if (index + 1 >= argc) throw std::invalid_argument("option is missing a value");
        const std::string_view name(argv[index]);
        if (name == "--rows") result.rows = integer(argv[index + 1], "rows");
        else if (name == "--width") result.width = integer(argv[index + 1], "width");
        else if (name == "--implementation") result.implementation = argv[index + 1];
        else if (name == "--warmup") {
            result.warmup = static_cast<int>(integer(argv[index + 1], "warmup"));
        } else if (name == "--repetitions") {
            result.repetitions =
                static_cast<int>(integer(argv[index + 1], "repetitions"));
        } else {
            throw std::invalid_argument("unknown option: " + std::string(name));
        }
    }
    if (result.rows <= 0 || result.width <= 0 || result.rows > 65536 ||
        result.width > 1048576 || result.warmup < 0 || result.repetitions <= 0 ||
        (result.implementation != "auto" && result.implementation != "scalar" &&
         result.implementation != "cooperative")) {
        throw std::invalid_argument("bias-gradient options are outside the safe contract");
    }
    return result;
}

microllm::ops::BiasGradientImplementation implementation(const std::string& name) {
    if (name == "scalar") {
        return microllm::ops::BiasGradientImplementation::ScalarColumns;
    }
    if (name == "cooperative") {
        return microllm::ops::BiasGradientImplementation::CooperativeRows;
    }
    return microllm::ops::BiasGradientImplementation::Auto;
}

double percentile(std::vector<double> values, double fraction) {
    std::sort(values.begin(), values.end());
    const auto index = static_cast<std::size_t>(
        std::ceil(fraction * static_cast<double>(values.size()))) - 1U;
    return values[std::min(index, values.size() - 1U)];
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const auto options = parse(argc, argv);
        if (microllm::runtime::hip_device_count() == 0) {
            throw std::runtime_error("bias-gradient benchmark requires a visible HIP device");
        }
        std::vector<float> values(
            static_cast<std::size_t>(options.rows * options.width));
        for (std::size_t index = 0; index < values.size(); ++index) {
            values[index] =
                static_cast<float>(static_cast<int>(index % 29) - 14) / 31.0F;
        }
        const auto cpu = microllm::Tensor::from_vector(
            values, {options.rows, options.width});
        const auto reference = microllm::ops::bias_gradient(cpu).to_vector();
        const auto device = microllm::Device::hip(0);
        const auto input = cpu.to(device);
        const auto selected = implementation(options.implementation);
        const auto checked = microllm::ops::bias_gradient_with_implementation(
            input, selected);
        microllm::runtime::synchronize(device);
        const auto actual = checked.to_vector();
        float maximum_error = 0.0F;
        double squared_error = 0.0;
        bool finite = true;
        for (std::size_t index = 0; index < actual.size(); ++index) {
            finite = finite && std::isfinite(actual[index]);
            const auto difference = std::abs(actual[index] - reference[index]);
            maximum_error = std::max(maximum_error, difference);
            squared_error += static_cast<double>(difference) * difference;
        }
        const auto rms_error = std::sqrt(
            squared_error / static_cast<double>(actual.size()));
        if (!finite || maximum_error > 3.0e-5F || rms_error > 1.0e-5) {
            throw std::runtime_error("bias-gradient complete-output gate failed");
        }

        microllm::Tensor output;
        for (int iteration = 0; iteration < options.warmup; ++iteration) {
            output = microllm::ops::bias_gradient_with_implementation(input, selected);
        }
        microllm::runtime::synchronize(device);
        microllm::runtime::Event start(device);
        microllm::runtime::Event finish(device);
        std::vector<double> event_times;
        std::vector<double> wall_times;
        for (int iteration = 0; iteration < options.repetitions; ++iteration) {
            const auto wall_start = std::chrono::steady_clock::now();
            start.record_default_stream();
            output = microllm::ops::bias_gradient_with_implementation(input, selected);
            finish.record_default_stream();
            finish.synchronize();
            const auto wall_finish = std::chrono::steady_clock::now();
            event_times.push_back(finish.elapsed_ms_since(start));
            wall_times.push_back(std::chrono::duration<double, std::milli>(
                wall_finish - wall_start).count());
        }
        const auto info = microllm::runtime::device_info(device);
        std::cout << std::setprecision(9)
                  << "{\"schema_version\":1,\"status\":\"pass\""
                  << ",\"op\":\"bias_gradient\""
                  << ",\"architecture\":\"" << info.architecture << "\""
                  << ",\"rows\":" << options.rows
                  << ",\"width\":" << options.width
                  << ",\"implementation\":\"" << options.implementation << "\""
                  << ",\"warmup\":" << options.warmup
                  << ",\"repetitions\":" << options.repetitions
                  << ",\"complete_output_elements\":" << actual.size()
                  << ",\"finite\":" << (finite ? "true" : "false")
                  << ",\"maximum_absolute_error\":" << maximum_error
                  << ",\"rms_error\":" << rms_error
                  << ",\"event_ms_p50\":" << percentile(event_times, 0.50)
                  << ",\"event_ms_p95\":" << percentile(event_times, 0.95)
                  << ",\"wall_ms_p50\":" << percentile(wall_times, 0.50)
                  << ",\"wall_ms_p95\":" << percentile(wall_times, 0.95)
                  << "}\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "benchmark_bias_gradient: " << error.what() << '\n';
        return 2;
    }
}
