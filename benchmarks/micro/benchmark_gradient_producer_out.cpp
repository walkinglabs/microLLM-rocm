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

#include <microllm/core/tensor.h>
#include <microllm/ops/ops.h>
#include <microllm/runtime/memory.h>
#include <microllm/runtime/runtime.h>

namespace {

struct Options {
    std::int64_t rows = 32;
    std::int64_t hidden = 384;
    std::int64_t width = 8192;
    int warmup = 5;
    int repetitions = 40;
    std::string order = "allocating-first";
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
        if (name == "--rows") result.rows = integer(argv[index + 1], "rows");
        else if (name == "--hidden") {
            result.hidden = integer(argv[index + 1], "hidden");
        } else if (name == "--width") {
            result.width = integer(argv[index + 1], "width");
        } else if (name == "--warmup") {
            result.warmup = static_cast<int>(integer(argv[index + 1], "warmup"));
        } else if (name == "--repetitions") {
            result.repetitions = static_cast<int>(
                integer(argv[index + 1], "repetitions"));
        } else if (name == "--order") {
            result.order = argv[index + 1];
        } else {
            throw std::invalid_argument("unknown option: " + std::string(name));
        }
    }
    if (result.rows <= 0 || result.rows > 4096 || result.hidden <= 0 ||
        result.hidden > 16384 || result.width <= 0 || result.width > 16384 ||
        result.warmup < 0 || result.repetitions <= 0 ||
        (result.order != "allocating-first" &&
         result.order != "direct-first")) {
        throw std::invalid_argument("gradient producer benchmark options are invalid");
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

struct Error {
    float maximum = 0.0F;
    double rms = 0.0;
};

struct Measured {
    Timing timing;
    double allocation_calls_per_invocation = 0.0;
};

Error compare(const std::vector<float>& expected,
              const std::vector<float>& actual) {
    if (expected.size() != actual.size()) {
        throw std::runtime_error("gradient producer output size changed");
    }
    Error error;
    double squared = 0.0;
    for (std::size_t index = 0; index < expected.size(); ++index) {
        if (!std::isfinite(actual[index])) {
            throw std::runtime_error("gradient producer output is non-finite");
        }
        const auto difference = std::abs(expected[index] - actual[index]);
        error.maximum = std::max(error.maximum, difference);
        squared += static_cast<double>(difference) * difference;
    }
    error.rms = expected.empty()
                    ? 0.0
                    : std::sqrt(squared / static_cast<double>(expected.size()));
    return error;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const auto command = options(argc, argv);
        if (microllm::runtime::hip_device_count() == 0) {
            throw std::runtime_error(
                "gradient producer benchmark requires a visible HIP GPU");
        }
        const auto device = microllm::Device::hip(0);
        std::vector<float> input_values(
            static_cast<std::size_t>(command.rows * command.hidden));
        std::vector<float> gradient_values(
            static_cast<std::size_t>(command.rows * command.width));
        for (std::size_t index = 0; index < input_values.size(); ++index) {
            input_values[index] = static_cast<float>(
                static_cast<int>((index * 17U + 3U) % 53U) - 26) / 257.0F;
        }
        for (std::size_t index = 0; index < gradient_values.size(); ++index) {
            gradient_values[index] = static_cast<float>(
                static_cast<int>((index * 13U + 5U) % 47U) - 23) / 263.0F;
        }
        const auto input = microllm::Tensor::from_vector(
            input_values, {command.rows, command.hidden}).to(device);
        const auto output_gradient = microllm::Tensor::from_vector(
            gradient_values, {command.rows, command.width}).to(device);
        microllm::Tensor allocating_target(
            {command.hidden, command.width}, microllm::DType::Float32, device);
        microllm::Tensor direct_target(
            {command.hidden, command.width}, microllm::DType::Float32, device);
        const microllm::ops::OpContext context{
            .mode = microllm::ops::OpMode::Training};
        const auto allocating = [&] {
            const auto produced = microllm::ops::matmul_with_implementation(
                input, output_gradient,
                microllm::ops::MatmulImplementation::HipBLASLt,
                true, false, context);
            microllm::ops::add_in_place_(allocating_target, produced, context);
        };
        const auto direct = [&] {
            microllm::ops::matmul_weight_gradient_out_(
                direct_target, input, output_gradient,
                microllm::ops::MatmulImplementation::HipBLASLt, context);
        };

        microllm::ops::fill_(allocating_target, 0.0F, context);
        microllm::ops::fill_(direct_target, 0.0F, context);
        allocating();
        direct();
        microllm::runtime::synchronize(device);
        const auto error = compare(
            allocating_target.to_vector(), direct_target.to_vector());
        if (error.maximum != 0.0F) {
            throw std::runtime_error(
                "allocating and caller-owned gradient producers differ");
        }

        microllm::runtime::enable_hip_caching_allocator(device);
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
        const auto invocations = static_cast<double>(
            command.warmup + command.repetitions);
        const auto measure = [&](const auto& operation) {
            const auto before = microllm::runtime::allocation_stats(device);
            const auto timing = time(operation);
            const auto after = microllm::runtime::allocation_stats(device);
            return Measured{
                timing,
                static_cast<double>(after.allocation_calls -
                                    before.allocation_calls) /
                    invocations};
        };
        Measured allocating_measurement;
        Measured direct_measurement;
        if (command.order == "direct-first") {
            direct_measurement = measure(direct);
            allocating_measurement = measure(allocating);
        } else {
            allocating_measurement = measure(allocating);
            direct_measurement = measure(direct);
        }
        const auto& allocating_time = allocating_measurement.timing;
        const auto& direct_time = direct_measurement.timing;
        const auto info = microllm::runtime::device_info(device);
        std::cout << std::setprecision(12)
                  << "{\"schema_version\":1,\"status\":\"pass\""
                  << ",\"record_type\":\"gradient_producer_out_probe\""
                  << ",\"device_name\":\"" << info.name << "\""
                  << ",\"architecture\":\"" << info.architecture << "\""
                  << ",\"rows\":" << command.rows
                  << ",\"hidden\":" << command.hidden
                  << ",\"width\":" << command.width
                  << ",\"warmup\":" << command.warmup
                  << ",\"repetitions\":" << command.repetitions
                  << ",\"order\":\"" << command.order << "\""
                  << ",\"complete_output_elements\":"
                  << command.hidden * command.width
                  << ",\"complete_output_max_error\":" << error.maximum
                  << ",\"complete_output_rms_error\":" << error.rms
                  << ",\"allocating_event_ms_p50\":" << allocating_time.event_p50
                  << ",\"allocating_event_ms_p95\":" << allocating_time.event_p95
                  << ",\"direct_event_ms_p50\":" << direct_time.event_p50
                  << ",\"direct_event_ms_p95\":" << direct_time.event_p95
                  << ",\"allocating_wall_ms_p50\":" << allocating_time.wall_p50
                  << ",\"allocating_wall_ms_p95\":" << allocating_time.wall_p95
                  << ",\"direct_wall_ms_p50\":" << direct_time.wall_p50
                  << ",\"direct_wall_ms_p95\":" << direct_time.wall_p95
                  << ",\"event_speedup\":"
                  << allocating_time.event_p50 / direct_time.event_p50
                  << ",\"wall_speedup\":"
                  << allocating_time.wall_p50 / direct_time.wall_p50
                  << ",\"allocating_calls_per_invocation\":"
                  << allocating_measurement.allocation_calls_per_invocation
                  << ",\"direct_calls_per_invocation\":"
                  << direct_measurement.allocation_calls_per_invocation
                  << "}\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "microllm_bench_gradient_producer_out: "
                  << error.what() << '\n';
        return 1;
    }
}
