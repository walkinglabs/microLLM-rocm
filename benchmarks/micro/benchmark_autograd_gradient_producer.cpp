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

#include <microllm/autograd/autograd.h>
#include <microllm/autograd/diagnostics.h>
#include <microllm/core/tensor.h>
#include <microllm/runtime/memory.h>
#include <microllm/runtime/runtime.h>

namespace {

struct Options {
    std::int64_t rows = 32;
    std::int64_t hidden = 384;
    std::int64_t width = 8192;
    int warmup = 5;
    int repetitions = 40;
    std::string order = "baseline-first";
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
        (result.order != "baseline-first" &&
         result.order != "direct-first")) {
        throw std::invalid_argument("Autograd producer benchmark options are invalid");
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

struct Measured {
    Timing timing;
    double allocation_calls_per_invocation = 0.0;
};

}  // namespace

int main(int argc, char** argv) {
    try {
        const auto command = options(argc, argv);
        if (microllm::runtime::hip_device_count() == 0) {
            throw std::runtime_error(
                "Autograd producer benchmark requires a visible HIP GPU");
        }
        const auto device = microllm::Device::hip(0);
        std::vector<float> input_values(
            static_cast<std::size_t>(command.rows * command.hidden));
        std::vector<float> weight_values(
            static_cast<std::size_t>(command.hidden * command.width));
        for (std::size_t index = 0; index < input_values.size(); ++index) {
            input_values[index] = static_cast<float>(
                static_cast<int>((index * 17U + 3U) % 53U) - 26) / 257.0F;
        }
        for (std::size_t index = 0; index < weight_values.size(); ++index) {
            weight_values[index] = static_cast<float>(
                static_cast<int>((index * 13U + 5U) % 47U) - 23) / 263.0F;
        }
        const microllm::autograd::Value input(
            microllm::Tensor::from_vector(
                input_values, {command.rows, command.hidden}).to(device));
        const auto weight_data = microllm::Tensor::from_vector(
            weight_values, {command.hidden, command.width}).to(device);
        microllm::autograd::Value baseline_weight(weight_data, true);
        microllm::autograd::Value direct_weight(weight_data, true);
        microllm::Tensor direct_target(
            {command.hidden, command.width}, microllm::DType::Float32, device);
        const auto* direct_address = direct_target.data();
        const auto baseline_loss = microllm::autograd::sum(
            microllm::autograd::matmul(input, baseline_weight));
        const auto direct_loss = microllm::autograd::sum(
            microllm::autograd::matmul(input, direct_weight));

        const auto baseline = [&] {
            baseline_weight.zero_grad();
            baseline_loss.backward();
        };
        const auto direct = [&] {
            direct_weight.set_overwrite_grad_accumulation_target(direct_target);
            direct_loss.backward();
        };

        microllm::autograd::enable_direct_weight_gradient_producer(false);
        baseline();
        microllm::autograd::enable_direct_weight_gradient_producer(true);
        direct();
        microllm::runtime::synchronize(device);
        microllm::autograd::enable_direct_weight_gradient_producer(false);
        const auto expected = baseline_weight.grad().to_vector();
        const auto actual = direct_weight.grad().to_vector();
        if (expected != actual || direct_weight.grad().data() != direct_address) {
            throw std::runtime_error(
                "scoped Autograd producer changed gradient values or address");
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
        const auto measure = [&](const auto& operation, bool enabled) {
            microllm::autograd::enable_direct_weight_gradient_producer(enabled);
            const auto before = microllm::runtime::allocation_stats(device);
            const auto timing = time(operation);
            const auto after = microllm::runtime::allocation_stats(device);
            return Measured{
                timing,
                static_cast<double>(after.allocation_calls -
                                    before.allocation_calls) /
                    invocations};
        };
        Measured baseline_measurement;
        Measured direct_measurement;
        microllm::autograd::reset_direct_weight_gradient_producer_calls();
        if (command.order == "direct-first") {
            direct_measurement = measure(direct, true);
            baseline_measurement = measure(baseline, false);
        } else {
            baseline_measurement = measure(baseline, false);
            direct_measurement = measure(direct, true);
        }
        microllm::autograd::enable_direct_weight_gradient_producer(false);
        const auto direct_calls =
            microllm::autograd::direct_weight_gradient_producer_calls();
        const auto& baseline_time = baseline_measurement.timing;
        const auto& direct_time = direct_measurement.timing;
        const auto info = microllm::runtime::device_info(device);
        std::cout << std::setprecision(12)
                  << "{\"schema_version\":1,\"status\":\"pass\""
                  << ",\"record_type\":\"autograd_gradient_producer_probe\""
                  << ",\"device_name\":\"" << info.name << "\""
                  << ",\"architecture\":\"" << info.architecture << "\""
                  << ",\"rows\":" << command.rows
                  << ",\"hidden\":" << command.hidden
                  << ",\"width\":" << command.width
                  << ",\"warmup\":" << command.warmup
                  << ",\"repetitions\":" << command.repetitions
                  << ",\"order\":\"" << command.order << "\""
                  << ",\"complete_gradient_elements\":"
                  << command.hidden * command.width
                  << ",\"complete_gradient_exact\":true"
                  << ",\"target_address_preserved\":true"
                  << ",\"direct_dispatches_per_invocation\":"
                  << static_cast<double>(direct_calls) / invocations
                  << ",\"baseline_event_ms_p50\":" << baseline_time.event_p50
                  << ",\"baseline_event_ms_p95\":" << baseline_time.event_p95
                  << ",\"direct_event_ms_p50\":" << direct_time.event_p50
                  << ",\"direct_event_ms_p95\":" << direct_time.event_p95
                  << ",\"baseline_wall_ms_p50\":" << baseline_time.wall_p50
                  << ",\"baseline_wall_ms_p95\":" << baseline_time.wall_p95
                  << ",\"direct_wall_ms_p50\":" << direct_time.wall_p50
                  << ",\"direct_wall_ms_p95\":" << direct_time.wall_p95
                  << ",\"event_speedup\":"
                  << baseline_time.event_p50 / direct_time.event_p50
                  << ",\"wall_speedup\":"
                  << baseline_time.wall_p50 / direct_time.wall_p50
                  << ",\"baseline_allocation_calls_per_invocation\":"
                  << baseline_measurement.allocation_calls_per_invocation
                  << ",\"direct_allocation_calls_per_invocation\":"
                  << direct_measurement.allocation_calls_per_invocation
                  << "}\n";
        return 0;
    } catch (const std::exception& error) {
        microllm::autograd::enable_direct_weight_gradient_producer(false);
        std::cerr << "microllm_bench_autograd_gradient_producer: "
                  << error.what() << '\n';
        return 1;
    }
}
