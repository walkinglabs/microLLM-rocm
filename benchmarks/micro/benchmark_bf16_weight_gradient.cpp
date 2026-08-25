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
    std::int64_t hidden = 64;
    std::int64_t width = 96;
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
        } else {
            throw std::invalid_argument("unknown option: " + std::string(name));
        }
    }
    if (result.rows <= 0 || result.rows > 4096 || result.hidden <= 0 ||
        result.hidden > 16384 || result.width <= 0 || result.width > 16384 ||
        result.warmup < 0 || result.repetitions <= 0) {
        throw std::invalid_argument("BF16 weight-gradient options are invalid");
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
    bool finite = true;
};

Error compare(const std::vector<float>& expected,
              const std::vector<float>& actual) {
    if (expected.size() != actual.size()) {
        throw std::runtime_error("BF16 weight-gradient output size changed");
    }
    Error result;
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
                "BF16 weight-gradient benchmark requires a visible HIP GPU");
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
        const auto host_input = microllm::Tensor::from_vector(
            input_values, {command.rows, command.hidden});
        const auto host_gradient = microllm::Tensor::from_vector(
            gradient_values, {command.rows, command.width});
        const auto rounded_input = host_input.cast(
            microllm::DType::BFloat16).cast(microllm::DType::Float32).to_vector();
        const auto rounded_gradient = host_gradient.cast(
            microllm::DType::BFloat16).cast(microllm::DType::Float32).to_vector();
        const auto input = host_input.to(device);
        const auto gradient = host_gradient.to(device);
        microllm::Tensor baseline_output(
            {command.hidden, command.width}, microllm::DType::Float32, device);
        microllm::Tensor input_transposed_bf16(
            {command.hidden, command.rows}, microllm::DType::BFloat16, device);
        microllm::Tensor gradient_bf16(
            {command.rows, command.width}, microllm::DType::BFloat16, device);
        microllm::Tensor candidate_output(
            {command.hidden, command.width}, microllm::DType::Float32, device);
        microllm::Tensor candidate_fallback(
            {command.hidden, command.width}, microllm::DType::BFloat16, device);
        const microllm::ops::OpContext context{
            .mode = microllm::ops::OpMode::Training};
        const auto baseline = [&] {
            microllm::ops::matmul_out_(
                baseline_output, input, gradient,
                microllm::ops::MatmulImplementation::HipBLASLt,
                true, false, context);
        };
        const auto candidate = [&] {
            microllm::ops::cast_transpose_2d_out_(
                input, input_transposed_bf16, context);
            microllm::ops::cast_out_(gradient, gradient_bf16, context);
            microllm::ops::bf16_matmul_output_out_(
                candidate_output, input_transposed_bf16, gradient_bf16,
                candidate_fallback, context);
        };
        const auto allocating = [&] {
            const auto output = microllm::ops::bf16_weight_gradient(
                input, gradient, context);
            (void)output;
        };
        baseline();
        candidate();
        microllm::runtime::synchronize(device);
        const auto baseline_values = baseline_output.to_vector();
        const auto candidate_values = candidate_output.to_vector();
        const auto fp32_error = compare(baseline_values, candidate_values);
        if (!fp32_error.finite) {
            throw std::runtime_error("BF16 weight-gradient produced non-finite output");
        }
        constexpr std::size_t maximum_samples = 64;
        const auto output_elements = static_cast<std::size_t>(
            command.hidden * command.width);
        const auto sample_count = std::min(maximum_samples, output_elements);
        float sample_maximum = 0.0F;
        double sample_squared = 0.0;
        for (std::size_t sample = 0; sample < sample_count; ++sample) {
            const auto flat = sample * output_elements / sample_count;
            const auto row = flat / static_cast<std::size_t>(command.width);
            const auto column = flat % static_cast<std::size_t>(command.width);
            float reference = 0.0F;
            for (std::int64_t inner = 0; inner < command.rows; ++inner) {
                reference += rounded_input[static_cast<std::size_t>(
                                 inner * command.hidden + row)] *
                             rounded_gradient[static_cast<std::size_t>(
                                 inner * command.width + column)];
            }
            const auto difference = std::abs(reference - candidate_values[flat]);
            sample_maximum = std::max(sample_maximum, difference);
            sample_squared += static_cast<double>(difference) * difference;
        }
        const auto sample_rms = std::sqrt(
            sample_squared / static_cast<double>(sample_count));
        if (!std::isfinite(sample_maximum) || sample_maximum > 2.0e-3F) {
            throw std::runtime_error(
                "BF16 weight-gradient sampled reference gate failed");
        }
        microllm::runtime::enable_hip_caching_allocator(device);
        {
            const auto allocating_output = microllm::ops::bf16_weight_gradient(
                input, gradient, context);
            microllm::runtime::synchronize(device);
            const auto allocating_values = allocating_output.to_vector();
            const auto allocating_error = compare(
                candidate_values, allocating_values);
            if (!allocating_error.finite || allocating_error.maximum != 0.0F) {
                throw std::runtime_error(
                    "allocating and preallocated BF16 weight gradients differ");
            }
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
        const auto allocation_before = microllm::runtime::allocation_stats(device);
        const auto allocating_time = time(allocating);
        const auto allocation_after = microllm::runtime::allocation_stats(device);
        const auto timed_invocations = static_cast<double>(
            command.warmup + command.repetitions);
        const auto info = microllm::runtime::device_info(device);
        std::cout << std::setprecision(12)
                  << "{\"schema_version\":1,\"status\":\"pass\""
                  << ",\"record_type\":\"bf16_weight_gradient_probe\""
                  << ",\"device_name\":\"" << info.name << "\""
                  << ",\"architecture\":\"" << info.architecture << "\""
                  << ",\"rows\":" << command.rows
                  << ",\"hidden\":" << command.hidden
                  << ",\"width\":" << command.width
                  << ",\"warmup\":" << command.warmup
                  << ",\"repetitions\":" << command.repetitions
                  << ",\"candidate_includes_input_cast_transpose\":true"
                  << ",\"candidate_includes_gradient_cast\":true"
                  << ",\"candidate_output_dtype\":\"float32\""
                  << ",\"complete_output_elements\":" << output_elements
                  << ",\"complete_output_finite\":true"
                  << ",\"bf16_reference_samples\":" << sample_count
                  << ",\"bf16_reference_sample_max_error\":" << sample_maximum
                  << ",\"bf16_reference_sample_rms_error\":" << sample_rms
                  << ",\"fp32_baseline_max_error\":" << fp32_error.maximum
                  << ",\"fp32_baseline_rms_error\":" << fp32_error.rms
                  << ",\"baseline_event_ms_p50\":" << baseline_time.event_p50
                  << ",\"baseline_event_ms_p95\":" << baseline_time.event_p95
                  << ",\"candidate_event_ms_p50\":" << candidate_time.event_p50
                  << ",\"candidate_event_ms_p95\":" << candidate_time.event_p95
                  << ",\"baseline_wall_ms_p50\":" << baseline_time.wall_p50
                  << ",\"baseline_wall_ms_p95\":" << baseline_time.wall_p95
                  << ",\"candidate_wall_ms_p50\":" << candidate_time.wall_p50
                  << ",\"candidate_wall_ms_p95\":" << candidate_time.wall_p95
                  << ",\"allocating_event_ms_p50\":"
                  << allocating_time.event_p50
                  << ",\"allocating_event_ms_p95\":"
                  << allocating_time.event_p95
                  << ",\"allocating_wall_ms_p50\":"
                  << allocating_time.wall_p50
                  << ",\"allocating_wall_ms_p95\":"
                  << allocating_time.wall_p95
                  << ",\"preallocated_over_allocating_event_speedup\":"
                  << allocating_time.event_p50 / candidate_time.event_p50
                  << ",\"preallocated_over_allocating_wall_speedup\":"
                  << allocating_time.wall_p50 / candidate_time.wall_p50
                  << ",\"allocating_allocation_calls_per_invocation\":"
                  << static_cast<double>(allocation_after.allocation_calls -
                                         allocation_before.allocation_calls) /
                         timed_invocations
                  << ",\"allocating_backend_allocation_calls_per_invocation\":"
                  << static_cast<double>(allocation_after.backend_allocation_calls -
                                         allocation_before.backend_allocation_calls) /
                         timed_invocations
                  << ",\"allocating_cache_reuse_calls_per_invocation\":"
                  << static_cast<double>(allocation_after.cache_reuse_calls -
                                         allocation_before.cache_reuse_calls) /
                         timed_invocations
                  << ",\"event_speedup\":"
                  << baseline_time.event_p50 / candidate_time.event_p50
                  << "}\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "microllm_bench_bf16_weight_gradient: "
                  << error.what() << '\n';
        return 1;
    }
}
