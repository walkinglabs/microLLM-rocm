#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include <microllm/base/low_precision.h>
#include <microllm/ops/ops.h>
#include <microllm/runtime/runtime.h>

namespace {

struct Options {
    std::int64_t elements = 1 << 20;
    bool mirror = true;
    std::string implementation = "auto";
    std::string moment_precision = "fp32";
    int warmup = 5;
    int repetitions = 20;
};

std::int64_t integer(const char* value, const char* name) {
    char* end = nullptr;
    const auto parsed = std::strtoll(value, &end, 10);
    if (end == value || *end != '\0') throw std::invalid_argument(std::string("invalid ") + name);
    return parsed;
}

Options options(int argc, char** argv) {
    Options result;
    for (int index = 1; index < argc; index += 2) {
        if (index + 1 >= argc) throw std::invalid_argument("benchmark option is missing a value");
        const std::string_view name(argv[index]);
        if (name == "--elements") result.elements = integer(argv[index + 1], "elements");
        else if (name == "--mirror") {
            const std::string_view value(argv[index + 1]);
            if (value != "true" && value != "false") {
                throw std::invalid_argument("mirror must be true or false");
            }
            result.mirror = value == "true";
        } else if (name == "--implementation") result.implementation = argv[index + 1];
        else if (name == "--moment-precision") result.moment_precision = argv[index + 1];
        else if (name == "--warmup") result.warmup = static_cast<int>(integer(argv[index + 1], "warmup"));
        else if (name == "--repetitions") {
            result.repetitions = static_cast<int>(integer(argv[index + 1], "repetitions"));
        } else throw std::invalid_argument("unknown benchmark option: " + std::string(name));
    }
    if (result.elements <= 0 || result.elements > 300000000 ||
        result.warmup < 0 || result.repetitions <= 0) {
        throw std::invalid_argument("elements/warmup/repetitions are outside safe ranges");
    }
    if (result.implementation != "auto" && result.implementation != "scalar" &&
        result.implementation != "vectorized") {
        throw std::invalid_argument("implementation must be auto, scalar, or vectorized");
    }
    if (result.moment_precision != "fp32" &&
        result.moment_precision != "bf16") {
        throw std::invalid_argument("moment precision must be fp32 or bf16");
    }
    if (result.moment_precision == "bf16" &&
        result.implementation == "vectorized") {
        throw std::invalid_argument(
            "BF16 moment benchmark has one explicit scalar implementation");
    }
    return result;
}

microllm::ops::AdamWImplementation implementation(const std::string& value) {
    if (value == "scalar") return microllm::ops::AdamWImplementation::Scalar;
    if (value == "vectorized") return microllm::ops::AdamWImplementation::Vectorized;
    return microllm::ops::AdamWImplementation::Auto;
}

struct Summary {
    double minimum;
    double mean;
    double maximum;
};

Summary summarize(const std::vector<double>& values) {
    return {*std::min_element(values.begin(), values.end()),
            std::accumulate(values.begin(), values.end(), 0.0) /
                static_cast<double>(values.size()),
            *std::max_element(values.begin(), values.end())};
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const auto settings = options(argc, argv);
        if (microllm::runtime::hip_device_count() == 0) {
            throw std::runtime_error("AdamW benchmark requires a visible HIP device");
        }
        const auto device = microllm::Device::hip(0);
        microllm::Tensor parameter({settings.elements}, microllm::DType::Float32, device);
        microllm::Tensor gradient({settings.elements}, microllm::DType::Float32, device);
        const auto moment_dtype = settings.moment_precision == "bf16"
                                      ? microllm::DType::BFloat16
                                      : microllm::DType::Float32;
        microllm::Tensor first({settings.elements}, moment_dtype, device);
        microllm::Tensor second({settings.elements}, moment_dtype, device);
        microllm::Tensor mirror;
        microllm::ops::fill_(parameter, 1.25F);
        microllm::ops::fill_(gradient, 0.5F);
        microllm::ops::fill_(first, 0.0F);
        microllm::ops::fill_(second, 0.0F);
        if (settings.mirror) {
            mirror = microllm::Tensor({settings.elements}, microllm::DType::BFloat16, device);
            microllm::ops::fill_(mirror, 0.0F);
        }
        microllm::runtime::synchronize(device);
        microllm::runtime::Stream stream(device);
        microllm::runtime::Event start(device);
        microllm::runtime::Event finish(device);
        const microllm::ops::OpContext context{&stream, nullptr, 0};
        constexpr float learning_rate = 0.01F;
        constexpr float beta1 = 0.9F;
        constexpr float beta2 = 0.99F;
        constexpr float epsilon = 1.0e-8F;
        constexpr float weight_decay = 0.1F;
        constexpr float first_correction = 0.1F;
        constexpr float second_correction = 0.01F;
        const auto selected = implementation(settings.implementation);
        const auto update = [&]() {
            if (settings.moment_precision == "bf16") {
                microllm::ops::adamw_update_bf16_moments_(
                    parameter, gradient, first, second,
                    settings.mirror ? &mirror : nullptr, learning_rate, beta1,
                    beta2, epsilon, weight_decay, first_correction,
                    second_correction, context);
                return;
            }
            if (settings.mirror) {
                microllm::ops::adamw_update_bf16_mirror_(
                    parameter, gradient, first, second, mirror,
                    learning_rate, beta1, beta2, epsilon, weight_decay,
                    first_correction, second_correction, context, selected);
            } else {
                microllm::ops::adamw_update_(
                    parameter, gradient, first, second,
                    learning_rate, beta1, beta2, epsilon, weight_decay,
                    first_correction, second_correction, context, selected);
            }
        };
        for (int index = 0; index < settings.warmup; ++index) update();
        stream.synchronize();
        std::vector<double> timings;
        timings.reserve(static_cast<std::size_t>(settings.repetitions));
        for (int index = 0; index < settings.repetitions; ++index) {
            start.record(stream);
            update();
            finish.record(stream);
            finish.synchronize();
            timings.push_back(finish.elapsed_ms_since(start));
        }
        const auto timing = summarize(timings);

        float expected_parameter = 1.25F;
        float expected_first = 0.0F;
        float expected_second = 0.0F;
        const auto updates = settings.warmup + settings.repetitions;
        for (int index = 0; index < updates; ++index) {
            expected_first = beta1 * expected_first + (1.0F - beta1) * 0.5F;
            expected_second = beta2 * expected_second + (1.0F - beta2) * 0.25F;
            if (settings.moment_precision == "bf16") {
                expected_first = static_cast<float>(microllm::BFloat16(expected_first));
                expected_second = static_cast<float>(microllm::BFloat16(expected_second));
            }
            expected_parameter *= 1.0F - learning_rate * weight_decay;
            expected_parameter -= learning_rate * (expected_first / first_correction) /
                                  (std::sqrt(expected_second / second_correction) + epsilon);
        }
        const auto first_parameter = parameter.slice(0, 0, 1).to_vector()[0];
        const auto last_parameter = parameter.slice(
            0, settings.elements - 1, settings.elements).to_vector()[0];
        const auto first_value = first.slice(0, 0, 1).to_vector()[0];
        const auto second_value = second.slice(0, 0, 1).to_vector()[0];
        float maximum_error = std::max({std::abs(first_parameter - expected_parameter),
                                        std::abs(last_parameter - expected_parameter),
                                        std::abs(first_value - expected_first),
                                        std::abs(second_value - expected_second)});
        if (settings.mirror) {
            const auto mirror_value = mirror.slice(0, settings.elements - 1,
                                                   settings.elements).to_vector()[0];
            const auto rounded = static_cast<float>(microllm::BFloat16(expected_parameter));
            maximum_error = std::max(maximum_error, std::abs(mirror_value - rounded));
        }
        const auto bytes_per_element =
            (settings.moment_precision == "bf16" ? 20.0 : 28.0) +
            (settings.mirror ? 2.0 : 0.0);
        const auto bandwidth_gb_s = static_cast<double>(settings.elements) *
                                    bytes_per_element / (timing.mean * 1.0e6);
        const auto info = microllm::runtime::device_info(device);
        std::cout << std::setprecision(9)
                  << "{\"schema_version\":1,\"op\":\"adamw\""
                  << ",\"device\":\"hip\",\"device_name\":\"" << info.name << "\""
                  << ",\"architecture\":\"" << info.architecture << "\""
                  << ",\"implementation\":\"" << settings.implementation << "\""
                  << ",\"moment_precision\":\"" << settings.moment_precision << "\""
                  << ",\"elements\":" << settings.elements
                  << ",\"bf16_mirror\":" << (settings.mirror ? "true" : "false")
                  << ",\"warmup\":" << settings.warmup
                  << ",\"repetitions\":" << settings.repetitions
                  << ",\"kernel_ms_min\":" << timing.minimum
                  << ",\"kernel_ms_mean\":" << timing.mean
                  << ",\"kernel_ms_max\":" << timing.maximum
                  << ",\"effective_bandwidth_gb_s\":" << bandwidth_gb_s
                  << ",\"sample_maximum_absolute_error\":" << maximum_error
                  << "}\n";
        return maximum_error <= 2.0e-5F ? 0 : 1;
    } catch (const std::exception& error) {
        std::cerr << "microllm_bench_adamw: " << error.what() << '\n';
        return 1;
    }
}
