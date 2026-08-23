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
    std::int64_t heads = 14;
    std::int64_t sequence = 512;
    std::int64_t width = 64;
    std::string implementation = "interleaved";
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
        if (name == "--batch") result.batch = integer(argv[index + 1], "batch");
        else if (name == "--heads") result.heads = integer(argv[index + 1], "heads");
        else if (name == "--sequence") {
            result.sequence = integer(argv[index + 1], "sequence");
        } else if (name == "--width") {
            result.width = integer(argv[index + 1], "width");
        } else if (name == "--implementation") {
            result.implementation = argv[index + 1];
        } else if (name == "--warmup") {
            result.warmup = static_cast<int>(integer(argv[index + 1], "warmup"));
        } else if (name == "--repetitions") {
            result.repetitions =
                static_cast<int>(integer(argv[index + 1], "repetitions"));
        } else {
            throw std::invalid_argument("unknown option: " + std::string(name));
        }
    }
    if (result.batch <= 0 || result.heads <= 0 || result.sequence <= 0 ||
        result.width <= 0 || result.batch > 16 || result.heads > 256 ||
        result.sequence > 4096 || result.width > 512 || result.warmup < 0 ||
        result.repetitions <= 0 ||
        (result.implementation != "materialized" &&
         result.implementation != "interleaved")) {
        throw std::invalid_argument("Attention layout options are outside the safe contract");
    }
    return result;
}

microllm::Tensor execute(const Options& options,
                         const microllm::Tensor& probabilities,
                         const microllm::Tensor& value) {
    if (options.implementation == "interleaved") {
        return microllm::ops::attention_probability_value_bthd(
            probabilities, value);
    }
    const auto value_bhtd = value.transpose(1, 2).contiguous();
    return microllm::ops::matmul_with_implementation(
               probabilities, value_bhtd,
               microllm::ops::MatmulImplementation::HipBLASLt)
        .transpose(1, 2).contiguous();
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
        if (microllm::runtime::hip_device_count() == 0 ||
            !microllm::ops::hipblaslt_available()) {
            throw std::runtime_error(
                "Attention layout benchmark requires HIP and hipBLASLt");
        }
        const auto probability_elements = options.batch * options.heads *
                                          options.sequence * options.sequence;
        const auto value_elements = options.batch * options.sequence *
                                    options.heads * options.width;
        std::vector<float> probability_values(
            static_cast<std::size_t>(probability_elements));
        std::vector<float> value_values(static_cast<std::size_t>(value_elements));
        for (std::size_t index = 0; index < probability_values.size(); ++index) {
            probability_values[index] =
                static_cast<float>(static_cast<int>(index % 31) - 15) / 1024.0F;
        }
        for (std::size_t index = 0; index < value_values.size(); ++index) {
            value_values[index] =
                static_cast<float>(static_cast<int>(index % 29) - 14) / 29.0F;
        }
        const auto device = microllm::Device::hip(0);
        const auto probabilities = microllm::Tensor::from_vector(
            probability_values,
            {options.batch, options.heads, options.sequence, options.sequence})
                                       .to(device);
        const auto value = microllm::Tensor::from_vector(
            value_values,
            {options.batch, options.sequence, options.heads, options.width})
                               .to(device);
        auto baseline_options = options;
        baseline_options.implementation = "materialized";
        const auto reference = execute(
            baseline_options, probabilities, value).to_vector();
        const auto checked = execute(options, probabilities, value);
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
        if (!finite || maximum_error > 3.0e-4F || rms_error > 1.0e-5) {
            throw std::runtime_error("Attention layout complete-output gate failed");
        }

        microllm::Tensor output;
        for (int iteration = 0; iteration < options.warmup; ++iteration) {
            output = execute(options, probabilities, value);
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
            output = execute(options, probabilities, value);
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
            throw std::runtime_error("Attention layout timing performed a payload transfer");
        }
        const auto info = microllm::runtime::device_info(device);
        std::cout << std::setprecision(9)
                  << "{\"schema_version\":1,\"status\":\"pass\""
                  << ",\"op\":\"attention_probability_value_bthd\""
                  << ",\"architecture\":\"" << info.architecture << "\""
                  << ",\"batch\":" << options.batch
                  << ",\"heads\":" << options.heads
                  << ",\"sequence\":" << options.sequence
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
                  << ",\"host_to_device_calls\":" << transfers.host_to_device_calls
                  << ",\"device_to_host_calls\":" << transfers.device_to_host_calls
                  << "}\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "benchmark_attention_layout: " << error.what() << '\n';
        return 2;
    }
}
