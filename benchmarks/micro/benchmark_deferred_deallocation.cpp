#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <microllm/core/tensor.h>
#include <microllm/ops/ops.h>
#include <microllm/runtime/diagnostics.h>
#include <microllm/runtime/memory.h>
#include <microllm/runtime/runtime.h>

namespace {

struct Options {
    std::string mode = "immediate_sync";
    std::int64_t nodes = 32;
    std::int64_t elements = 4096;
    int warmup = 3;
    int repetitions = 10;
};

struct ChainResult {
    microllm::Tensor output;
    std::size_t deferred_blocks = 0;
    std::size_t deferred_bytes = 0;
    std::size_t overflow_flushes = 0;
};

Options parse_options(int argc, char** argv) {
    Options result;
    for (int index = 1; index < argc; index += 2) {
        if (index + 1 >= argc) throw std::invalid_argument("missing CLI value");
        const std::string name = argv[index];
        const std::string value = argv[index + 1];
        if (name == "--mode") result.mode = value;
        else if (name == "--nodes") result.nodes = std::stoll(value);
        else if (name == "--elements") result.elements = std::stoll(value);
        else if (name == "--warmup") result.warmup = std::stoi(value);
        else if (name == "--repetitions") result.repetitions = std::stoi(value);
        else throw std::invalid_argument("unknown option: " + name);
    }
    if (result.mode != "immediate_sync" && result.mode != "deferred") {
        throw std::invalid_argument("--mode must be immediate_sync or deferred");
    }
    if (result.nodes <= 0 || result.elements <= 0 || result.warmup < 0 ||
        result.repetitions <= 0) {
        throw std::invalid_argument(
            "nodes/elements/repetitions must be positive and warmup nonnegative");
    }
    return result;
}

double percentile(std::vector<double> values, double fraction) {
    if (values.empty()) throw std::invalid_argument("percentile input is empty");
    std::sort(values.begin(), values.end());
    const auto position = fraction * static_cast<double>(values.size() - 1U);
    const auto lower = static_cast<std::size_t>(std::floor(position));
    const auto upper = static_cast<std::size_t>(std::ceil(position));
    const auto weight = position - static_cast<double>(lower);
    return values[lower] * (1.0 - weight) + values[upper] * weight;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const auto options = parse_options(argc, argv);
        if (microllm::runtime::hip_device_count() == 0) {
            throw std::runtime_error("no visible HIP device");
        }
        const auto gpu = microllm::Device::hip(0);
        const auto base = microllm::Tensor::from_vector(
            std::vector<float>(static_cast<std::size_t>(options.elements), 0.0F),
            {options.elements}).to(gpu);
        const auto source = microllm::Tensor::from_vector(
            std::vector<float>(static_cast<std::size_t>(options.elements), 1.0F),
            {options.elements}).to(gpu);
        microllm::runtime::Stream stream(gpu);
        microllm::ops::OpContext context;
        context.stream = &stream;

        const auto immediate = [&] {
            auto current = base;
            for (std::int64_t node = 0; node < options.nodes; ++node) {
                auto next = microllm::ops::add(current, source, context);
                stream.synchronize();
                current = std::move(next);
            }
            return ChainResult{.output = std::move(current)};
        };
        const auto deferred = [&] {
            microllm::runtime::DeferredHipDeallocationScope lifetime(
                stream, static_cast<std::size_t>(options.nodes + 1));
            auto current = base;
            for (std::int64_t node = 0; node < options.nodes; ++node) {
                current = microllm::ops::add(current, source, context);
            }
            ChainResult result{.output = std::move(current),
                               .deferred_blocks = lifetime.total_deferred_blocks(),
                               .deferred_bytes = lifetime.total_deferred_bytes(),
                               .overflow_flushes = lifetime.overflow_flushes()};
            lifetime.finish();
            return result;
        };
        const auto run = [&] {
            return options.mode == "deferred" ? deferred() : immediate();
        };

        for (int iteration = 0; iteration < options.warmup; ++iteration) {
            (void)run();
        }
        microllm::runtime::reset_transfer_stats();
        microllm::runtime::reset_allocation_peak(gpu);
        std::vector<double> wall_ms;
        wall_ms.reserve(static_cast<std::size_t>(options.repetitions));
        ChainResult last;
        for (int iteration = 0; iteration < options.repetitions; ++iteration) {
            const auto start = std::chrono::steady_clock::now();
            last = run();
            const auto finish = std::chrono::steady_clock::now();
            wall_ms.push_back(std::chrono::duration<double, std::milli>(
                                  finish - start).count());
        }
        const auto transfers = microllm::runtime::transfer_stats();
        const auto allocation = microllm::runtime::allocation_stats(gpu);
        const auto output = last.output.to_vector();
        double maximum_error = 0.0;
        for (const auto value : output) {
            maximum_error = std::max(
                maximum_error,
                std::abs(static_cast<double>(value) -
                         static_cast<double>(options.nodes)));
        }
        const auto pass = maximum_error == 0.0 &&
                          transfers.host_to_device_calls == 0 &&
                          transfers.device_to_host_calls == 0 &&
                          transfers.device_to_device_calls == 0;
        const auto info = microllm::runtime::device_info(gpu);
        std::cout << std::setprecision(9)
                  << "{\"schema_version\":1,\"status\":\""
                  << (pass ? "pass" : "fail") << "\""
                  << ",\"mode\":\"" << options.mode << "\""
                  << ",\"device_name\":\"" << info.name << "\""
                  << ",\"architecture\":\"" << info.architecture << "\""
                  << ",\"nodes\":" << options.nodes
                  << ",\"elements\":" << options.elements
                  << ",\"warmup\":" << options.warmup
                  << ",\"repetitions\":" << options.repetitions
                  << ",\"wall_median_ms\":" << percentile(wall_ms, 0.5)
                  << ",\"wall_p95_ms\":" << percentile(wall_ms, 0.95)
                  << ",\"maximum_absolute_error\":" << maximum_error
                  << ",\"deferred_blocks\":" << last.deferred_blocks
                  << ",\"deferred_bytes\":" << last.deferred_bytes
                  << ",\"overflow_flushes\":" << last.overflow_flushes
                  << ",\"engine_peak_bytes\":" << allocation.peak_bytes
                  << ",\"engine_allocation_calls\":" << allocation.allocation_calls
                  << ",\"engine_backend_allocation_calls\":"
                  << allocation.backend_allocation_calls
                  << ",\"engine_backend_deallocation_calls\":"
                  << allocation.backend_deallocation_calls
                  << ",\"host_to_device_calls\":"
                  << transfers.host_to_device_calls
                  << ",\"device_to_host_calls\":"
                  << transfers.device_to_host_calls
                  << ",\"device_to_device_calls\":"
                  << transfers.device_to_device_calls << "}\n";
        return pass ? EXIT_SUCCESS : EXIT_FAILURE;
    } catch (const std::exception& error) {
        std::cerr << "microllm_bench_deferred_deallocation: " << error.what() << '\n';
        return EXIT_FAILURE;
    }
}
