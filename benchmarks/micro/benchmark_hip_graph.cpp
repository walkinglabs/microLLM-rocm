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
#include <microllm/ops/low_level.h>
#include <microllm/ops/ops.h>
#include <microllm/runtime/diagnostics.h>
#include <microllm/runtime/runtime.h>

namespace {

struct Options {
    std::string mode = "eager";
    std::int64_t nodes = 128;
    std::int64_t elements = 1;
    int warmup = 5;
    int repetitions = 20;
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
    if (result.mode != "eager" && result.mode != "graph") {
        throw std::invalid_argument("--mode must be eager or graph");
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
        auto state = microllm::Tensor::from_vector(
            std::vector<float>(static_cast<std::size_t>(options.elements), 0.0F),
            {options.elements}).to(gpu);
        const auto source = microllm::Tensor::from_vector(
            std::vector<float>(static_cast<std::size_t>(options.elements), 1.0F),
            {options.elements}).to(gpu);
        microllm::runtime::Stream stream(gpu);
        microllm::ops::OpContext context;
        context.stream = &stream;
        const auto work = [&] {
            microllm::ops::fill_(state, 0.0F, context);
            for (std::int64_t node = 0; node < options.nodes; ++node) {
                microllm::ops::add_out(
                    state.view(), std::as_const(state).view(), source.view(), context);
            }
        };

        microllm::runtime::HipGraphExecutable graph;
        double setup_ms = 0.0;
        if (options.mode == "graph") {
            const auto setup_start = std::chrono::steady_clock::now();
            graph = microllm::runtime::HipGraphExecutable::capture(stream, work);
            const auto setup_finish = std::chrono::steady_clock::now();
            setup_ms = std::chrono::duration<double, std::milli>(
                           setup_finish - setup_start).count();
            if (graph.node_count() !=
                static_cast<std::size_t>(options.nodes + 1)) {
                throw std::runtime_error("captured graph node count is incomplete");
            }
        }
        const auto run = [&] {
            if (options.mode == "graph") graph.launch(stream);
            else work();
        };

        for (int iteration = 0; iteration < options.warmup; ++iteration) run();
        stream.synchronize();
        microllm::runtime::reset_transfer_stats();
        microllm::runtime::Event start(gpu);
        microllm::runtime::Event finish(gpu);
        std::vector<double> event_ms;
        std::vector<double> wall_ms;
        event_ms.reserve(static_cast<std::size_t>(options.repetitions));
        wall_ms.reserve(static_cast<std::size_t>(options.repetitions));
        for (int iteration = 0; iteration < options.repetitions; ++iteration) {
            const auto wall_start = std::chrono::steady_clock::now();
            start.record(stream);
            run();
            finish.record(stream);
            finish.synchronize();
            const auto wall_finish = std::chrono::steady_clock::now();
            event_ms.push_back(static_cast<double>(finish.elapsed_ms_since(start)));
            wall_ms.push_back(std::chrono::duration<double, std::milli>(
                                  wall_finish - wall_start).count());
        }
        const auto transfers = microllm::runtime::transfer_stats();
        const auto output = state.to_vector();
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
                  << ",\"device\":\"" << gpu.str() << "\""
                  << ",\"device_name\":\"" << info.name << "\""
                  << ",\"architecture\":\"" << info.architecture << "\""
                  << ",\"nodes\":" << options.nodes
                  << ",\"captured_nodes\":"
                  << (graph.defined() ? graph.node_count() : 0U)
                  << ",\"elements\":" << options.elements
                  << ",\"warmup\":" << options.warmup
                  << ",\"repetitions\":" << options.repetitions
                  << ",\"setup_ms\":" << setup_ms
                  << ",\"event_median_ms\":" << percentile(event_ms, 0.5)
                  << ",\"event_p95_ms\":" << percentile(event_ms, 0.95)
                  << ",\"wall_median_ms\":" << percentile(wall_ms, 0.5)
                  << ",\"wall_p95_ms\":" << percentile(wall_ms, 0.95)
                  << ",\"maximum_absolute_error\":" << maximum_error
                  << ",\"host_to_device_calls\":"
                  << transfers.host_to_device_calls
                  << ",\"device_to_host_calls\":"
                  << transfers.device_to_host_calls
                  << ",\"device_to_device_calls\":"
                  << transfers.device_to_device_calls << "}\n";
        return pass ? EXIT_SUCCESS : EXIT_FAILURE;
    } catch (const std::exception& error) {
        std::cerr << "microllm_bench_hip_graph: " << error.what() << '\n';
        return EXIT_FAILURE;
    }
}
