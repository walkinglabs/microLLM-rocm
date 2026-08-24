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
#include <microllm/runtime/runtime.h>

namespace {

struct Options {
    std::string mode = "eager";
    std::int64_t calls = 8;
    std::int64_t rows = 512;
    std::int64_t inner = 896;
    std::int64_t columns = 896;
    int warmup = 3;
    int repetitions = 10;
};

Options parse_options(int argc, char** argv) {
    Options result;
    for (int index = 1; index < argc; index += 2) {
        if (index + 1 >= argc) throw std::invalid_argument("missing CLI value");
        const std::string name = argv[index];
        const std::string value = argv[index + 1];
        if (name == "--mode") result.mode = value;
        else if (name == "--calls") result.calls = std::stoll(value);
        else if (name == "--rows") result.rows = std::stoll(value);
        else if (name == "--inner") result.inner = std::stoll(value);
        else if (name == "--columns") result.columns = std::stoll(value);
        else if (name == "--warmup") result.warmup = std::stoi(value);
        else if (name == "--repetitions") result.repetitions = std::stoi(value);
        else throw std::invalid_argument("unknown option: " + name);
    }
    if (result.mode != "eager" && result.mode != "graph") {
        throw std::invalid_argument("--mode must be eager or graph");
    }
    if (result.calls <= 0 || result.rows <= 0 || result.inner <= 0 ||
        result.columns <= 0 || result.warmup < 0 || result.repetitions <= 0) {
        throw std::invalid_argument(
            "calls/shapes/repetitions must be positive and warmup nonnegative");
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
        std::vector<float> left_values(
            static_cast<std::size_t>(options.rows * options.inner));
        std::vector<float> right_values(
            static_cast<std::size_t>(options.inner * options.columns));
        for (std::size_t index = 0; index < left_values.size(); ++index) {
            left_values[index] =
                static_cast<float>(static_cast<int>(index % 17) - 8) / 128.0F;
        }
        for (std::size_t index = 0; index < right_values.size(); ++index) {
            right_values[index] =
                static_cast<float>(static_cast<int>(index % 13) - 6) / 128.0F;
        }
        const auto left = microllm::Tensor::from_vector(
            left_values, {options.rows, options.inner}).to(gpu);
        const auto right = microllm::Tensor::from_vector(
            right_values, {options.inner, options.columns}).to(gpu);
        microllm::Tensor output(
            {options.rows, options.columns}, microllm::DType::Float32, gpu);
        microllm::Tensor reference(
            {options.rows, options.columns}, microllm::DType::Float32, gpu);
        const auto* output_address = output.storage().data();
        microllm::runtime::Stream stream(gpu);
        microllm::ops::OpContext context;
        context.stream = &stream;
        microllm::ops::matmul_out_(
            reference, left, right, microllm::ops::MatmulImplementation::HipBLASLt,
            false, false, context);
        stream.synchronize();
        const auto expected = reference.to_vector();

        const auto work = [&] {
            for (std::int64_t call = 0; call < options.calls; ++call) {
                microllm::ops::matmul_out_(
                    output, left, right,
                    microllm::ops::MatmulImplementation::HipBLASLt,
                    false, false, context);
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
        const auto actual = output.to_vector();
        double maximum_error = 0.0;
        double squared_error = 0.0;
        for (std::size_t index = 0; index < actual.size(); ++index) {
            const auto difference = static_cast<double>(actual[index]) - expected[index];
            maximum_error = std::max(maximum_error, std::abs(difference));
            squared_error += difference * difference;
        }
        const auto rms_error = std::sqrt(
            squared_error / static_cast<double>(actual.size()));
        const auto pass = maximum_error == 0.0 && output.storage().data() == output_address &&
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
                  << ",\"calls\":" << options.calls
                  << ",\"captured_nodes\":"
                  << (graph.defined() ? graph.node_count() : 0U)
                  << ",\"rows\":" << options.rows
                  << ",\"inner\":" << options.inner
                  << ",\"columns\":" << options.columns
                  << ",\"warmup\":" << options.warmup
                  << ",\"repetitions\":" << options.repetitions
                  << ",\"setup_ms\":" << setup_ms
                  << ",\"event_median_ms\":" << percentile(event_ms, 0.5)
                  << ",\"event_p95_ms\":" << percentile(event_ms, 0.95)
                  << ",\"wall_median_ms\":" << percentile(wall_ms, 0.5)
                  << ",\"wall_p95_ms\":" << percentile(wall_ms, 0.95)
                  << ",\"maximum_absolute_error\":" << maximum_error
                  << ",\"rms_error\":" << rms_error
                  << ",\"output_address_stable\":"
                  << (output.storage().data() == output_address ? "true" : "false")
                  << ",\"host_to_device_calls\":"
                  << transfers.host_to_device_calls
                  << ",\"device_to_host_calls\":"
                  << transfers.device_to_host_calls
                  << ",\"device_to_device_calls\":"
                  << transfers.device_to_device_calls << "}\n";
        return pass ? EXIT_SUCCESS : EXIT_FAILURE;
    } catch (const std::exception& error) {
        std::cerr << "microllm_bench_hip_graph_gemm: " << error.what() << '\n';
        return EXIT_FAILURE;
    }
}
