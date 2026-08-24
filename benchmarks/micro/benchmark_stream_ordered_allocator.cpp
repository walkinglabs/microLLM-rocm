#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <set>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <microllm/core/tensor.h>
#include <microllm/ops/low_level.h>
#include <microllm/ops/ops.h>
#include <microllm/runtime/runtime.h>

namespace {

struct Options {
    std::string mode = "async";
    std::int64_t nodes = 32;
    std::int64_t elements = 4096;
    int warmup = 3;
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
    if (result.mode != "deferred" && result.mode != "async" &&
        result.mode != "graph") {
        throw std::invalid_argument("--mode must be deferred, async, or graph");
    }
    if (result.nodes <= 0 || result.elements <= 0 || result.warmup < 0 ||
        result.repetitions <= 0) {
        throw std::invalid_argument(
            "nodes/elements/repetitions must be positive and warmup nonnegative");
    }
    return result;
}

double percentile(std::vector<double> values, double fraction) {
    std::sort(values.begin(), values.end());
    const auto position = fraction * static_cast<double>(values.size() - 1);
    const auto low = static_cast<std::size_t>(position);
    const auto high = std::min(low + 1, values.size() - 1);
    const auto weight = position - static_cast<double>(low);
    return values[low] * (1.0 - weight) + values[high] * weight;
}

struct RunStats {
    double wall_ms = 0.0;
    double event_ms = 0.0;
    std::size_t unique_addresses = 0;
    std::size_t deferred_blocks = 0;
    std::size_t deferred_bytes = 0;
    std::size_t overflow_flushes = 0;
};

}  // namespace

int main(int argc, char** argv) {
    try {
        const auto command = parse_options(argc, argv);
        if (!microllm::runtime::hip_compiled() ||
            microllm::runtime::hip_device_count() == 0) {
            throw std::runtime_error("benchmark requires a visible HIP device");
        }
        const auto device = microllm::Device::hip(0);
        if (!microllm::runtime::stream_ordered_allocator_supported(device)) {
            throw std::runtime_error("HIP Stream ordered allocator is unsupported");
        }
        const auto bytes = static_cast<std::size_t>(command.elements) * sizeof(float);
        const std::array<std::int64_t, 1> shape{command.elements};
        const std::array<std::int64_t, 1> strides{1};
        const auto input = microllm::Tensor::from_vector(
            std::vector<float>(static_cast<std::size_t>(command.elements), 0.0F),
            {command.elements}).to(device);
        const auto source = microllm::Tensor::from_vector(
            std::vector<float>(static_cast<std::size_t>(command.elements), 1.0F),
            {command.elements}).to(device);
        const auto zero = microllm::Tensor::from_vector(
            std::vector<float>(static_cast<std::size_t>(command.elements), 0.0F),
            {command.elements}).to(device);
        microllm::Tensor output({command.elements}, microllm::DType::Float32, device);
        microllm::runtime::Stream stream(device);
        microllm::ops::OpContext context;
        context.stream = &stream;
        microllm::runtime::trim_default_hip_memory_pool(device, 0);
        microllm::runtime::set_default_hip_memory_pool_release_threshold(
            device, 8ULL * 1024ULL * 1024ULL * 1024ULL);

        const auto mutable_view = [&](void* pointer) {
            return microllm::TensorView{
                pointer, microllm::DType::Float32, device, shape, strides};
        };
        const auto const_view = [&](const void* pointer) {
            return microllm::ConstTensorView{
                pointer, microllm::DType::Float32, device, shape, strides};
        };
        const auto submit_async_chain = [&] {
            std::set<std::uintptr_t> addresses;
            microllm::runtime::StreamOrderedHipBuffer current(stream, bytes);
            addresses.insert(reinterpret_cast<std::uintptr_t>(current.data()));
            microllm::ops::add_out(
                mutable_view(current.data()), input.view(), source.view(), context);
            for (std::int64_t node = 1; node < command.nodes; ++node) {
                microllm::runtime::StreamOrderedHipBuffer next(stream, bytes);
                addresses.insert(reinterpret_cast<std::uintptr_t>(next.data()));
                microllm::ops::add_out(
                    mutable_view(next.data()), const_view(current.data()),
                    source.view(), context);
                current.release();
                current = std::move(next);
            }
            microllm::ops::add_out(
                output.view(), const_view(current.data()), zero.view(), context);
            current.release();
            return addresses.size();
        };
        microllm::runtime::HipGraphExecutable graph;
        std::size_t graph_nodes = 0;
        std::size_t graph_unique_addresses = 0;
        double graph_setup_ms = 0.0;
        if (command.mode == "graph") {
            const auto setup_start = std::chrono::steady_clock::now();
            graph = microllm::runtime::HipGraphExecutable::capture(stream, [&] {
                graph_unique_addresses = submit_async_chain();
            });
            const auto setup_finish = std::chrono::steady_clock::now();
            graph_nodes = graph.node_count();
            graph_setup_ms = std::chrono::duration<double, std::milli>(
                                 setup_finish - setup_start).count();
        }
        const auto execute = [&]() {
            RunStats stats;
            microllm::runtime::Event start(device);
            microllm::runtime::Event finish(device);
            const auto wall_start = std::chrono::steady_clock::now();
            start.record(stream);
            if (command.mode == "async") {
                stats.unique_addresses = submit_async_chain();
                finish.record(stream);
                stream.synchronize();
            } else if (command.mode == "graph") {
                graph.launch(stream);
                stats.unique_addresses = graph_unique_addresses;
                finish.record(stream);
                stream.synchronize();
            } else {
                microllm::runtime::ScopedDeferredHipStream scope(
                    stream, static_cast<std::size_t>(command.nodes + 16));
                auto current = input;
                for (std::int64_t node = 0; node < command.nodes; ++node) {
                    current = microllm::ops::add(current, source);
                }
                microllm::ops::add_out(
                    output.view(), std::as_const(current).view(), zero.view(), context);
                current = {};
                stats.deferred_blocks = scope.total_deferred_blocks();
                stats.deferred_bytes = scope.total_deferred_bytes();
                stats.overflow_flushes = scope.overflow_flushes();
                finish.record(stream);
                scope.finish();
            }
            const auto wall_finish = std::chrono::steady_clock::now();
            stats.wall_ms = std::chrono::duration<double, std::milli>(
                                wall_finish - wall_start).count();
            stats.event_ms = finish.elapsed_ms_since(start);
            return stats;
        };

        for (int iteration = 0; iteration < command.warmup; ++iteration) {
            (void)execute();
        }
        std::vector<double> wall;
        std::vector<double> event;
        std::size_t maximum_unique_addresses = 0;
        std::size_t maximum_deferred_blocks = 0;
        std::size_t maximum_deferred_bytes = 0;
        std::size_t overflow_flushes = 0;
        for (int iteration = 0; iteration < command.repetitions; ++iteration) {
            const auto stats = execute();
            wall.push_back(stats.wall_ms);
            event.push_back(stats.event_ms);
            maximum_unique_addresses = std::max(
                maximum_unique_addresses, stats.unique_addresses);
            maximum_deferred_blocks = std::max(
                maximum_deferred_blocks, stats.deferred_blocks);
            maximum_deferred_bytes = std::max(
                maximum_deferred_bytes, stats.deferred_bytes);
            overflow_flushes += stats.overflow_flushes;
        }
        const auto values = output.to_vector();
        double maximum_error = 0.0;
        for (const auto value : values) {
            maximum_error = std::max(
                maximum_error,
                std::abs(static_cast<double>(value) -
                         static_cast<double>(command.nodes)));
        }
        const auto pool = microllm::runtime::default_hip_memory_pool_stats(device);
        std::cout << std::setprecision(12)
                  << "{\"schema_version\":1,\"status\":\""
                  << (maximum_error == 0.0 ? "pass" : "fail") << "\""
                  << ",\"record_type\":\"stream_ordered_allocator_measurement\""
                  << ",\"mode\":\"" << command.mode << "\""
                  << ",\"device\":\"" << device.str() << "\""
                  << ",\"architecture\":\""
                  << microllm::runtime::device_info(device).architecture << "\""
                  << ",\"nodes\":" << command.nodes
                  << ",\"elements\":" << command.elements
                  << ",\"warmup\":" << command.warmup
                  << ",\"repetitions\":" << command.repetitions
                  << ",\"wall_p50_ms\":" << percentile(wall, 0.5)
                  << ",\"wall_p95_ms\":" << percentile(wall, 0.95)
                  << ",\"event_p50_ms\":" << percentile(event, 0.5)
                  << ",\"event_p95_ms\":" << percentile(event, 0.95)
                  << ",\"maximum_absolute_error\":" << maximum_error
                  << ",\"maximum_unique_addresses\":"
                  << maximum_unique_addresses
                  << ",\"maximum_deferred_blocks\":"
                  << maximum_deferred_blocks
                  << ",\"maximum_deferred_bytes\":"
                  << maximum_deferred_bytes
                  << ",\"overflow_flushes\":" << overflow_flushes
                  << ",\"pool_reserved_current_bytes\":"
                  << pool.reserved_current_bytes
                  << ",\"pool_reserved_high_bytes\":"
                  << pool.reserved_high_bytes
                  << ",\"pool_used_current_bytes\":" << pool.used_current_bytes
                  << ",\"pool_used_high_bytes\":" << pool.used_high_bytes
                  << ",\"pool_release_threshold_bytes\":"
                  << pool.release_threshold_bytes
                  << ",\"graph_setup_ms\":" << graph_setup_ms
                  << ",\"graph_node_count\":" << graph_nodes
                  << "}\n";
        return maximum_error == 0.0 ? 0 : 2;
    } catch (const std::exception& error) {
        std::cerr << "microllm_bench_stream_ordered_allocator: "
                  << error.what() << '\n';
        return 1;
    }
}
