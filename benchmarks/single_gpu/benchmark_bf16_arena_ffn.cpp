#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <microllm/core/storage.h>
#include <microllm/core/tensor.h>
#include <microllm/ops/ops.h>
#include <microllm/runtime/memory.h>
#include <microllm/runtime/runtime.h>

namespace {

struct Options {
    std::string model = "qwen";
    std::string mode = "arena_graph";
    std::int64_t rows = 1;
    int warmup = 3;
    int repetitions = 20;
};

Options parse_options(int argc, char** argv) {
    Options result;
    for (int index = 1; index < argc; index += 2) {
        if (index + 1 >= argc) throw std::invalid_argument("missing CLI value");
        const std::string name = argv[index];
        const std::string value = argv[index + 1];
        if (name == "--model") result.model = value;
        else if (name == "--mode") result.mode = value;
        else if (name == "--rows") result.rows = std::stoll(value);
        else if (name == "--warmup") result.warmup = std::stoi(value);
        else if (name == "--repetitions") result.repetitions = std::stoi(value);
        else throw std::invalid_argument("unknown option: " + name);
    }
    if (result.model != "qwen" && result.model != "deepseek") {
        throw std::invalid_argument("--model must be qwen or deepseek");
    }
    if (result.mode != "baseline" && result.mode != "arena" &&
        result.mode != "arena_graph") {
        throw std::invalid_argument(
            "--mode must be baseline, arena, or arena_graph");
    }
    if (result.rows <= 0 || result.warmup < 0 || result.repetitions <= 0) {
        throw std::invalid_argument(
            "rows/repetitions must be positive and warmup nonnegative");
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

std::size_t aligned(std::size_t bytes, std::size_t alignment = 256) {
    return (bytes + alignment - 1) & ~(alignment - 1);
}

microllm::Tensor external_tensor(
    void* pointer, microllm::Shape shape, microllm::DType dtype,
    microllm::Device device) {
    const auto bytes = static_cast<std::size_t>(microllm::checked_numel(shape)) *
                       microllm::dtype_size(dtype);
    auto storage = microllm::Storage::from_external(pointer, bytes, device);
    return microllm::Tensor::from_storage(
        std::move(storage), shape, microllm::contiguous_strides(shape), 0,
        dtype);
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const auto command = parse_options(argc, argv);
        if (!microllm::runtime::hip_compiled() ||
            microllm::runtime::hip_device_count() == 0 ||
            !microllm::ops::hipblaslt_available()) {
            throw std::runtime_error(
                "benchmark requires HIP and hipBLASLt");
        }
        const std::int64_t hidden = command.model == "qwen" ? 896 : 1536;
        const std::int64_t intermediate =
            command.model == "qwen" ? 4864 : 8960;
        const auto device = microllm::Device::hip(0);
        microllm::Tensor input(
            {command.rows, hidden}, microllm::DType::Float32, device);
        microllm::Tensor gate_weight(
            {hidden, intermediate}, microllm::DType::BFloat16, device);
        microllm::Tensor up_weight(
            {hidden, intermediate}, microllm::DType::BFloat16, device);
        microllm::Tensor down_weight(
            {intermediate, hidden}, microllm::DType::BFloat16, device);
        microllm::ops::fill_(input, 0.01F);
        microllm::ops::fill_(gate_weight, 0.001F);
        microllm::ops::fill_(up_weight, 0.002F);
        microllm::ops::fill_(down_weight, 0.001F);
        microllm::runtime::synchronize(device);

        const auto reference = microllm::ops::bf16_ffn(
            input, gate_weight, up_weight, down_weight);
        microllm::runtime::synchronize(device);
        const auto expected = reference.to_vector();

        microllm::runtime::Stream stream(device);
        microllm::ops::OpContext context;
        context.stream = &stream;
        const auto input_bytes = static_cast<std::size_t>(
            command.rows * hidden) * microllm::dtype_size(
                microllm::DType::BFloat16);
        const auto intermediate_bytes = static_cast<std::size_t>(
            command.rows * intermediate) * microllm::dtype_size(
                microllm::DType::BFloat16);
        const auto capacity = aligned(input_bytes) * 2 +
                              aligned(intermediate_bytes) * 3;
        microllm::runtime::HipActivationArena arena(stream, capacity);
        auto input_bf16 = external_tensor(
            arena.allocate_slice(input_bytes), {command.rows, hidden},
            microllm::DType::BFloat16, device);
        auto gate = external_tensor(
            arena.allocate_slice(intermediate_bytes),
            {command.rows, intermediate}, microllm::DType::BFloat16, device);
        auto up = external_tensor(
            arena.allocate_slice(intermediate_bytes),
            {command.rows, intermediate}, microllm::DType::BFloat16, device);
        auto activated = external_tensor(
            arena.allocate_slice(intermediate_bytes),
            {command.rows, intermediate}, microllm::DType::BFloat16, device);
        auto output_fallback = external_tensor(
            arena.allocate_slice(input_bytes), {command.rows, hidden},
            microllm::DType::BFloat16, device);
        microllm::ops::Bf16FfnWorkspace workspace{
            .input_bf16 = input_bf16,
            .gate = gate,
            .up = up,
            .activated = activated,
            .output_fallback_bf16 = output_fallback};
        microllm::Tensor output(
            {command.rows, hidden}, microllm::DType::Float32, device);
        const auto submit_arena = [&] {
            microllm::ops::bf16_ffn_out_(
                output, workspace, input, gate_weight, up_weight, down_weight,
                context);
        };

        // Establish the runtime-specific direct-FP32 decision before capture.
        submit_arena();
        stream.synchronize();
        microllm::runtime::HipGraphExecutable graph;
        double setup_ms = 0.0;
        std::size_t graph_nodes = 0;
        if (command.mode == "arena_graph") {
            const auto start = std::chrono::steady_clock::now();
            graph = microllm::runtime::HipGraphExecutable::capture(
                stream, submit_arena);
            const auto finish = std::chrono::steady_clock::now();
            setup_ms = std::chrono::duration<double, std::milli>(
                           finish - start).count();
            graph_nodes = graph.node_count();
        }

        microllm::Tensor baseline_output;
        const auto execute = [&] {
            microllm::runtime::Event start(device);
            microllm::runtime::Event finish(device);
            const auto wall_start = std::chrono::steady_clock::now();
            start.record(stream);
            if (command.mode == "baseline") {
                microllm::runtime::ScopedDeferredHipStream scope(stream, 32);
                baseline_output = microllm::ops::bf16_ffn(
                    input, gate_weight, up_weight, down_weight);
                finish.record(stream);
                scope.finish();
            } else if (command.mode == "arena") {
                submit_arena();
                finish.record(stream);
                stream.synchronize();
            } else {
                graph.launch(stream);
                finish.record(stream);
                stream.synchronize();
            }
            const auto wall_finish = std::chrono::steady_clock::now();
            return std::pair{
                std::chrono::duration<double, std::milli>(
                    wall_finish - wall_start).count(),
                static_cast<double>(finish.elapsed_ms_since(start))};
        };

        for (int iteration = 0; iteration < command.warmup; ++iteration) {
            (void)execute();
        }
        microllm::runtime::reset_allocation_peak(device);
        std::vector<double> wall;
        std::vector<double> event;
        for (int iteration = 0; iteration < command.repetitions; ++iteration) {
            const auto [wall_ms, event_ms] = execute();
            wall.push_back(wall_ms);
            event.push_back(event_ms);
        }
        const auto actual = (command.mode == "baseline" ? baseline_output
                                                         : output).to_vector();
        double maximum_error = 0.0;
        double square_error = 0.0;
        for (std::size_t index = 0; index < actual.size(); ++index) {
            const auto difference = std::abs(
                static_cast<double>(actual[index]) - expected[index]);
            maximum_error = std::max(maximum_error, difference);
            square_error += difference * difference;
        }
        const auto rms_error = std::sqrt(
            square_error / static_cast<double>(actual.size()));
        const auto allocations = microllm::runtime::allocation_stats(device);
        std::cout << std::setprecision(12)
                  << "{\"schema_version\":1,\"status\":\""
                  << (maximum_error == 0.0 ? "pass" : "fail") << "\""
                  << ",\"record_type\":\"bf16_arena_ffn_measurement\""
                  << ",\"model\":\"" << command.model << "\""
                  << ",\"mode\":\"" << command.mode << "\""
                  << ",\"rows\":" << command.rows
                  << ",\"hidden\":" << hidden
                  << ",\"intermediate\":" << intermediate
                  << ",\"warmup\":" << command.warmup
                  << ",\"repetitions\":" << command.repetitions
                  << ",\"wall_p50_ms\":" << percentile(wall, 0.5)
                  << ",\"wall_p95_ms\":" << percentile(wall, 0.95)
                  << ",\"event_p50_ms\":" << percentile(event, 0.5)
                  << ",\"event_p95_ms\":" << percentile(event, 0.95)
                  << ",\"maximum_absolute_error\":" << maximum_error
                  << ",\"rms_error\":" << rms_error
                  << ",\"arena_capacity_bytes\":" << arena.capacity_bytes()
                  << ",\"arena_planned_bytes\":" << arena.planned_bytes()
                  << ",\"graph_setup_ms\":" << setup_ms
                  << ",\"graph_node_count\":" << graph_nodes
                  << ",\"measured_allocation_calls\":"
                  << allocations.allocation_calls
                  << ",\"measured_deallocation_calls\":"
                  << allocations.deallocation_calls << "}\n";
        return maximum_error == 0.0 ? 0 : 2;
    } catch (const std::exception& error) {
        std::cerr << "microllm_bench_bf16_arena_ffn: " << error.what()
                  << '\n';
        return 1;
    }
}
