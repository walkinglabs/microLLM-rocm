#include <algorithm>
#include <chrono>
#include <cmath>
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
#include <microllm/runtime/runtime.h>

namespace {

struct Options {
    std::string model = "qwen";
    std::string mode = "arena_graph";
    std::int64_t rows = 32;
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
    if (result.mode != "deferred" && result.mode != "arena" &&
        result.mode != "arena_graph") {
        throw std::invalid_argument("--mode must be deferred, arena, or arena_graph");
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

microllm::Tensor external_tensor(
    void* pointer, microllm::Shape shape, microllm::Device device) {
    const auto elements = microllm::checked_numel(shape);
    const auto bytes = static_cast<std::size_t>(elements) * sizeof(float);
    auto storage = microllm::Storage::from_external(pointer, bytes, device);
    auto strides = microllm::contiguous_strides(shape);
    return microllm::Tensor::from_storage(
        std::move(storage), std::move(shape), std::move(strides), 0,
        microllm::DType::Float32);
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const auto command = parse_options(argc, argv);
        if (!microllm::runtime::hip_compiled() ||
            microllm::runtime::hip_device_count() == 0) {
            throw std::runtime_error("benchmark requires a visible HIP device");
        }
        const auto hidden = command.model == "qwen" ? 896 : 1536;
        const auto intermediate = command.model == "qwen" ? 4864 : 8960;
        const auto device = microllm::Device::hip(0);
        microllm::Tensor input({command.rows, hidden}, microllm::DType::Float32, device);
        microllm::Tensor gate_weight({hidden, intermediate}, microllm::DType::Float32, device);
        microllm::Tensor up_weight({hidden, intermediate}, microllm::DType::Float32, device);
        microllm::Tensor down_weight({intermediate, hidden}, microllm::DType::Float32, device);
        microllm::ops::fill_(input, 0.01F);
        microllm::ops::fill_(gate_weight, 0.001F);
        microllm::ops::fill_(up_weight, 0.002F);
        microllm::ops::fill_(down_weight, 0.001F);
        microllm::runtime::synchronize(device);

        const auto reference_gate = microllm::ops::matmul_with_implementation(
            input, gate_weight, microllm::ops::MatmulImplementation::HipBLASLt);
        const auto reference_up = microllm::ops::matmul_with_implementation(
            input, up_weight, microllm::ops::MatmulImplementation::HipBLASLt);
        const auto reference_activated = microllm::ops::swiglu(
            reference_gate, reference_up);
        const auto reference = microllm::ops::matmul_with_implementation(
            reference_activated, down_weight,
            microllm::ops::MatmulImplementation::HipBLASLt);
        microllm::runtime::synchronize(device);
        const auto expected = reference.to_vector();

        microllm::runtime::Stream stream(device);
        microllm::ops::OpContext context;
        context.stream = &stream;
        const auto intermediate_bytes =
            static_cast<std::size_t>(command.rows * intermediate) * sizeof(float);
        const auto aligned_bytes = (intermediate_bytes + 255U) & ~std::size_t{255U};
        microllm::runtime::HipActivationArena arena(stream, aligned_bytes * 3);
        auto gate = external_tensor(
            arena.allocate_slice(intermediate_bytes, 256),
            {command.rows, intermediate}, device);
        auto up = external_tensor(
            arena.allocate_slice(intermediate_bytes, 256),
            {command.rows, intermediate}, device);
        auto activated = external_tensor(
            arena.allocate_slice(intermediate_bytes, 256),
            {command.rows, intermediate}, device);
        microllm::Tensor output({command.rows, hidden}, microllm::DType::Float32, device);

        const auto submit_arena = [&] {
            microllm::ops::matmul_out_(
                gate, input, gate_weight,
                microllm::ops::MatmulImplementation::HipBLASLt,
                false, false, context);
            microllm::ops::matmul_out_(
                up, input, up_weight,
                microllm::ops::MatmulImplementation::HipBLASLt,
                false, false, context);
            microllm::ops::swiglu_out_(activated, gate, up, context);
            microllm::ops::matmul_out_(
                output, activated, down_weight,
                microllm::ops::MatmulImplementation::HipBLASLt,
                false, false, context);
        };

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

        const auto execute = [&] {
            microllm::runtime::Event start(device);
            microllm::runtime::Event finish(device);
            const auto wall_start = std::chrono::steady_clock::now();
            start.record(stream);
            if (command.mode == "deferred") {
                microllm::runtime::ScopedDeferredHipStream scope(stream, 32);
                {
                    auto current_gate = microllm::ops::matmul_with_implementation(
                        input, gate_weight,
                        microllm::ops::MatmulImplementation::HipBLASLt);
                    auto current_up = microllm::ops::matmul_with_implementation(
                        input, up_weight,
                        microllm::ops::MatmulImplementation::HipBLASLt);
                    auto current_activated = microllm::ops::swiglu(
                        current_gate, current_up);
                    microllm::ops::matmul_out_(
                        output, current_activated, down_weight,
                        microllm::ops::MatmulImplementation::HipBLASLt,
                        false, false, context);
                }
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
        std::vector<double> wall;
        std::vector<double> event;
        for (int iteration = 0; iteration < command.repetitions; ++iteration) {
            const auto [wall_ms, event_ms] = execute();
            wall.push_back(wall_ms);
            event.push_back(event_ms);
        }
        const auto actual = output.to_vector();
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
        std::cout << std::setprecision(12)
                  << "{\"schema_version\":1,\"status\":\""
                  << (maximum_error == 0.0 ? "pass" : "fail") << "\""
                  << ",\"record_type\":\"arena_ffn_measurement\""
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
                  << "}\n";
        return maximum_error == 0.0 ? 0 : 2;
    } catch (const std::exception& error) {
        std::cerr << "microllm_bench_arena_ffn: " << error.what() << '\n';
        return 1;
    }
}
