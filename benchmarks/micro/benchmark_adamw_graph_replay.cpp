#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include <microllm/autograd/autograd.h>
#include <microllm/runtime/diagnostics.h>
#include <microllm/runtime/runtime.h>
#include <microllm/training/optimizer.h>

namespace {

struct Options {
    std::string mode = "graph";
    std::string precision = "fp32";
    std::int64_t tensors = 64;
    std::int64_t elements = 1024;
    int warmup = 3;
    int repetitions = 20;
};

Options options(int argc, char** argv) {
    Options result;
    for (int index = 1; index < argc; index += 2) {
        if (index + 1 >= argc) throw std::invalid_argument("missing option value");
        const std::string_view name(argv[index]);
        if (name == "--mode") result.mode = argv[index + 1];
        else if (name == "--precision") result.precision = argv[index + 1];
        else if (name == "--tensors") result.tensors = std::stoll(argv[index + 1]);
        else if (name == "--elements") result.elements = std::stoll(argv[index + 1]);
        else if (name == "--warmup") result.warmup = std::stoi(argv[index + 1]);
        else if (name == "--repetitions") {
            result.repetitions = std::stoi(argv[index + 1]);
        } else {
            throw std::invalid_argument("unknown option: " + std::string(name));
        }
    }
    if ((result.mode != "eager" && result.mode != "graph" &&
         result.mode != "graph-multi") ||
        (result.precision != "fp32" && result.precision != "bf16") ||
        result.tensors <= 0 || result.elements <= 0 || result.warmup < 0 ||
        result.repetitions <= 0) {
        throw std::invalid_argument("AdamW Graph benchmark options are invalid");
    }
    return result;
}

void emit_array(const std::vector<float>& values) {
    std::cout << '[';
    for (std::size_t index = 0; index < values.size(); ++index) {
        if (index != 0) std::cout << ',';
        std::cout << values[index];
    }
    std::cout << ']';
}

std::vector<float> prefix(std::vector<float> values, std::size_t count = 16) {
    values.resize(std::min(values.size(), count));
    return values;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const auto command = options(argc, argv);
        if (!microllm::runtime::hip_compiled() ||
            microllm::runtime::hip_device_count() == 0) {
            throw std::runtime_error("AdamW Graph benchmark requires HIP");
        }
        const auto device = microllm::Device::hip(0);
        std::vector<float> initial(
            static_cast<std::size_t>(command.elements));
        std::vector<float> gradient_values(
            static_cast<std::size_t>(command.elements));
        for (std::size_t index = 0; index < initial.size(); ++index) {
            initial[index] =
                static_cast<float>(static_cast<int>(index % 31) - 15) / 17.0F;
            gradient_values[index] =
                static_cast<float>(static_cast<int>(index % 13) - 6) / 19.0F;
        }
        const auto shared_gradient = microllm::Tensor::from_vector(
            gradient_values, {command.elements}).to(device);
        std::vector<microllm::autograd::Value> values;
        values.reserve(static_cast<std::size_t>(command.tensors));
        for (std::int64_t index = 0; index < command.tensors; ++index) {
            values.emplace_back(
                microllm::Tensor::from_vector(initial, {command.elements})
                    .to(device),
                true);
            values.back().set_grad(shared_gradient);
        }
        std::vector<microllm::Tensor> mirrors;
        mirrors.reserve(values.size());
        for (const auto& value : values) {
            mirrors.push_back(value.data().cast(microllm::DType::BFloat16));
        }
        microllm::training::Parameters parameters;
        microllm::training::Bf16ParameterMirrors mirror_map;
        parameters.reserve(values.size());
        mirror_map.reserve(values.size());
        for (std::size_t index = 0; index < values.size(); ++index) {
            parameters.push_back(&values[index]);
            mirror_map.emplace_back(&values[index], &mirrors[index]);
        }
        microllm::training::AdamWConfig config{
            .learning_rate = 1.0e-3F,
            .beta1 = 0.9F,
            .beta2 = 0.99F,
            .epsilon = 1.0e-8F,
            .weight_decay = 0.01F,
            .moment_precision =
                command.precision == "bf16"
                    ? microllm::training::AdamWConfig::MomentPrecision::BFloat16
                    : microllm::training::AdamWConfig::MomentPrecision::Float32};
        microllm::training::AdamW optimizer(
            parameters, config, mirror_map,
            microllm::ops::AdamWImplementation::Auto, 0);
        std::optional<microllm::ops::AdamWGraphStepState> graph_state;
        std::optional<microllm::training::AdamWGraphWorkspace> graph_workspace;
        double preparation_ms = 0.0;
        const auto preparation_started = std::chrono::steady_clock::now();
        if (command.mode == "graph") {
            graph_state.emplace(optimizer.make_graph_step_state());
        } else if (command.mode == "graph-multi") {
            graph_workspace.emplace(optimizer.make_graph_workspace());
        }
        preparation_ms = std::chrono::duration<double, std::milli>(
                             std::chrono::steady_clock::now() -
                             preparation_started)
                             .count();
        microllm::runtime::Stream stream(device);
        microllm::runtime::ScopedDeferredHipStream lifetime(stream, 1024);
        microllm::ops::OpContext context;
        context.stream = &stream;
        microllm::runtime::HipGraphExecutable graph;
        double setup_ms = 0.0;
        if (command.mode != "eager") {
            const auto started = std::chrono::steady_clock::now();
            graph = microllm::runtime::HipGraphExecutable::capture(stream, [&] {
                if (command.mode == "graph") {
                    optimizer.step_graph_replayable(*graph_state, context);
                } else {
                    optimizer.step_graph_replayable(*graph_workspace, context);
                }
            });
            setup_ms = std::chrono::duration<double, std::milli>(
                           std::chrono::steady_clock::now() - started)
                           .count();
        }
        const auto run_step = [&] {
            if (command.mode != "eager") graph.launch(stream);
            else optimizer.step();
        };
        for (int index = 0; index < command.warmup; ++index) run_step();
        stream.synchronize();
        microllm::runtime::reset_transfer_stats();
        microllm::runtime::Event start(device);
        microllm::runtime::Event stop(device);
        start.record(stream);
        const auto wall_started = std::chrono::steady_clock::now();
        for (int index = 0; index < command.repetitions; ++index) run_step();
        stop.record(stream);
        stop.synchronize();
        const auto wall_ms = std::chrono::duration<double, std::milli>(
                                 std::chrono::steady_clock::now() - wall_started)
                                 .count();
        const auto event_ms = stop.elapsed_ms_since(start);
        const auto transfers = microllm::runtime::transfer_stats();
        if (command.mode == "graph") {
            optimizer.synchronize_graph_step(*graph_state);
        } else if (command.mode == "graph-multi") {
            optimizer.synchronize_graph_step(*graph_workspace);
        }
        const auto state = optimizer.state();
        const auto parameter_sample = prefix(values.front().data().to_vector());
        const auto first_sample = prefix(state.first_moments.front().to_vector());
        const auto second_sample = prefix(state.second_moments.front().to_vector());
        const auto mirror_sample = prefix(mirrors.front().to_vector());
        const auto info = microllm::runtime::device_info(device);
        std::cout << std::setprecision(12)
                  << "{\"schema_version\":1,\"status\":\"pass\""
                  << ",\"record_type\":\"adamw_graph_replay_measurement\""
                  << ",\"mode\":\"" << command.mode << "\""
                  << ",\"precision\":\"" << command.precision << "\""
                  << ",\"tensors\":" << command.tensors
                  << ",\"elements\":" << command.elements
                  << ",\"warmup\":" << command.warmup
                  << ",\"repetitions\":" << command.repetitions
                  << ",\"final_step\":" << optimizer.step_count()
                  << ",\"captured_nodes\":"
                  << (graph.defined() ? graph.node_count() : 0U)
                  << ",\"preparation_ms\":" << preparation_ms
                  << ",\"setup_ms\":" << setup_ms
                  << ",\"event_ms_per_step\":"
                  << event_ms / static_cast<double>(command.repetitions)
                  << ",\"wall_ms_per_step\":"
                  << wall_ms / static_cast<double>(command.repetitions)
                  << ",\"timed_host_to_device_calls\":"
                  << transfers.host_to_device_calls
                  << ",\"timed_device_to_host_calls\":"
                  << transfers.device_to_host_calls
                  << ",\"timed_device_to_device_calls\":"
                  << transfers.device_to_device_calls
                  << ",\"device_name\":\"" << info.name << "\""
                  << ",\"architecture\":\"" << info.architecture << "\""
                  << ",\"parameter_sample\":";
        emit_array(parameter_sample);
        std::cout << ",\"first_moment_sample\":";
        emit_array(first_sample);
        std::cout << ",\"second_moment_sample\":";
        emit_array(second_sample);
        std::cout << ",\"mirror_sample\":";
        emit_array(mirror_sample);
        std::cout << "}\n";
        graph = microllm::runtime::HipGraphExecutable();
        lifetime.finish();
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "microllm_bench_adamw_graph_replay: " << error.what()
                  << '\n';
        return 1;
    }
}
