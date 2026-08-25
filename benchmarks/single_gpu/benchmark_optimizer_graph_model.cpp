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

#include <microllm/io/safetensors.h>
#include <microllm/model/huggingface.h>
#include <microllm/model/model.h>
#include <microllm/ops/ops.h>
#include <microllm/runtime/memory.h>
#include <microllm/runtime/runtime.h>
#include <microllm/training/optimizer.h>

namespace {

struct Options {
    std::string model = "qwen";
    std::string mode = "eager";
    std::string config;
    int context = 8;
    int steps = 2;
};

Options options(int argc, char** argv) {
    Options result;
    for (int index = 1; index < argc; index += 2) {
        if (index + 1 >= argc) throw std::invalid_argument("missing option value");
        const std::string_view name(argv[index]);
        if (name == "--model") result.model = argv[index + 1];
        else if (name == "--mode") result.mode = argv[index + 1];
        else if (name == "--config") result.config = argv[index + 1];
        else if (name == "--context") result.context = std::stoi(argv[index + 1]);
        else if (name == "--steps") result.steps = std::stoi(argv[index + 1]);
        else throw std::invalid_argument("unknown option: " + std::string(name));
    }
    if ((result.model != "qwen" && result.model != "deepseek") ||
        (result.mode != "eager" && result.mode != "graph" &&
         result.mode != "preflight") ||
        result.config.empty() || result.context < 2 || result.steps < 2) {
        throw std::invalid_argument("optimizer model Graph options are invalid");
    }
    return result;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const auto command = options(argc, argv);
        if (!microllm::runtime::hip_compiled() ||
            microllm::runtime::hip_device_count() == 0) {
            throw std::runtime_error("optimizer model Graph benchmark requires HIP");
        }
        auto config =
            microllm::model::load_huggingface_config(command.config).model;
        config.linear_precision = microllm::model::LinearPrecision::BFloat16;
        const auto device = microllm::Device::hip(0);
        microllm::model::TransformerModel model(
            config, 19,
            microllm::model::ParameterInitialization::Uninitialized);
        model.to(device);
        const auto named = model.named_parameters();
        for (const auto& [name, parameter] : named) {
            const auto norm = name.find("norm.weight") != std::string::npos;
            const auto bias = name.ends_with(".bias");
            microllm::ops::fill_(parameter->mutable_data(),
                                 norm ? 1.0F : bias ? 0.0F : 0.01F);
        }
        microllm::io::StateDict synthetic_state;
        for (const auto& [name, parameter] : named) {
            synthetic_state.emplace(name, parameter->data());
        }
        (void)model.load_state_dict(synthetic_state);
        auto mirrors = model.prepare_bf16_training_mirrors();
        microllm::training::AdamW optimizer(
            model.parameters(),
            {.learning_rate = 1.0e-5F,
             .beta1 = 0.9F,
             .beta2 = 0.999F,
             .epsilon = 1.0e-8F,
             .weight_decay = 0.01F,
             .moment_precision = microllm::training::AdamWConfig::
                 MomentPrecision::BFloat16},
            mirrors, microllm::ops::AdamWImplementation::Auto, 1 << 20);
        std::vector<std::int32_t> input_values;
        std::vector<std::int32_t> target_values;
        for (int index = 0; index < command.context; ++index) {
            input_values.push_back(1 + index % 31);
            target_values.push_back(2 + index % 31);
        }
        const auto inputs = microllm::Tensor::from_int32_vector(
                                input_values, {1, command.context})
                                .to(device);
        const auto targets = microllm::Tensor::from_int32_vector(
                                 target_values, {1, command.context})
                                 .to(device);
        // Construct the required capture Stream before any allocator warmup.
        // This deliberately exercises the runtime's real contract: one
        // non-default Stream permanently disables default-Stream exact-size
        // reuse on this device.
        microllm::runtime::Stream optimizer_stream(device);
        microllm::ops::OpContext optimizer_context;
        optimizer_context.stream = &optimizer_stream;
        const auto backward_only = [&]() {
            microllm::training::zero_grad(model.parameters());
            const auto loss = model.loss(inputs, targets);
            const auto value = loss.data().to_vector().front();
            loss.backward();
            microllm::runtime::synchronize(device);
            if (!std::isfinite(value)) {
                throw std::runtime_error("optimizer model Graph loss is non-finite");
            }
            return value;
        };
        (void)backward_only();
        (void)backward_only();
        const auto preparation_started = std::chrono::steady_clock::now();
        std::optional<microllm::training::AdamWGraphWorkspace> workspace;
        microllm::runtime::HipGraphExecutable graph;
        if (command.mode != "eager") {
            workspace.emplace(optimizer.make_graph_workspace());
        }
        if (command.mode == "graph") {
            graph = microllm::runtime::HipGraphExecutable::capture(
                optimizer_stream, [&] {
                    optimizer.step_graph_replayable(*workspace,
                                                    optimizer_context);
                });
        }
        const auto preparation_ms = std::chrono::duration<double, std::milli>(
                                        std::chrono::steady_clock::now() -
                                        preparation_started)
                                        .count();
        if (command.mode == "preflight") {
            (void)backward_only();
            const auto eligible =
                optimizer.graph_workspace_matches_current_gradients(*workspace);
            std::cout << std::setprecision(12)
                      << "{\"schema_version\":1,\"status\":\"pass\""
                      << ",\"record_type\":\"optimizer_graph_model_measurement\""
                      << ",\"model\":\"" << command.model << "\""
                      << ",\"mode\":\"preflight\""
                      << ",\"context\":" << command.context
                      << ",\"parameter_count\":" << model.parameter_count()
                      << ",\"parameter_tensors\":" << named.size()
                      << ",\"gradient_snapshot_matches\":"
                      << (eligible ? "true" : "false")
                      << ",\"caching_allocator_enabled\":"
                      << (microllm::runtime::hip_caching_allocator_enabled(device)
                              ? "true" : "false")
                      << ",\"graph_launched\":false"
                      << ",\"captured_nodes\":0"
                      << ",\"preparation_ms\":" << preparation_ms
                      << "}\n";
            return 0;
        }
        std::vector<float> losses;
        losses.reserve(static_cast<std::size_t>(command.steps));
        double optimizer_ms = 0.0;
        std::uint64_t optimizer_h2d = 0;
        std::uint64_t optimizer_d2h = 0;
        std::uint64_t optimizer_d2d = 0;
        bool all_addresses_match = true;
        microllm::runtime::reset_allocation_peak(device);
        microllm::runtime::reset_transfer_stats();
        const auto measured_started = std::chrono::steady_clock::now();
        for (int step = 0; step < command.steps; ++step) {
            losses.push_back(backward_only());
            if (command.mode == "graph" &&
                !optimizer.graph_workspace_matches_current_gradients(*workspace)) {
                all_addresses_match = false;
                break;
            }
            microllm::runtime::reset_transfer_stats();
            const auto optimizer_started = std::chrono::steady_clock::now();
            if (command.mode == "graph") graph.launch(optimizer_stream);
            else optimizer.step();
            microllm::runtime::synchronize(device);
            optimizer_ms += std::chrono::duration<double, std::milli>(
                                std::chrono::steady_clock::now() -
                                optimizer_started)
                                .count();
            const auto optimizer_transfers =
                microllm::runtime::transfer_stats();
            optimizer_h2d += optimizer_transfers.host_to_device_calls;
            optimizer_d2h += optimizer_transfers.device_to_host_calls;
            optimizer_d2d += optimizer_transfers.device_to_device_calls;
        }
        const auto measured_ms = std::chrono::duration<double, std::milli>(
                                     std::chrono::steady_clock::now() -
                                     measured_started)
                                     .count();
        if (command.mode == "graph") {
            optimizer.synchronize_graph_step(*workspace);
        }
        if (!all_addresses_match) {
            throw std::runtime_error(
                "optimizer Graph gradient snapshot changed before launch");
        }
        const auto transfers = microllm::runtime::transfer_stats();
        const auto allocation = microllm::runtime::allocation_stats(device);
        const auto observed = named.back().second->data().to_vector().front();
        const auto info = microllm::runtime::device_info(device);
        std::cout << std::setprecision(12)
                  << "{\"schema_version\":1,\"status\":\"pass\""
                  << ",\"record_type\":\"optimizer_graph_model_measurement\""
                  << ",\"model\":\"" << command.model << "\""
                  << ",\"mode\":\"" << command.mode << "\""
                  << ",\"context\":" << command.context
                  << ",\"steps\":" << command.steps
                  << ",\"parameter_count\":" << model.parameter_count()
                  << ",\"parameter_tensors\":" << named.size()
                  << ",\"gradient_snapshot_matches\":"
                  << (all_addresses_match ? "true" : "false")
                  << ",\"caching_allocator_enabled\":"
                  << (microllm::runtime::hip_caching_allocator_enabled(device)
                          ? "true" : "false")
                  << ",\"graph_launched\":"
                  << (command.mode == "graph" ? "true" : "false")
                  << ",\"captured_nodes\":"
                  << (graph.defined() ? graph.node_count() : 0U)
                  << ",\"preparation_ms\":" << preparation_ms
                  << ",\"measured_ms\":" << measured_ms
                  << ",\"mean_step_ms\":"
                  << measured_ms / static_cast<double>(command.steps)
                  << ",\"optimizer_ms\":" << optimizer_ms
                  << ",\"mean_optimizer_ms\":"
                  << optimizer_ms / static_cast<double>(command.steps)
                  << ",\"optimizer_step\":" << optimizer.step_count()
                  << ",\"observed_parameter\":" << observed
                  << ",\"optimizer_host_to_device_calls\":"
                  << optimizer_h2d
                  << ",\"optimizer_device_to_host_calls\":"
                  << optimizer_d2h
                  << ",\"optimizer_device_to_device_calls\":"
                  << optimizer_d2d
                  << ",\"host_to_device_calls\":"
                  << transfers.host_to_device_calls
                  << ",\"device_to_host_calls\":"
                  << transfers.device_to_host_calls
                  << ",\"device_to_device_calls\":"
                  << transfers.device_to_device_calls
                  << ",\"engine_peak_bytes\":" << allocation.peak_bytes
                  << ",\"device_name\":\"" << info.name << "\""
                  << ",\"architecture\":\"" << info.architecture << "\""
                  << ",\"losses\":[";
        for (std::size_t index = 0; index < losses.size(); ++index) {
            if (index != 0) std::cout << ',';
            std::cout << losses[index];
        }
        std::cout << "]}\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "microllm_bench_optimizer_graph_model: "
                  << error.what() << '\n';
        return 1;
    }
}
