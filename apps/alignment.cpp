#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <string_view>

#include <microllm/model/model.h>
#include <microllm/profiling/trace.h>
#include <microllm/runtime/runtime.h>

namespace {

struct Options {
    std::filesystem::path output_directory;
    std::string device = "cpu";
    std::string run_id = "tiny-alignment";
    std::uint64_t seed = 211;
    std::uint64_t warmup = 2;
    std::uint64_t repetitions = 10;
    std::size_t max_captured_elements = 4096;
};

std::uint64_t parse_unsigned(std::string_view value, const char* name) {
    std::size_t consumed = 0;
    const auto parsed = std::stoull(std::string(value), &consumed);
    if (consumed != value.size()) throw std::invalid_argument(std::string(name) + " is invalid");
    return parsed;
}

Options parse_options(int argc, char** argv) {
    Options options;
    for (int index = 1; index < argc; ++index) {
        const std::string argument = argv[index];
        auto next = [&](const char* name) -> std::string {
            if (index + 1 >= argc) throw std::invalid_argument(std::string(name) + " requires a value");
            return argv[++index];
        };
        if (argument == "--output") options.output_directory = next("--output");
        else if (argument == "--device") options.device = next("--device");
        else if (argument == "--run-id") options.run_id = next("--run-id");
        else if (argument == "--seed") options.seed = parse_unsigned(next("--seed"), "seed");
        else if (argument == "--warmup") options.warmup = parse_unsigned(next("--warmup"), "warmup");
        else if (argument == "--repetitions") {
            options.repetitions = parse_unsigned(next("--repetitions"), "repetitions");
        } else if (argument == "--max-captured-elements") {
            options.max_captured_elements = static_cast<std::size_t>(
                parse_unsigned(next("--max-captured-elements"), "max captured elements"));
        } else {
            throw std::invalid_argument("unknown argument: " + argument);
        }
    }
    if (options.output_directory.empty()) throw std::invalid_argument("--output is required");
    if (options.repetitions == 0) throw std::invalid_argument("repetitions must be positive");
    if (options.max_captured_elements == 0) {
        throw std::invalid_argument("max captured elements must be positive");
    }
    return options;
}

microllm::model::ModelConfig tiny_config() {
    return {.vocabulary_size = 8,
            .dimension = 8,
            .layers = 1,
            .heads = 2,
            .kv_heads = 1,
            .ffn_dimension = 16,
            .max_sequence_length = 4,
            .rope_base = 10000.0F,
            .tie_embeddings = false};
}

microllm::Device parse_device(const Options& options) {
    if (options.device == "cpu") return microllm::Device::cpu();
    if (options.device == "hip") {
        if (!microllm::runtime::hip_compiled() || microllm::runtime::hip_device_count() == 0) {
            throw std::runtime_error("HIP alignment requested without a visible HIP device");
        }
        return microllm::Device::hip(0);
    }
    throw std::invalid_argument("device must be cpu or hip");
}

void write_run_metadata(const std::filesystem::path& path, const Options& options,
                        microllm::Device device,
                        const microllm::model::ModelConfig& config) {
    std::ofstream output(path, std::ios::trunc);
    if (!output) throw std::runtime_error("cannot open alignment metadata output");
    output << "{\n"
           << "  \"schema_version\": 1,\n"
           << "  \"framework\": \"microllm\",\n"
           << "  \"run_id\": \"" << options.run_id << "\",\n"
           << "  \"device\": \"" << device.str() << "\",\n"
           << "  \"seed\": " << options.seed << ",\n"
           << "  \"warmup\": " << options.warmup << ",\n"
           << "  \"repetitions\": " << options.repetitions << ",\n"
           << "  \"model\": {\n"
           << "    \"vocabulary_size\": " << config.vocabulary_size << ",\n"
           << "    \"dimension\": " << config.dimension << ",\n"
           << "    \"layers\": " << config.layers << ",\n"
           << "    \"heads\": " << config.heads << ",\n"
           << "    \"kv_heads\": " << config.kv_heads << ",\n"
           << "    \"ffn_dimension\": " << config.ffn_dimension << ",\n"
           << "    \"max_sequence_length\": " << config.max_sequence_length << ",\n"
           << "    \"rope_base\": " << config.rope_base << "\n"
           << "  }\n"
           << "}\n";
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const auto options = parse_options(argc, argv);
        const auto device = parse_device(options);
        const auto config = tiny_config();
        std::filesystem::create_directories(options.output_directory);

        microllm::model::TransformerModel model(config, options.seed);
        model.to(device);
        auto tokens = microllm::Tensor::from_int32_vector({0, 1, 2, 3}, {1, 4});
        auto targets = microllm::Tensor::from_int32_vector({1, 2, 3, 0}, {1, 4});
        if (device != microllm::Device::cpu()) tokens = tokens.to(device);
        if (device != microllm::Device::cpu()) targets = targets.to(device);

        for (std::uint64_t iteration = 0; iteration < options.warmup; ++iteration) {
            (void)model.forward(tokens);
        }
        microllm::runtime::synchronize(device);

        microllm::profiling::TraceOptions parameter_options;
        parameter_options.phase = "parameters";
        parameter_options.record_operators = false;
        parameter_options.record_layers = false;
        parameter_options.record_model = false;
        parameter_options.capture_values = true;
        parameter_options.max_captured_elements = options.max_captured_elements;
        microllm::profiling::TraceSession parameter_session(
            "microllm", options.run_id, parameter_options);
        for (const auto& [name, tensor] : model.state_dict()) {
            parameter_session.record(microllm::profiling::TraceKind::Parameter,
                                     name, tensor);
        }
        parameter_session.write_jsonl(options.output_directory / "microllm_parameters.jsonl");

        microllm::profiling::TraceOptions value_options;
        value_options.phase = "values";
        value_options.capture_values = true;
        value_options.max_captured_elements = options.max_captured_elements;
        microllm::profiling::TraceSession value_session("microllm", options.run_id,
                                                         value_options);
        {
            microllm::profiling::ScopedTraceSession active(value_session);
            value_session.set_iteration(0);
            (void)model.forward(tokens);
        }
        value_session.write_jsonl(options.output_directory / "microllm_values.jsonl");

        microllm::profiling::TraceOptions training_value_options;
        training_value_options.phase = "training_values";
        training_value_options.record_operators = false;
        training_value_options.record_layers = false;
        training_value_options.record_model = false;
        training_value_options.capture_values = true;
        training_value_options.max_captured_elements = options.max_captured_elements;
        microllm::profiling::TraceSession training_value_session(
            "microllm", options.run_id, training_value_options);
        for (auto* parameter : model.parameters()) parameter->zero_grad();
        const auto training_loss = model.loss(tokens, targets);
        training_loss.backward();
        training_value_session.record(microllm::profiling::TraceKind::Output,
                                      "training.loss", training_loss.data());
        for (const auto& [name, parameter] : model.named_parameters()) {
            if (!parameter->has_grad()) {
                throw std::runtime_error("missing gradient for parameter: " + name);
            }
            training_value_session.record(microllm::profiling::TraceKind::Parameter,
                                          "gradient." + name, parameter->grad());
        }
        training_value_session.write_jsonl(
            options.output_directory / "microllm_training_values.jsonl");

        microllm::profiling::TraceOptions operator_options;
        operator_options.phase = "operator_timing";
        operator_options.record_operators = true;
        operator_options.record_layers = false;
        operator_options.record_model = false;
        operator_options.capture_values = false;
        microllm::profiling::TraceSession operator_session(
            "microllm", options.run_id, operator_options);
        {
            microllm::profiling::ScopedTraceSession active(operator_session);
            for (std::uint64_t iteration = 0; iteration < options.repetitions; ++iteration) {
                operator_session.set_iteration(iteration);
                (void)model.forward(tokens);
            }
        }
        operator_session.write_jsonl(
            options.output_directory / "microllm_operator_timing.jsonl");

        microllm::profiling::TraceOptions layer_options;
        layer_options.phase = "layer_timing";
        layer_options.record_operators = false;
        layer_options.record_layers = true;
        layer_options.record_model = true;
        layer_options.capture_values = false;
        microllm::profiling::TraceSession layer_session("microllm", options.run_id,
                                                         layer_options);
        {
            microllm::profiling::ScopedTraceSession active(layer_session);
            for (std::uint64_t iteration = 0; iteration < options.repetitions; ++iteration) {
                layer_session.set_iteration(iteration);
                (void)model.forward(tokens);
            }
        }
        layer_session.write_jsonl(options.output_directory / "microllm_layer_timing.jsonl");

        for (std::uint64_t iteration = 0; iteration < options.warmup; ++iteration) {
            for (auto* parameter : model.parameters()) parameter->zero_grad();
            model.loss(tokens, targets).backward();
        }
        microllm::profiling::TraceOptions backward_options;
        backward_options.phase = "backward_timing";
        backward_options.record_operators = false;
        backward_options.record_layers = false;
        backward_options.record_model = true;
        backward_options.capture_values = false;
        microllm::profiling::TraceSession backward_session(
            "microllm", options.run_id, backward_options);
        for (std::uint64_t iteration = 0; iteration < options.repetitions; ++iteration) {
            for (auto* parameter : model.parameters()) parameter->zero_grad();
            const auto loss = model.loss(tokens, targets);
            backward_session.set_iteration(iteration);
            microllm::profiling::ScopedTraceSession active(backward_session);
            microllm::profiling::TraceTimer timer(
                microllm::profiling::TraceKind::Model, "model.backward", device);
            loss.backward();
            timer.finish(loss.data());
        }
        backward_session.write_jsonl(
            options.output_directory / "microllm_backward_timing.jsonl");

        write_run_metadata(options.output_directory / "microllm_run.json",
                           options, device, config);
        std::cout << "alignment_output=" << options.output_directory << '\n';
        std::cout << "framework=microllm\n";
        std::cout << "device=" << device.str() << '\n';
        std::cout << "status=pass\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "microllm_alignment: " << error.what() << '\n';
        return 1;
    }
}
