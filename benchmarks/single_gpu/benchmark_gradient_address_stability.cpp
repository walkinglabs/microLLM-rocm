#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <map>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include <microllm/autograd/autograd.h>
#include <microllm/io/safetensors.h>
#include <microllm/model/huggingface.h>
#include <microllm/model/model.h>
#include <microllm/ops/ops.h>
#include <microllm/runtime/diagnostics.h>
#include <microllm/runtime/memory.h>
#include <microllm/runtime/runtime.h>
#include <microllm/training/optimizer.h>

namespace {

struct Options {
    std::string model = "tiny";
    std::string precision = "bf16";
    std::string config;
    int warmup = 1;
    int steps = 2;
    int context = 8;
};

Options options(int argc, char** argv) {
    Options result;
    for (int index = 1; index < argc; index += 2) {
        if (index + 1 >= argc) throw std::invalid_argument("missing option value");
        const std::string_view name(argv[index]);
        if (name == "--model") result.model = argv[index + 1];
        else if (name == "--precision") result.precision = argv[index + 1];
        else if (name == "--config") result.config = argv[index + 1];
        else if (name == "--warmup") result.warmup = std::stoi(argv[index + 1]);
        else if (name == "--steps") result.steps = std::stoi(argv[index + 1]);
        else if (name == "--context") result.context = std::stoi(argv[index + 1]);
        else throw std::invalid_argument("unknown option: " + std::string(name));
    }
    if ((result.model != "tiny" && result.model != "qwen" &&
         result.model != "deepseek") ||
        (result.precision != "fp32" && result.precision != "bf16") ||
        result.warmup < 1 || result.steps < 2 || result.context < 2 ||
        (result.model != "tiny" && result.config.empty())) {
        throw std::invalid_argument("gradient address options are invalid");
    }
    return result;
}

microllm::model::ModelConfig model_config(const Options& options) {
    if (options.model == "tiny") {
        return {.vocabulary_size = 64,
                .dimension = 32,
                .layers = 2,
                .heads = 4,
                .kv_heads = 2,
                .ffn_dimension = 64,
                .max_sequence_length = 32,
                .rope_base = 10000.0F,
                .tie_embeddings = false};
    }
    return microllm::model::load_huggingface_config(options.config).model;
}

struct AddressRecord {
    const void* first = nullptr;
    std::uint64_t bytes = 0;
    std::uint64_t elements = 0;
    std::size_t observations = 0;
    std::size_t changes = 0;
    std::size_t minimum_use_count = 0;
    std::size_t maximum_use_count = 0;
};

}  // namespace

int main(int argc, char** argv) {
    try {
        const auto command = options(argc, argv);
        if (!microllm::runtime::hip_compiled() ||
            microllm::runtime::hip_device_count() == 0) {
            throw std::runtime_error("gradient address audit requires HIP");
        }
        auto config = model_config(command);
        if (command.precision == "bf16") {
            config.linear_precision =
                microllm::model::LinearPrecision::BFloat16;
        }
        const auto device = microllm::Device::hip(0);
        microllm::model::TransformerModel model(
            config, 17,
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
        microllm::model::Bf16TrainingMirrors mirrors;
        if (command.precision == "bf16") {
            mirrors = model.prepare_bf16_training_mirrors();
        }
        std::vector<std::int32_t> input_values;
        std::vector<std::int32_t> target_values;
        input_values.reserve(static_cast<std::size_t>(command.context));
        target_values.reserve(static_cast<std::size_t>(command.context));
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
        std::map<std::string, AddressRecord> records;
        std::vector<float> losses;
        const auto started = std::chrono::steady_clock::now();
        for (int iteration = 0;
             iteration < command.warmup + command.steps; ++iteration) {
            microllm::training::zero_grad(model.parameters());
            {
                const auto loss = model.loss(inputs, targets);
                const auto loss_value = loss.data().to_vector().front();
                if (!std::isfinite(loss_value)) {
                    throw std::runtime_error("gradient address audit loss is non-finite");
                }
                loss.backward();
                microllm::runtime::synchronize(device);
                if (iteration >= command.warmup) {
                    losses.push_back(loss_value);
                    for (const auto& [name, parameter] : named) {
                        if (!parameter->has_grad()) {
                            throw std::runtime_error(
                                "gradient address audit found a missing gradient");
                        }
                        const auto storage = parameter->grad().storage();
                        const auto use_count = static_cast<std::size_t>(
                            storage.use_count());
                        auto& record = records[name];
                        const auto* address = storage.data();
                        if (record.observations == 0) {
                            record.first = address;
                            record.bytes = storage.num_bytes();
                            record.elements = static_cast<std::uint64_t>(
                                parameter->grad().numel());
                            record.minimum_use_count = use_count;
                            record.maximum_use_count = use_count;
                        } else if (address != record.first) {
                            ++record.changes;
                        }
                        record.minimum_use_count = std::min(
                            record.minimum_use_count, use_count);
                        record.maximum_use_count = std::max(
                            record.maximum_use_count, use_count);
                        ++record.observations;
                    }
                }
            }
            if (iteration + 1 == command.warmup) {
                microllm::runtime::enable_hip_caching_allocator(device);
                microllm::runtime::reset_allocation_peak(device);
            }
        }
        microllm::runtime::synchronize(device);
        const auto elapsed_ms = std::chrono::duration<double, std::milli>(
                                    std::chrono::steady_clock::now() - started)
                                    .count();
        std::size_t stable_tensors = 0;
        std::uint64_t stable_bytes = 0;
        std::uint64_t changed_bytes = 0;
        for (const auto& [name, record] : records) {
            (void)name;
            if (record.changes == 0) {
                ++stable_tensors;
                stable_bytes += record.bytes;
            } else {
                changed_bytes += record.bytes;
            }
        }
        const auto allocation = microllm::runtime::allocation_stats(device);
        const auto info = microllm::runtime::device_info(device);
        std::cout << std::setprecision(12)
                  << "{\"schema_version\":1,\"status\":\"pass\""
                  << ",\"record_type\":\"gradient_address_stability\""
                  << ",\"model\":\"" << command.model << "\""
                  << ",\"precision\":\"" << command.precision << "\""
                  << ",\"warmup\":" << command.warmup
                  << ",\"steps\":" << command.steps
                  << ",\"context\":" << command.context
                  << ",\"parameter_count\":" << model.parameter_count()
                  << ",\"parameter_tensors\":" << records.size()
                  << ",\"stable_gradient_tensors\":" << stable_tensors
                  << ",\"changed_gradient_tensors\":"
                  << records.size() - stable_tensors
                  << ",\"stable_gradient_bytes\":" << stable_bytes
                  << ",\"changed_gradient_bytes\":" << changed_bytes
                  << ",\"all_gradient_addresses_stable\":"
                  << (stable_tensors == records.size() ? "true" : "false")
                  << ",\"elapsed_ms\":" << elapsed_ms
                  << ",\"engine_allocation_calls\":"
                  << allocation.allocation_calls
                  << ",\"engine_cache_reuse_calls\":"
                  << allocation.cache_reuse_calls
                  << ",\"engine_peak_bytes\":" << allocation.peak_bytes
                  << ",\"device_name\":\"" << info.name << "\""
                  << ",\"architecture\":\"" << info.architecture << "\""
                  << ",\"losses\":[";
        for (std::size_t index = 0; index < losses.size(); ++index) {
            if (index != 0) std::cout << ',';
            std::cout << losses[index];
        }
        std::cout << "],\"records\":[";
        std::size_t index = 0;
        for (const auto& [name, record] : records) {
            if (index++ != 0) std::cout << ',';
            std::cout << "{\"name\":\"" << name << "\""
                      << ",\"elements\":" << record.elements
                      << ",\"bytes\":" << record.bytes
                      << ",\"observations\":" << record.observations
                      << ",\"address_changes\":" << record.changes
                      << ",\"address_stable\":"
                      << (record.changes == 0 ? "true" : "false")
                      << ",\"minimum_storage_use_count\":"
                      << record.minimum_use_count
                      << ",\"maximum_storage_use_count\":"
                      << record.maximum_use_count << '}';
        }
        std::cout << "]}\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "microllm_bench_gradient_address_stability: "
                  << error.what() << '\n';
        return 1;
    }
}
