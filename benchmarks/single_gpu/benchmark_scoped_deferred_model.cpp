#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

#include <microllm/autograd/autograd.h>
#include <microllm/core/tensor.h>
#include <microllm/model/huggingface.h>
#include <microllm/model/model.h>
#include <microllm/runtime/memory.h>
#include <microllm/runtime/runtime.h>
#include <microllm/training/optimizer.h>

namespace {

struct Options {
    std::filesystem::path config;
    std::filesystem::path weights;
    std::filesystem::path logits_output;
    std::string model;
    std::string mode = "inference";
    std::string policy = "legacy";
    std::string precision = "bf16";
    std::int64_t context = 512;
    std::int64_t batch = 1;
    int warmup = 1;
    int steps = 2;
    std::size_t maximum_blocks = 8192;
};

Options parse_options(int argc, char** argv) {
    Options result;
    for (int index = 1; index < argc; index += 2) {
        if (index + 1 >= argc) throw std::invalid_argument("missing CLI value");
        const std::string name = argv[index];
        const std::string value = argv[index + 1];
        if (name == "--config") result.config = value;
        else if (name == "--weights") result.weights = value;
        else if (name == "--logits-output") result.logits_output = value;
        else if (name == "--model") result.model = value;
        else if (name == "--mode") result.mode = value;
        else if (name == "--policy") result.policy = value;
        else if (name == "--precision") result.precision = value;
        else if (name == "--context") result.context = std::stoll(value);
        else if (name == "--batch") result.batch = std::stoll(value);
        else if (name == "--warmup") result.warmup = std::stoi(value);
        else if (name == "--steps") result.steps = std::stoi(value);
        else if (name == "--maximum-blocks") {
            result.maximum_blocks = static_cast<std::size_t>(std::stoull(value));
        } else {
            throw std::invalid_argument("unknown option: " + name);
        }
    }
    if (result.config.empty() || result.weights.empty() || result.model.empty()) {
        throw std::invalid_argument("--config, --weights, and --model are required");
    }
    if (result.mode != "inference" && result.mode != "training") {
        throw std::invalid_argument("--mode must be inference or training");
    }
    if (result.policy != "legacy" && result.policy != "deferred") {
        throw std::invalid_argument("--policy must be legacy or deferred");
    }
    if (result.precision != "fp32" && result.precision != "bf16") {
        throw std::invalid_argument("--precision must be fp32 or bf16");
    }
    if (result.context <= 0 || result.batch <= 0 || result.warmup < 0 ||
        result.steps <= 0 || result.maximum_blocks == 0) {
        throw std::invalid_argument(
            "context, batch, steps and capacity must be positive; warmup is nonnegative");
    }
    if (result.mode == "training" && !result.logits_output.empty()) {
        throw std::invalid_argument("training does not write logits");
    }
    return result;
}

struct DeferredStats {
    std::uint64_t blocks = 0;
    std::uint64_t bytes = 0;
    std::uint64_t maximum_bytes = 0;
    std::uint64_t overflow_flushes = 0;
};

void add_stats(DeferredStats& destination,
               const microllm::runtime::ScopedDeferredHipStream& scope) {
    destination.blocks += scope.total_deferred_blocks();
    destination.bytes += scope.total_deferred_bytes();
    destination.maximum_bytes = std::max<std::uint64_t>(
        destination.maximum_bytes, scope.total_deferred_bytes());
    destination.overflow_flushes += scope.overflow_flushes();
}

std::vector<std::int32_t> make_tokens(
    std::int64_t batch, std::int64_t context, std::int64_t vocabulary,
    std::int64_t offset) {
    std::vector<std::int32_t> values;
    values.reserve(static_cast<std::size_t>(batch * context));
    for (std::int64_t row = 0; row < batch; ++row) {
        for (std::int64_t position = 0; position < context; ++position) {
            values.push_back(static_cast<std::int32_t>(
                (offset + row * 53 + position * 37) % vocabulary));
        }
    }
    return values;
}

void write_logits(const std::filesystem::path& path,
                  const std::vector<float>& values) {
    if (path.empty()) return;
    if (path.has_parent_path()) std::filesystem::create_directories(path.parent_path());
    std::ofstream output(path, std::ios::binary | std::ios::trunc);
    if (!output) throw std::runtime_error("cannot open logits output");
    output.write(reinterpret_cast<const char*>(values.data()),
                 static_cast<std::streamsize>(values.size() * sizeof(float)));
    if (!output) throw std::runtime_error("cannot write logits output");
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const auto command = parse_options(argc, argv);
        if (!microllm::runtime::hip_compiled() ||
            microllm::runtime::hip_device_count() == 0) {
            throw std::runtime_error("benchmark requires a visible HIP device");
        }
        const auto device = microllm::Device::hip(0);
        auto external = microllm::model::load_huggingface_config(command.config);
        if (command.context > external.model.max_sequence_length) {
            throw std::invalid_argument("context exceeds model maximum sequence length");
        }
        if (command.mode == "training" && command.precision == "bf16") {
            external.model.linear_precision =
                microllm::model::LinearPrecision::BFloat16;
        }
        microllm::model::TransformerModel model(
            external.model, 1,
            microllm::model::ParameterInitialization::Uninitialized);
        model.to(device);
        microllm::model::LoadWeightsOptions load_options;
        load_options.mapping =
            microllm::model::qwen_style_weight_mapping(external.model);
        load_options.aliases =
            microllm::model::qwen3_tied_weight_aliases(external.model);
        const auto load_report = model.load_safetensors(command.weights, load_options);
        microllm::runtime::synchronize(device);

        microllm::model::Bf16TrainingMirrors mirrors;
        if (command.precision == "bf16") {
            if (command.mode == "inference") {
                (void)model.prepare_bf16_ffn_inference();
                (void)model.prepare_bf16_attention_inference();
            } else {
                mirrors = model.prepare_bf16_training_mirrors();
            }
        }
        std::unique_ptr<microllm::training::AdamW> optimizer;
        if (command.mode == "training") {
            optimizer = std::make_unique<microllm::training::AdamW>(
                model.parameters(),
                microllm::training::AdamWConfig{.learning_rate = 1.0e-5F,
                                                 .beta1 = 0.9F,
                                                 .beta2 = 0.999F,
                                                 .epsilon = 1.0e-8F,
                                                 .weight_decay = 0.01F},
                mirrors);
        }
        microllm::runtime::synchronize(device);

        const auto input_values = make_tokens(
            command.batch, command.context, external.model.vocabulary_size, 11);
        auto inputs = microllm::Tensor::from_int32_vector(
            input_values, {command.batch, command.context}).to(device);
        microllm::Tensor targets;
        if (command.mode == "training") {
            const auto target_values = make_tokens(
                command.batch, command.context, external.model.vocabulary_size, 48);
            targets = microllm::Tensor::from_int32_vector(
                target_values, {command.batch, command.context}).to(device);
        }
        std::optional<microllm::runtime::Stream> candidate_stream;
        if (command.policy == "deferred") candidate_stream.emplace(device);

        microllm::autograd::Value* observed = nullptr;
        if (command.mode == "training") {
            for (const auto& [name, parameter] : model.named_parameters()) {
                if (name == "final_norm.weight") observed = parameter;
            }
            if (observed == nullptr) {
                throw std::logic_error("final_norm.weight is missing");
            }
        }

        microllm::Tensor last_logits;
        float last_loss = 0.0F;
        DeferredStats warmup_deferred;
        DeferredStats measured_deferred;
        const auto run_inference = [&](DeferredStats& deferred) {
            microllm::Tensor output;
            if (candidate_stream.has_value()) {
                microllm::runtime::ScopedDeferredHipStream scope(
                    *candidate_stream, command.maximum_blocks);
                output = model.forward_inference_last_logits(inputs);
                add_stats(deferred, scope);
                scope.finish();
            } else {
                output = model.forward_inference_last_logits(inputs);
                microllm::runtime::synchronize(device);
            }
            return output;
        };
        const auto run_training = [&](DeferredStats& deferred) {
            microllm::Tensor loss_tensor;
            if (candidate_stream.has_value()) {
                microllm::runtime::ScopedDeferredHipStream scope(
                    *candidate_stream, command.maximum_blocks);
                optimizer->zero_grad();
                {
                    auto loss = model.loss(inputs, targets);
                    loss_tensor = loss.data();
                    loss.backward();
                    optimizer->step();
                }
                add_stats(deferred, scope);
                scope.finish();
            } else {
                optimizer->zero_grad();
                {
                    auto loss = model.loss(inputs, targets);
                    loss_tensor = loss.data();
                    loss.backward();
                    optimizer->step();
                }
                microllm::runtime::synchronize(device);
            }
            return loss_tensor.to_vector().front();
        };

        for (int iteration = 0; iteration < command.warmup; ++iteration) {
            if (command.mode == "inference") {
                last_logits = run_inference(warmup_deferred);
            } else {
                last_loss = run_training(warmup_deferred);
            }
        }
        if (command.policy == "legacy") {
            microllm::runtime::enable_hip_caching_allocator(device);
        }
        microllm::runtime::reset_allocation_peak(device);
        microllm::runtime::reset_transfer_stats();
        const auto observed_before = observed == nullptr
                                         ? 0.0F
                                         : observed->data().to_vector().front();
        const auto start = std::chrono::steady_clock::now();
        for (int iteration = 0; iteration < command.steps; ++iteration) {
            if (command.mode == "inference") {
                last_logits = run_inference(measured_deferred);
            } else {
                last_loss = run_training(measured_deferred);
            }
        }
        const auto finish = std::chrono::steady_clock::now();
        const auto observed_after = observed == nullptr
                                        ? 0.0F
                                        : observed->data().to_vector().front();
        const auto transfers = microllm::runtime::transfer_stats();
        const auto allocation = microllm::runtime::allocation_stats(device);
        const auto elapsed_ms = std::chrono::duration<double, std::milli>(
                                    finish - start).count();
        std::vector<float> logits;
        std::size_t top_index = 0;
        float top_value = 0.0F;
        double logits_sum = 0.0;
        double logits_square_sum = 0.0;
        if (last_logits.defined()) {
            logits = last_logits.to_vector();
            top_index = static_cast<std::size_t>(
                std::max_element(logits.begin(), logits.end()) - logits.begin());
            top_value = logits[top_index];
            for (const auto value : logits) {
                logits_sum += value;
                logits_square_sum += static_cast<double>(value) * value;
            }
            write_logits(command.logits_output, logits);
        }
        const auto tokens = static_cast<std::uint64_t>(command.batch) *
                            static_cast<std::uint64_t>(command.context) *
                            static_cast<std::uint64_t>(command.steps);
        const auto info = microllm::runtime::device_info(device);
        std::cout << std::setprecision(12)
                  << "{\"schema_version\":1,\"status\":\"pass\""
                  << ",\"record_type\":\"scoped_deferred_model_measurement\""
                  << ",\"model\":\"" << command.model << "\""
                  << ",\"mode\":\"" << command.mode << "\""
                  << ",\"policy\":\"" << command.policy << "\""
                  << ",\"precision\":\"" << command.precision << "\""
                  << ",\"device\":\"" << device.str() << "\""
                  << ",\"architecture\":\"" << info.architecture << "\""
                  << ",\"loaded_tensors\":" << load_report.loaded.size()
                  << ",\"parameter_count\":" << model.parameter_count()
                  << ",\"batch\":" << command.batch
                  << ",\"context\":" << command.context
                  << ",\"warmup\":" << command.warmup
                  << ",\"steps\":" << command.steps
                  << ",\"measured_tokens\":" << tokens
                  << ",\"measured_ms\":" << elapsed_ms
                  << ",\"tokens_per_second\":"
                  << static_cast<double>(tokens) * 1000.0 / elapsed_ms
                  << ",\"loss\":" << last_loss
                  << ",\"observed_parameter_before\":" << observed_before
                  << ",\"observed_parameter_after\":" << observed_after
                  << ",\"parameter_changed\":"
                  << (observed_before != observed_after ? "true" : "false")
                  << ",\"logit_count\":" << logits.size()
                  << ",\"top_index\":" << top_index
                  << ",\"top_value\":" << top_value
                  << ",\"logits_sum\":" << logits_sum
                  << ",\"logits_square_sum\":" << logits_square_sum
                  << ",\"deferred_blocks\":" << measured_deferred.blocks
                  << ",\"deferred_bytes\":" << measured_deferred.bytes
                  << ",\"maximum_deferred_bytes\":"
                  << measured_deferred.maximum_bytes
                  << ",\"deferred_overflow_flushes\":"
                  << measured_deferred.overflow_flushes
                  << ",\"warmup_deferred_bytes\":" << warmup_deferred.bytes
                  << ",\"engine_current_bytes\":" << allocation.current_bytes
                  << ",\"engine_peak_bytes\":" << allocation.peak_bytes
                  << ",\"engine_allocation_calls\":" << allocation.allocation_calls
                  << ",\"engine_deallocation_calls\":" << allocation.deallocation_calls
                  << ",\"engine_backend_allocation_calls\":"
                  << allocation.backend_allocation_calls
                  << ",\"engine_backend_deallocation_calls\":"
                  << allocation.backend_deallocation_calls
                  << ",\"engine_cache_reuse_calls\":" << allocation.cache_reuse_calls
                  << ",\"engine_cached_bytes\":" << allocation.cached_bytes
                  << ",\"engine_reserved_bytes\":" << allocation.reserved_bytes
                  << ",\"measured_h2d_calls\":" << transfers.host_to_device_calls
                  << ",\"measured_d2h_calls\":" << transfers.device_to_host_calls
                  << ",\"measured_d2d_calls\":" << transfers.device_to_device_calls
                  << "}\n";
        const auto valid_training = command.mode != "training" ||
                                    (std::isfinite(last_loss) &&
                                     observed_before != observed_after);
        return valid_training ? 0 : 2;
    } catch (const std::exception& error) {
        std::cerr << "microllm_bench_scoped_deferred_model: "
                  << error.what() << '\n';
        return 1;
    }
}
