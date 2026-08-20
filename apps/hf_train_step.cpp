#include <chrono>
#include <charconv>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include <microllm/model/huggingface.h>
#include <microllm/model/model.h>
#include <microllm/runtime/memory.h>
#include <microllm/runtime/runtime.h>
#include <microllm/training/optimizer.h>

namespace {

struct StepResult {
    float loss = 0.0F;
    double optimizer_ms = 0.0;
    microllm::runtime::TransferStats transfers;
};

std::vector<std::int32_t> parse_tokens(std::string_view text) {
    std::vector<std::int32_t> output;
    while (!text.empty()) {
        const auto comma = text.find(',');
        const auto item = text.substr(0, comma);
        std::int32_t value = 0;
        const auto parsed = std::from_chars(item.data(), item.data() + item.size(), value);
        if (item.empty() || parsed.ec != std::errc{} ||
            parsed.ptr != item.data() + item.size() || value < 0) {
            throw std::invalid_argument("tokens must be comma-separated nonnegative IDs");
        }
        output.push_back(value);
        if (comma == std::string_view::npos) break;
        text.remove_prefix(comma + 1);
    }
    if (output.size() < 2) throw std::invalid_argument("training requires at least two tokens");
    return output;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        std::filesystem::path config_path;
        std::filesystem::path weights_path;
        std::string token_text;
        std::string device_text = "hip";
        float learning_rate = 1.0e-5F;
        int warmup = 0;
        int steps = 1;
        int batch_size = 1;
        std::string linear_precision = "fp32";
        bool bf16_weight_mirrors = true;
        for (int index = 1; index < argc; index += 2) {
            if (index + 1 >= argc) throw std::invalid_argument("missing CLI value");
            const std::string name = argv[index];
            if (name == "--config") config_path = argv[index + 1];
            else if (name == "--weights") weights_path = argv[index + 1];
            else if (name == "--tokens") token_text = argv[index + 1];
            else if (name == "--device") device_text = argv[index + 1];
            else if (name == "--learning-rate") learning_rate = std::stof(argv[index + 1]);
            else if (name == "--warmup") warmup = std::stoi(argv[index + 1]);
            else if (name == "--steps") steps = std::stoi(argv[index + 1]);
            else if (name == "--batch") batch_size = std::stoi(argv[index + 1]);
            else if (name == "--linear-precision") linear_precision = argv[index + 1];
            else if (name == "--bf16-weight-mirrors") {
                const std::string value = argv[index + 1];
                if (value != "true" && value != "false") {
                    throw std::invalid_argument(
                        "--bf16-weight-mirrors must be true or false");
                }
                bf16_weight_mirrors = value == "true";
            }
            else throw std::invalid_argument("unknown option: " + name);
        }
        if (config_path.empty() || weights_path.empty() || token_text.empty()) {
            throw std::invalid_argument("--config, --weights, and --tokens are required");
        }
        if (warmup < 0 || steps <= 0 || batch_size <= 0) {
            throw std::invalid_argument(
                "--warmup must be nonnegative; --steps and --batch must be positive");
        }
        if (linear_precision != "fp32" && linear_precision != "bf16") {
            throw std::invalid_argument("--linear-precision must be fp32 or bf16");
        }
        const auto device = device_text == "hip" ? microllm::Device::hip(0)
                                                   : microllm::Device::cpu();
        if (device_text != "cpu" && device_text != "hip") {
            throw std::invalid_argument("--device must be cpu or hip");
        }
        auto external = microllm::model::load_huggingface_config(config_path);
        if (linear_precision == "bf16") {
            external.model.linear_precision = microllm::model::LinearPrecision::BFloat16;
        }
        microllm::runtime::reset_allocation_peak(device);
        microllm::model::TransformerModel model(external.model, 1);
        model.to(device);
        microllm::model::LoadWeightsOptions load_options;
        load_options.mapping = microllm::model::qwen_style_weight_mapping(external.model);
        const auto report = model.load_safetensors(weights_path, load_options);
        microllm::model::Bf16TrainingMirrors bf16_mirrors;
        if (linear_precision == "bf16" && bf16_weight_mirrors) {
            bf16_mirrors = model.prepare_bf16_training_mirrors();
        }
        microllm::training::AdamW optimizer(
            model.parameters(), {.learning_rate = learning_rate,
                                 .beta1 = 0.9F,
                                 .beta2 = 0.999F,
                                 .epsilon = 1.0e-8F,
                                 .weight_decay = 0.01F},
            bf16_mirrors);
        const auto all_tokens = parse_tokens(token_text);
        const std::vector<std::int32_t> input_ids(all_tokens.begin(), all_tokens.end() - 1);
        const std::vector<std::int32_t> target_ids(all_tokens.begin() + 1, all_tokens.end());
        std::vector<std::int32_t> batched_inputs;
        std::vector<std::int32_t> batched_targets;
        batched_inputs.reserve(input_ids.size() * static_cast<std::size_t>(batch_size));
        batched_targets.reserve(target_ids.size() * static_cast<std::size_t>(batch_size));
        for (int batch = 0; batch < batch_size; ++batch) {
            batched_inputs.insert(batched_inputs.end(), input_ids.begin(), input_ids.end());
            batched_targets.insert(batched_targets.end(), target_ids.begin(), target_ids.end());
        }
        auto inputs = microllm::Tensor::from_int32_vector(
            batched_inputs, {batch_size, static_cast<std::int64_t>(input_ids.size())});
        auto targets = microllm::Tensor::from_int32_vector(
            batched_targets, {batch_size, static_cast<std::int64_t>(target_ids.size())});
        if (device.is_hip()) { inputs = inputs.to(device); targets = targets.to(device); }
        auto named = model.named_parameters();
        microllm::autograd::Value* observed = nullptr;
        for (const auto& [name, parameter] : named) {
            if (name == "final_norm.weight") observed = parameter;
        }
        if (observed == nullptr) throw std::logic_error("final_norm.weight is missing");
        const auto run_step = [&]() {
            optimizer.zero_grad();
            const auto loss = model.loss(inputs, targets);
            const auto loss_value = loss.data().to_vector()[0];
            loss.backward();
            microllm::runtime::reset_transfer_stats();
            const auto optimizer_start = std::chrono::steady_clock::now();
            optimizer.step();
            microllm::runtime::synchronize(device);
            const auto optimizer_finish = std::chrono::steady_clock::now();
            return StepResult{
                loss_value,
                std::chrono::duration<double, std::milli>(optimizer_finish - optimizer_start)
                    .count(),
                microllm::runtime::transfer_stats()};
        };
        const auto warmup_start = std::chrono::steady_clock::now();
        for (int iteration = 0; iteration < warmup; ++iteration) (void)run_step();
        microllm::runtime::synchronize(device);
        const auto warmup_finish = std::chrono::steady_clock::now();
        if (device.is_hip()) microllm::runtime::enable_hip_caching_allocator(device);
        microllm::runtime::reset_allocation_peak(device);
        const auto before = observed->data().to_vector().front();
        float first_loss = 0.0F;
        float final_loss = 0.0F;
        double optimizer_ms = 0.0;
        std::uint64_t optimizer_h2d = 0;
        std::uint64_t optimizer_d2h = 0;
        const auto start = std::chrono::steady_clock::now();
        for (int iteration = 0; iteration < steps; ++iteration) {
            const auto result = run_step();
            if (iteration == 0) first_loss = result.loss;
            final_loss = result.loss;
            optimizer_ms += result.optimizer_ms;
            optimizer_h2d += result.transfers.host_to_device_calls;
            optimizer_d2h += result.transfers.device_to_host_calls;
        }
        const auto finish = std::chrono::steady_clock::now();
        const auto after = observed->data().to_vector().front();
        const auto allocation = microllm::runtime::allocation_stats(device);
        const auto info = device.is_cpu()
                              ? microllm::runtime::DeviceInfo{device, "host CPU", "host"}
                              : microllm::runtime::device_info(device);
        const auto measured_ms =
            std::chrono::duration<double, std::milli>(finish - start).count();
        const auto warmup_ms =
            std::chrono::duration<double, std::milli>(warmup_finish - warmup_start).count();
        const auto trained_tokens = input_ids.size() * static_cast<std::size_t>(batch_size) *
                                    static_cast<std::size_t>(steps);
        std::cout << std::setprecision(9)
                  << "{\"schema_version\":1,\"status\":\"pass\""
                  << ",\"device\":\"" << device.str() << "\""
                  << ",\"device_name\":\"" << info.name << "\""
                  << ",\"architecture\":\"" << info.architecture << "\""
                  << ",\"hip_runtime_version\":"
                  << microllm::runtime::hip_runtime_version()
                  << ",\"hip_driver_version\":" << microllm::runtime::hip_driver_version()
                  << ",\"compute_dtype\":\""
                  << (linear_precision == "bf16" ? "bf16_linear_fp32_master" : "float32")
                  << "\""
                  << ",\"bf16_weight_mirrors_enabled\":"
                  << (!bf16_mirrors.empty() ? "true" : "false")
                  << ",\"measurement_profile\":\""
                  << (warmup > 0 || steps > 1 ? "comparison" : "smoke") << "\""
                  << ",\"loaded_tensors\":" << report.loaded.size()
                  << ",\"parameter_count\":" << model.parameter_count()
                  << ",\"fp32_weight_bytes\":"
                  << external.model.weight_bytes(sizeof(float))
                  << ",\"bf16_training_mirror_tensors\":" << bf16_mirrors.size()
                  << ",\"bf16_training_mirror_bytes\":"
                  << [&] {
                         std::uint64_t bytes = 0;
                         for (const auto& [master, mirror] : bf16_mirrors) {
                             (void)master;
                             bytes += static_cast<std::uint64_t>(mirror->storage().num_bytes());
                         }
                         return bytes;
                     }()
                  << ",\"warmup\":" << warmup
                  << ",\"steps\":" << steps
                  << ",\"batch\":" << batch_size
                  << ",\"context\":" << input_ids.size()
                  << ",\"warmup_ms\":" << warmup_ms
                  << ",\"trained_tokens\":" << trained_tokens
                  << ",\"first_loss\":" << first_loss
                  << ",\"final_loss\":" << final_loss
                  << ",\"loss\":" << final_loss
                  << ",\"observed_parameter_before\":" << before
                  << ",\"observed_parameter_after\":" << after
                  << ",\"parameter_changed\":" << (before != after ? "true" : "false")
                  << ",\"step_ms\":" << measured_ms
                  << ",\"measured_ms\":" << measured_ms
                  << ",\"mean_step_ms\":" << measured_ms / static_cast<double>(steps)
                  << ",\"tokens_per_second\":"
                  << static_cast<double>(trained_tokens) * 1000.0 / measured_ms
                  << ",\"milliseconds_per_token\":"
                  << measured_ms / static_cast<double>(trained_tokens)
                  << ",\"optimizer_ms\":" << optimizer_ms
                  << ",\"mean_optimizer_ms\":" << optimizer_ms / static_cast<double>(steps)
                  << ",\"optimizer_host_to_device_calls\":" << optimizer_h2d
                  << ",\"optimizer_device_to_host_calls\":" << optimizer_d2h
                  << ",\"engine_current_bytes\":" << allocation.current_bytes
                  << ",\"engine_peak_bytes\":" << allocation.peak_bytes
                  << ",\"engine_total_allocated_bytes\":"
                  << allocation.total_allocated_bytes
                  << ",\"engine_allocation_calls\":" << allocation.allocation_calls
                  << ",\"engine_deallocation_calls\":" << allocation.deallocation_calls
                  << ",\"engine_backend_allocation_calls\":"
                  << allocation.backend_allocation_calls
                  << ",\"engine_backend_deallocation_calls\":"
                  << allocation.backend_deallocation_calls
                  << ",\"engine_cache_reuse_calls\":" << allocation.cache_reuse_calls
                  << ",\"engine_cached_bytes\":" << allocation.cached_bytes
                  << ",\"engine_reserved_bytes\":" << allocation.reserved_bytes
                  << "}\n";
        return before != after && std::isfinite(final_loss) ? 0 : 2;
    } catch (const std::exception& error) {
        std::cerr << "microllm_hf_train_step: " << error.what() << '\n';
        return 1;
    }
}
