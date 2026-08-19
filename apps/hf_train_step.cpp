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
        for (int index = 1; index < argc; index += 2) {
            if (index + 1 >= argc) throw std::invalid_argument("missing CLI value");
            const std::string name = argv[index];
            if (name == "--config") config_path = argv[index + 1];
            else if (name == "--weights") weights_path = argv[index + 1];
            else if (name == "--tokens") token_text = argv[index + 1];
            else if (name == "--device") device_text = argv[index + 1];
            else if (name == "--learning-rate") learning_rate = std::stof(argv[index + 1]);
            else throw std::invalid_argument("unknown option: " + name);
        }
        if (config_path.empty() || weights_path.empty() || token_text.empty()) {
            throw std::invalid_argument("--config, --weights, and --tokens are required");
        }
        const auto device = device_text == "hip" ? microllm::Device::hip(0)
                                                   : microllm::Device::cpu();
        if (device_text != "cpu" && device_text != "hip") {
            throw std::invalid_argument("--device must be cpu or hip");
        }
        const auto external = microllm::model::load_huggingface_config(config_path);
        microllm::runtime::reset_allocation_peak(device);
        microllm::model::TransformerModel model(external.model, 1);
        model.to(device);
        microllm::model::LoadWeightsOptions load_options;
        load_options.mapping = microllm::model::qwen_style_weight_mapping(external.model);
        const auto report = model.load_safetensors(weights_path, load_options);
        microllm::training::AdamW optimizer(
            model.parameters(), {.learning_rate = learning_rate,
                                 .beta1 = 0.9F,
                                 .beta2 = 0.999F,
                                 .epsilon = 1.0e-8F,
                                 .weight_decay = 0.01F});
        const auto all_tokens = parse_tokens(token_text);
        const std::vector<std::int32_t> input_ids(all_tokens.begin(), all_tokens.end() - 1);
        const std::vector<std::int32_t> target_ids(all_tokens.begin() + 1, all_tokens.end());
        auto inputs = microllm::Tensor::from_int32_vector(
            input_ids, {1, static_cast<std::int64_t>(input_ids.size())});
        auto targets = microllm::Tensor::from_int32_vector(
            target_ids, {1, static_cast<std::int64_t>(target_ids.size())});
        if (device.is_hip()) { inputs = inputs.to(device); targets = targets.to(device); }
        auto named = model.named_parameters();
        microllm::autograd::Value* observed = nullptr;
        for (const auto& [name, parameter] : named) {
            if (name == "final_norm.weight") observed = parameter;
        }
        if (observed == nullptr) throw std::logic_error("final_norm.weight is missing");
        const auto before = observed->data().to_vector().front();
        optimizer.zero_grad();
        const auto start = std::chrono::steady_clock::now();
        const auto loss = model.loss(inputs, targets);
        const auto loss_value = loss.data().to_vector()[0];
        loss.backward();
        microllm::runtime::reset_transfer_stats();
        const auto optimizer_start = std::chrono::steady_clock::now();
        optimizer.step();
        microllm::runtime::synchronize(device);
        const auto finish = std::chrono::steady_clock::now();
        const auto transfers = microllm::runtime::transfer_stats();
        const auto after = observed->data().to_vector().front();
        const auto allocation = microllm::runtime::allocation_stats(device);
        std::cout << std::setprecision(9)
                  << "{\"schema_version\":1,\"status\":\"pass\""
                  << ",\"device\":\"" << device.str() << "\""
                  << ",\"loaded_tensors\":" << report.loaded.size()
                  << ",\"parameter_count\":" << model.parameter_count()
                  << ",\"loss\":" << loss_value
                  << ",\"observed_parameter_before\":" << before
                  << ",\"observed_parameter_after\":" << after
                  << ",\"parameter_changed\":" << (before != after ? "true" : "false")
                  << ",\"step_ms\":"
                  << std::chrono::duration<double, std::milli>(finish - start).count()
                  << ",\"optimizer_ms\":"
                  << std::chrono::duration<double, std::milli>(finish - optimizer_start).count()
                  << ",\"optimizer_host_to_device_calls\":" << transfers.host_to_device_calls
                  << ",\"optimizer_device_to_host_calls\":" << transfers.device_to_host_calls
                  << ",\"engine_current_bytes\":" << allocation.current_bytes
                  << ",\"engine_peak_bytes\":" << allocation.peak_bytes << "}\n";
        return before != after && std::isfinite(loss_value) ? 0 : 2;
    } catch (const std::exception& error) {
        std::cerr << "microllm_hf_train_step: " << error.what() << '\n';
        return 1;
    }
}
