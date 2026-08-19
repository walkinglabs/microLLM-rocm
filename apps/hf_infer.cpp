#include <algorithm>
#include <chrono>
#include <charconv>
#include <cstdint>
#include <filesystem>
#include <iomanip>
#include <fstream>
#include <iostream>
#include <numeric>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <microllm/model/huggingface.h>
#include <microllm/model/model.h>
#include <microllm/io/huggingface_bpe_tokenizer.h>
#include <microllm/io/chat_template.h>
#include <microllm/runtime/runtime.h>
#include <microllm/runtime/memory.h>
#include <microllm/inference/generator.h>

namespace {

struct Options {
    std::filesystem::path config;
    std::filesystem::path weights;
    std::string tokens;
    std::string device = "cpu";
    std::int64_t top_k = 10;
    std::filesystem::path logits_output;
    std::string text;
    std::filesystem::path vocabulary;
    std::filesystem::path merges;
    std::int64_t new_tokens = 0;
    std::string tokenizer_family = "qwen2";
    std::string chat_user;
};

Options options(int argc, char** argv) {
    Options result;
    for (int index = 1; index < argc; index += 2) {
        if (index + 1 >= argc) throw std::invalid_argument("missing CLI value");
        const std::string name = argv[index];
        if (name == "--config") result.config = argv[index + 1];
        else if (name == "--weights") result.weights = argv[index + 1];
        else if (name == "--tokens") result.tokens = argv[index + 1];
        else if (name == "--device") result.device = argv[index + 1];
        else if (name == "--top-k") result.top_k = std::stoll(argv[index + 1]);
        else if (name == "--logits-output") result.logits_output = argv[index + 1];
        else if (name == "--text") result.text = argv[index + 1];
        else if (name == "--vocab") result.vocabulary = argv[index + 1];
        else if (name == "--merges") result.merges = argv[index + 1];
        else if (name == "--new-tokens") result.new_tokens = std::stoll(argv[index + 1]);
        else if (name == "--tokenizer-family") result.tokenizer_family = argv[index + 1];
        else if (name == "--chat-user") result.chat_user = argv[index + 1];
        else throw std::invalid_argument("unknown CLI option: " + name);
    }
    if (result.config.empty() || result.weights.empty()) {
        throw std::invalid_argument("--config and --weights are required");
    }
    const auto token_mode = !result.tokens.empty();
    const auto text_mode = (!result.text.empty() || !result.chat_user.empty()) &&
                           !result.vocabulary.empty() && !result.merges.empty();
    if (token_mode == text_mode) {
        throw std::invalid_argument(
            "provide either --tokens or all of --text/--vocab/--merges");
    }
    if (result.device != "cpu" && result.device != "hip") {
        throw std::invalid_argument("--device must be cpu or hip");
    }
    if (result.top_k <= 0) throw std::invalid_argument("--top-k must be positive");
    if (result.new_tokens < 0) throw std::invalid_argument("--new-tokens cannot be negative");
    return result;
}

std::vector<std::int32_t> tokens(std::string_view text) {
    std::vector<std::int32_t> output;
    while (!text.empty()) {
        const auto comma = text.find(',');
        const auto item = text.substr(0, comma);
        std::int32_t value = 0;
        const auto parsed = std::from_chars(item.data(), item.data() + item.size(), value);
        if (item.empty() || parsed.ec != std::errc{} || parsed.ptr != item.data() + item.size() ||
            value < 0) throw std::invalid_argument("--tokens must be comma-separated nonnegative IDs");
        output.push_back(value);
        if (comma == std::string_view::npos) break;
        text.remove_prefix(comma + 1);
    }
    return output;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const auto command = options(argc, argv);
        auto external = microllm::model::load_huggingface_config(command.config);
        const auto device = command.device == "hip" ? microllm::Device::hip(0)
                                                     : microllm::Device::cpu();
        if (device.is_hip() && microllm::runtime::hip_device_count() == 0) {
            throw std::runtime_error("HIP inference requested without a visible device");
        }
        microllm::runtime::reset_allocation_peak(device);
        microllm::model::TransformerModel model(external.model, 1);
        model.to(device);
        microllm::model::LoadWeightsOptions load_options;
        load_options.mapping = microllm::model::qwen_style_weight_mapping(external.model);
        const auto load_start = std::chrono::steady_clock::now();
        const auto report = model.load_safetensors(command.weights, load_options);
        const auto load_finish = std::chrono::steady_clock::now();
        std::optional<microllm::io::HuggingFaceBpeTokenizer> tokenizer;
        std::vector<std::int32_t> ids;
        if (!command.tokens.empty()) {
            ids = tokens(command.tokens);
        } else {
            tokenizer = microllm::io::HuggingFaceBpeTokenizer::load(
                command.vocabulary, command.merges);
            if (command.tokenizer_family == "deepseek-distill") {
                tokenizer->add_special_token("<｜end▁of▁sentence｜>", 151643);
                tokenizer->add_special_token("<｜User｜>", 151644);
                tokenizer->add_special_token("<｜Assistant｜>", 151645);
                tokenizer->add_special_token("<｜begin▁of▁sentence｜>", 151646);
                tokenizer->add_special_token("<think>", 151648);
                tokenizer->add_special_token("</think>", 151649);
            } else if (command.tokenizer_family == "qwen2") {
                tokenizer->add_special_token("<|endoftext|>", 151643);
                tokenizer->add_special_token("<|im_start|>", 151644);
                tokenizer->add_special_token("<|im_end|>", 151645);
            } else {
                throw std::invalid_argument("unknown tokenizer family");
            }
            const auto prompt = command.chat_user.empty()
                                    ? command.text
                                    : command.tokenizer_family == "deepseek-distill"
                                          ? microllm::io::render_deepseek_distill_chat(
                                                {{"user", command.chat_user}})
                                          : microllm::io::render_qwen2_chat(
                                                {{"user", command.chat_user}});
            ids = tokenizer->encode(prompt);
        }
        if (ids.size() > static_cast<std::size_t>(external.model.max_sequence_length)) {
            throw std::invalid_argument("token sequence exceeds model context");
        }
        auto token_tensor = microllm::Tensor::from_int32_vector(
            ids, {1, static_cast<std::int64_t>(ids.size())});
        if (device.is_hip()) token_tensor = token_tensor.to(device);
        const auto forward_start = std::chrono::steady_clock::now();
        const auto logits = model.forward(token_tensor).data().to_vector();
        const auto forward_finish = std::chrono::steady_clock::now();
        const auto allocation = microllm::runtime::allocation_stats(device);
        const auto vocabulary = static_cast<std::size_t>(external.model.vocabulary_size);
        const auto offset = (ids.size() - 1) * vocabulary;
        std::vector<std::size_t> order(vocabulary);
        std::iota(order.begin(), order.end(), 0U);
        const auto selected = std::min<std::size_t>(static_cast<std::size_t>(command.top_k),
                                                    vocabulary);
        if (!command.logits_output.empty()) {
            std::ofstream output(command.logits_output, std::ios::binary | std::ios::trunc);
            if (!output) throw std::runtime_error("cannot open logits output");
            output.write(reinterpret_cast<const char*>(logits.data() + offset),
                         static_cast<std::streamsize>(vocabulary * sizeof(float)));
            if (!output) throw std::runtime_error("failed writing logits output");
        }
        std::partial_sort(order.begin(), order.begin() + static_cast<std::ptrdiff_t>(selected),
                          order.end(), [&](std::size_t left, std::size_t right) {
                              return logits[offset + left] > logits[offset + right];
                          });
        std::cout << std::setprecision(9)
                  << "{\"schema_version\":1,\"status\":\"pass\""
                  << ",\"device\":\"" << device.str() << "\""
                  << ",\"parameter_count\":" << model.parameter_count()
                  << ",\"loaded_tensors\":" << report.loaded.size()
                  << ",\"token_count\":" << ids.size()
                  << ",\"load_ms\":"
                  << std::chrono::duration<double, std::milli>(load_finish - load_start).count()
                  << ",\"forward_ms\":"
                  << std::chrono::duration<double, std::milli>(forward_finish - forward_start).count()
                  << ",\"engine_current_bytes\":" << allocation.current_bytes
                  << ",\"engine_peak_bytes\":" << allocation.peak_bytes
                  << ",\"top_logits\":[";
        for (std::size_t index = 0; index < selected; ++index) {
            if (index != 0) std::cout << ',';
            std::cout << "{\"token\":" << order[index]
                      << ",\"logit\":" << logits[offset + order[index]] << '}';
        }
        std::cout << ']';
        if (command.new_tokens > 0) {
            const auto generation_start = std::chrono::steady_clock::now();
            const auto generated = microllm::inference::generate(
                model, ids, {.max_new_tokens = command.new_tokens,
                             .temperature = 0.0F,
                             .top_k = 1,
                             .seed = 1});
            const auto generation_finish = std::chrono::steady_clock::now();
            std::cout << ",\"generation_ms\":"
                      << std::chrono::duration<double, std::milli>(
                             generation_finish - generation_start).count();
            std::cout << ",\"generated_tokens\":[";
            for (std::size_t index = ids.size(); index < generated.size(); ++index) {
                if (index != ids.size()) std::cout << ',';
                std::cout << generated[index];
            }
            std::cout << ']';
            if (tokenizer.has_value()) {
                std::cout << ",\"generated_text\":\"";
                const std::vector<std::int32_t> suffix(
                    generated.begin() + static_cast<std::ptrdiff_t>(ids.size()),
                    generated.end());
                for (const auto character : tokenizer->decode(suffix)) {
                    if (character == '"' || character == '\\') std::cout << '\\';
                    if (character == '\n') std::cout << "\\n";
                    else if (character == '\r') std::cout << "\\r";
                    else if (character == '\t') std::cout << "\\t";
                    else std::cout << character;
                }
                std::cout << '"';
            }
        }
        std::cout << "}\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "microllm_hf_infer: " << error.what() << '\n';
        return 1;
    }
}
