#include <microllm/model/huggingface.h>

#include <charconv>
#include <cmath>
#include <fstream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string_view>

namespace microllm::model {
namespace {

void whitespace(std::string_view text, std::size_t& position) {
    while (position < text.size() &&
           (text[position] == ' ' || text[position] == '\n' ||
            text[position] == '\r' || text[position] == '\t')) ++position;
}

std::string parse_string(std::string_view text, std::size_t& position) {
    whitespace(text, position);
    if (position >= text.size() || text[position++] != '"') {
        throw std::runtime_error("Hugging Face JSON expected a string");
    }
    std::string output;
    while (position < text.size()) {
        const auto character = text[position++];
        if (character == '"') return output;
        if (character == '\\') {
            if (position >= text.size()) throw std::runtime_error("unfinished JSON escape");
            const auto escaped = text[position++];
            switch (escaped) {
                case '"': output.push_back('"'); break;
                case '\\': output.push_back('\\'); break;
                case '/': output.push_back('/'); break;
                case 'b': output.push_back('\b'); break;
                case 'f': output.push_back('\f'); break;
                case 'n': output.push_back('\n'); break;
                case 'r': output.push_back('\r'); break;
                case 't': output.push_back('\t'); break;
                default: throw std::runtime_error("unsupported JSON escape in config");
            }
        } else {
            output.push_back(character);
        }
    }
    throw std::runtime_error("unterminated JSON string");
}

void skip_value(std::string_view text, std::size_t& position) {
    whitespace(text, position);
    if (position >= text.size()) throw std::runtime_error("missing JSON value");
    if (text[position] == '"') {
        (void)parse_string(text, position);
        return;
    }
    if (text[position] == '{' || text[position] == '[') {
        const auto open = text[position++];
        const auto close = open == '{' ? '}' : ']';
        int depth = 1;
        while (position < text.size() && depth != 0) {
            if (text[position] == '"') {
                (void)parse_string(text, position);
            } else {
                if (text[position] == open) ++depth;
                if (text[position] == close) --depth;
                ++position;
            }
        }
        if (depth != 0) throw std::runtime_error("unterminated nested JSON value");
        return;
    }
    while (position < text.size() && text[position] != ',' && text[position] != '}') ++position;
}

std::string_view member(std::string_view text, std::string_view wanted) {
    std::size_t position = 0;
    whitespace(text, position);
    if (position >= text.size() || text[position++] != '{') {
        throw std::runtime_error("Hugging Face config root must be an object");
    }
    while (true) {
        whitespace(text, position);
        if (position >= text.size()) throw std::runtime_error("unterminated config object");
        if (text[position] == '}') break;
        const auto key = parse_string(text, position);
        whitespace(text, position);
        if (position >= text.size() || text[position++] != ':') {
            throw std::runtime_error("Hugging Face config member is missing ':'");
        }
        whitespace(text, position);
        const auto begin = position;
        skip_value(text, position);
        auto end = position;
        while (end > begin && (text[end - 1] == ' ' || text[end - 1] == '\n' ||
                               text[end - 1] == '\r' || text[end - 1] == '\t')) --end;
        if (key == wanted) return text.substr(begin, end - begin);
        whitespace(text, position);
        if (position < text.size() && text[position] == ',') ++position;
    }
    throw std::runtime_error("Hugging Face config is missing " + std::string(wanted));
}

std::int64_t integer(std::string_view text, std::string_view key) {
    const auto value = member(text, key);
    std::int64_t output = 0;
    const auto result = std::from_chars(value.data(), value.data() + value.size(), output);
    if (result.ec != std::errc{} || result.ptr != value.data() + value.size()) {
        throw std::runtime_error(std::string(key) + " must be an integer");
    }
    return output;
}

double number(std::string_view text, std::string_view key) {
    const std::string value(member(text, key));
    std::size_t consumed = 0;
    const auto output = std::stod(value, &consumed);
    if (consumed != value.size() || !std::isfinite(output)) {
        throw std::runtime_error(std::string(key) + " must be finite");
    }
    return output;
}

bool boolean(std::string_view text, std::string_view key) {
    const auto value = member(text, key);
    if (value == "true") return true;
    if (value == "false") return false;
    throw std::runtime_error(std::string(key) + " must be boolean");
}

std::string string_value(std::string_view text, std::string_view key) {
    const auto value = member(text, key);
    std::size_t position = 0;
    auto output = parse_string(value, position);
    whitespace(value, position);
    if (position != value.size()) throw std::runtime_error(std::string(key) + " is invalid");
    return output;
}

}  // namespace

HuggingFaceModelConfig load_huggingface_config(const std::filesystem::path& path) {
    std::ifstream input(path, std::ios::binary);
    if (!input) throw std::runtime_error("cannot open Hugging Face config: " + path.string());
    std::ostringstream buffer;
    buffer << input.rdbuf();
    const auto text = buffer.str();
    if (text.empty() || text.size() > 1024U * 1024U) {
        throw std::runtime_error("Hugging Face config size is invalid");
    }
    HuggingFaceModelConfig output;
    output.model_type = string_value(text, "model_type");
    if (output.model_type != "qwen2") {
        throw std::invalid_argument("first Hugging Face target requires model_type=qwen2");
    }
    if (string_value(text, "hidden_act") != "silu") {
        throw std::invalid_argument("Qwen2 hidden_act must be silu");
    }
    if (boolean(text, "use_sliding_window") || boolean(text, "use_mrope")) {
        throw std::invalid_argument("sliding-window and MRoPE Qwen variants are not supported");
    }
    output.model = {.vocabulary_size = integer(text, "vocab_size"),
                    .dimension = integer(text, "hidden_size"),
                    .layers = integer(text, "num_hidden_layers"),
                    .heads = integer(text, "num_attention_heads"),
                    .kv_heads = integer(text, "num_key_value_heads"),
                    .ffn_dimension = integer(text, "intermediate_size"),
                    .max_sequence_length = integer(text, "max_position_embeddings"),
                    .rope_base = static_cast<float>(number(text, "rope_theta")),
                    .tie_embeddings = boolean(text, "tie_word_embeddings"),
                    .linear_precision = LinearPrecision::Float32,
                    .fp8_activation_scale = 0.025F,
                    .fp8_weight_scale = 0.005F,
                    .fp8_weight_scale_mode = Fp8WeightScaleMode::Fixed,
                    .rms_norm_epsilon = static_cast<float>(number(text, "rms_norm_eps")),
                    .attention_bias = true,
                    .rope_layout = RopeLayout::SplitHalf};
    output.torch_dtype = string_value(text, "torch_dtype");
    output.bos_token_id = integer(text, "bos_token_id");
    output.eos_token_id = integer(text, "eos_token_id");
    output.model.validate();
    return output;
}

}  // namespace microllm::model
