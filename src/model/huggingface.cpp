#include <microllm/model/huggingface.h>

#include <charconv>
#include <cmath>
#include <fstream>
#include <limits>
#include <optional>
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

std::optional<std::string_view> optional_member(
    std::string_view text, std::string_view wanted) {
    try {
        return member(text, wanted);
    } catch (const std::runtime_error& error) {
        if (error.what() ==
            "Hugging Face config is missing " + std::string(wanted)) {
            return std::nullopt;
        }
        throw;
    }
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

bool optional_boolean(std::string_view text, std::string_view key,
                      bool fallback) {
    const auto value = optional_member(text, key);
    if (!value) return fallback;
    if (*value == "true") return true;
    if (*value == "false") return false;
    throw std::runtime_error(std::string(key) + " must be boolean");
}

std::int64_t optional_integer(std::string_view text, std::string_view key,
                              std::int64_t fallback) {
    if (!optional_member(text, key)) return fallback;
    return integer(text, key);
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
    if (output.model_type != "qwen2" && output.model_type != "qwen3" &&
        output.model_type != "qwen3_moe") {
        throw std::invalid_argument(
            "Hugging Face target requires model_type=qwen2, qwen3, or qwen3_moe");
    }
    const auto is_qwen3_family =
        output.model_type == "qwen3" || output.model_type == "qwen3_moe";
    if (string_value(text, "hidden_act") != "silu") {
        throw std::invalid_argument("Qwen2 hidden_act must be silu");
    }
    if (boolean(text, "use_sliding_window") ||
        optional_boolean(text, "use_mrope", false)) {
        throw std::invalid_argument("sliding-window and MRoPE Qwen variants are not supported");
    }
    if (const auto rope_scaling = optional_member(text, "rope_scaling");
        rope_scaling && *rope_scaling != "null") {
        throw std::invalid_argument("non-null RoPE scaling is not supported");
    }
    const auto hidden = integer(text, "hidden_size");
    const auto heads = integer(text, "num_attention_heads");
    output.model = {.vocabulary_size = integer(text, "vocab_size"),
                    .dimension = hidden,
                    .layers = integer(text, "num_hidden_layers"),
                    .heads = heads,
                    .kv_heads = integer(text, "num_key_value_heads"),
                    .attention_head_dimension = is_qwen3_family
                        ? optional_integer(text, "head_dim", hidden / heads) : 0,
                    .ffn_dimension = integer(text, "intermediate_size"),
                    .max_sequence_length = integer(text, "max_position_embeddings"),
                    .rope_base = static_cast<float>(number(text, "rope_theta")),
                    .tie_embeddings = boolean(text, "tie_word_embeddings"),
                    .linear_precision = LinearPrecision::Float32,
                    .fp8_activation_scale = 0.025F,
                    .fp8_activation_minimum_scale = 1.0e-4F,
                    .fp8_weight_scale = 0.005F,
                    .fp8_weight_scale_mode = Fp8WeightScaleMode::Fixed,
                    .fp8_activation_scale_mode = Fp8ActivationScaleMode::Fixed,
                    .rms_norm_epsilon = static_cast<float>(number(text, "rms_norm_eps")),
                    .attention_bias = optional_boolean(
                        text, "attention_bias", output.model_type == "qwen2"),
                    .qk_norm = is_qwen3_family,
                    .rope_layout = RopeLayout::SplitHalf,
                    .moe_num_experts = output.model_type == "qwen3_moe"
                        ? integer(text, "num_experts") : 0,
                    .moe_num_experts_per_tok = output.model_type == "qwen3_moe"
                        ? integer(text, "num_experts_per_tok") : 0,
                    .moe_intermediate_size = output.model_type == "qwen3_moe"
                        ? integer(text, "moe_intermediate_size") : 0,
                    .moe_norm_topk_prob = output.model_type == "qwen3_moe" &&
                        boolean(text, "norm_topk_prob")};
    if (output.model_type == "qwen3_moe") {
        // decoder_sparse_step==1 and an empty mlp_only_layers mean every layer is
        // MoE; anything else is a per-layer dense/MoE mix this parser does not
        // represent.
        if (optional_integer(text, "decoder_sparse_step", 1) != 1) {
            throw std::invalid_argument(
                "qwen3_moe decoder_sparse_step must be 1 (every layer is MoE)");
        }
        if (const auto mlp_only_layers = optional_member(text, "mlp_only_layers");
            mlp_only_layers && *mlp_only_layers != "[]") {
            throw std::invalid_argument(
                "qwen3_moe mlp_only_layers must be empty (per-layer dense/MoE mixing "
                "is not supported)");
        }
        // router_aux_loss_coef is intentionally accepted without effect: M4
        // originally rejected its presence outright (never silently ignore a
        // documented field), but M7 found every real Qwen3-MoE config --
        // official and third-party alike -- serializes this field with its
        // dataclass default, so rejecting it made every real checkpoint
        // unloadable. It only configures a training-time load-balancing loss
        // this repo does not implement anywhere, so there is no behavior to
        // misrepresent by not reading it.
    }
    // Informational only (never read to select compute dtype -- that is
    // ModelConfig::linear_precision, controlled separately). Newer Hugging
    // Face configs serialize this as "dtype" instead of "torch_dtype" (seen
    // on a real Qwen3-MoE checkpoint while preparing the M7 fixture); accept
    // either key rather than rejecting an otherwise-loadable config over a
    // metadata field's name.
    output.torch_dtype = optional_member(text, "torch_dtype")
                              ? string_value(text, "torch_dtype")
                              : string_value(text, "dtype");
    output.bos_token_id = integer(text, "bos_token_id");
    output.eos_token_id = integer(text, "eos_token_id");
    output.model.validate();
    return output;
}

}  // namespace microllm::model
