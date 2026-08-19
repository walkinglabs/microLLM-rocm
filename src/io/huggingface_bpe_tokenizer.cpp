#include <microllm/io/huggingface_bpe_tokenizer.h>

#include <algorithm>
#include <charconv>
#include <cctype>
#include <fstream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <utility>

namespace microllm::io {
namespace {

std::string utf8(std::uint32_t codepoint) {
    std::string output;
    if (codepoint <= 0x7fU) {
        output.push_back(static_cast<char>(codepoint));
    } else if (codepoint <= 0x7ffU) {
        output.push_back(static_cast<char>(0xc0U | (codepoint >> 6U)));
        output.push_back(static_cast<char>(0x80U | (codepoint & 0x3fU)));
    } else if (codepoint <= 0xffffU) {
        output.push_back(static_cast<char>(0xe0U | (codepoint >> 12U)));
        output.push_back(static_cast<char>(0x80U | ((codepoint >> 6U) & 0x3fU)));
        output.push_back(static_cast<char>(0x80U | (codepoint & 0x3fU)));
    } else {
        output.push_back(static_cast<char>(0xf0U | (codepoint >> 18U)));
        output.push_back(static_cast<char>(0x80U | ((codepoint >> 12U) & 0x3fU)));
        output.push_back(static_cast<char>(0x80U | ((codepoint >> 6U) & 0x3fU)));
        output.push_back(static_cast<char>(0x80U | (codepoint & 0x3fU)));
    }
    return output;
}

std::string json_string(std::string_view text, std::size_t& position) {
    if (position >= text.size() || text[position++] != '"') {
        throw std::runtime_error("vocabulary JSON expected string");
    }
    std::string output;
    while (position < text.size()) {
        const auto character = text[position++];
        if (character == '"') return output;
        if (character != '\\') {
            output.push_back(character);
            continue;
        }
        if (position >= text.size()) throw std::runtime_error("unfinished vocabulary escape");
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
            case 'u': {
                if (position + 4 > text.size()) throw std::runtime_error("short unicode escape");
                std::uint32_t codepoint = 0;
                for (int digit = 0; digit < 4; ++digit) {
                    const auto value = text[position++];
                    codepoint <<= 4U;
                    if (value >= '0' && value <= '9') codepoint |= value - '0';
                    else if (value >= 'a' && value <= 'f') codepoint |= value - 'a' + 10U;
                    else if (value >= 'A' && value <= 'F') codepoint |= value - 'A' + 10U;
                    else throw std::runtime_error("invalid unicode escape");
                }
                output += utf8(codepoint);
                break;
            }
            default: throw std::runtime_error("unsupported vocabulary escape");
        }
    }
    throw std::runtime_error("unterminated vocabulary string");
}

std::string pair_key(std::string_view left, std::string_view right) {
    std::string key(left);
    key.push_back('\0');
    key.append(right);
    return key;
}

struct Codepoint {
    std::uint32_t value = 0;
    std::size_t begin = 0;
    std::size_t end = 0;
};

std::vector<Codepoint> codepoints(std::string_view text) {
    std::vector<Codepoint> output;
    for (std::size_t position = 0; position < text.size();) {
        const auto begin = position;
        const auto lead = static_cast<unsigned char>(text[position++]);
        std::uint32_t value = 0;
        std::size_t remaining = 0;
        if (lead < 0x80U) value = lead;
        else if ((lead & 0xe0U) == 0xc0U) { value = lead & 0x1fU; remaining = 1; }
        else if ((lead & 0xf0U) == 0xe0U) { value = lead & 0x0fU; remaining = 2; }
        else if ((lead & 0xf8U) == 0xf0U) { value = lead & 0x07U; remaining = 3; }
        else throw std::runtime_error("invalid UTF-8 lead byte");
        for (std::size_t index = 0; index < remaining; ++index) {
            if (position >= text.size()) throw std::runtime_error("truncated UTF-8");
            const auto continuation = static_cast<unsigned char>(text[position++]);
            if ((continuation & 0xc0U) != 0x80U) throw std::runtime_error("invalid UTF-8 continuation");
            value = (value << 6U) | (continuation & 0x3fU);
        }
        output.push_back({value, begin, position});
    }
    return output;
}

bool is_space(std::uint32_t value) {
    return value == ' ' || value == '\t' || value == '\n' || value == '\r' ||
           value == 0x0bU || value == 0x0cU || value == 0x85U ||
           value == 0xa0U || value == 0x3000U;
}

bool is_newline(std::uint32_t value) { return value == '\n' || value == '\r'; }

bool is_number(std::uint32_t value) {
    return (value >= '0' && value <= '9') ||
           (value >= 0xff10U && value <= 0xff19U);
}

bool is_letter(std::uint32_t value) {
    return (value >= 'A' && value <= 'Z') || (value >= 'a' && value <= 'z') ||
           (value >= 0xc0U && value <= 0x2afU) ||
           (value >= 0x370U && value <= 0x52fU) ||
           (value >= 0x400U && value <= 0x52fU) ||
           (value >= 0x3040U && value <= 0x30ffU) ||
           (value >= 0x3400U && value <= 0x9fffU) ||
           (value >= 0xac00U && value <= 0xd7afU);
}

std::vector<std::string_view> pretokenize(std::string_view text) {
    const auto points = codepoints(text);
    std::vector<std::string_view> output;
    auto emit = [&](std::size_t begin, std::size_t end) {
        output.push_back(text.substr(points[begin].begin,
                                     points[end - 1].end - points[begin].begin));
    };
    for (std::size_t position = 0; position < points.size();) {
        // Case-insensitive English contractions from the Qwen Split regex.
        if (points[position].value == '\'' && position + 1 < points.size()) {
            std::size_t end = position + 1;
            while (end < points.size() && is_letter(points[end].value) && end - position <= 3) ++end;
            const auto suffix = std::string(text.substr(points[position].begin,
                points[end - 1].end - points[position].begin));
            std::string lowered = suffix;
            std::transform(lowered.begin(), lowered.end(), lowered.begin(),
                           [](unsigned char value) { return static_cast<char>(std::tolower(value)); });
            if (lowered == "'s" || lowered == "'t" || lowered == "'re" ||
                lowered == "'ve" || lowered == "'m" || lowered == "'ll" || lowered == "'d") {
                emit(position, end);
                position = end;
                continue;
            }
        }
        // Preserve all but the final whitespace codepoint before a word/symbol so the
        // optional prefix in the following regex alternative consumes the last one.
        if (is_space(points[position].value) && !is_newline(points[position].value)) {
            auto end = position;
            while (end < points.size() && is_space(points[end].value) &&
                   !is_newline(points[end].value)) ++end;
            if (end < points.size() && end - position > 1) {
                emit(position, end - 1);
                position = end - 1;
                continue;
            }
        }
        // Optional one non-newline/non-letter/non-number prefix plus one or more letters.
        std::size_t letter_begin = position;
        if (!is_newline(points[position].value) && !is_letter(points[position].value) &&
            !is_number(points[position].value) && position + 1 < points.size() &&
            is_letter(points[position + 1].value)) ++letter_begin;
        if (is_letter(points[letter_begin].value)) {
            auto end = letter_begin + 1;
            while (end < points.size() && is_letter(points[end].value)) ++end;
            emit(position, end);
            position = end;
            continue;
        }
        if (is_number(points[position].value)) {
            emit(position, position + 1);
            ++position;
            continue;
        }
        // Whitespace followed by one or more newlines.
        if (is_space(points[position].value)) {
            auto end = position;
            while (end < points.size() && is_space(points[end].value)) ++end;
            const auto contains_newline = std::any_of(
                points.begin() + static_cast<std::ptrdiff_t>(position),
                points.begin() + static_cast<std::ptrdiff_t>(end),
                [](const auto& point) { return is_newline(point.value); });
            if (contains_newline) {
                emit(position, end);
                position = end;
                continue;
            }
        }
        // Optional ASCII space plus punctuation/symbol run and trailing newlines.
        auto end = position;
        if (points[end].value == ' ' && end + 1 < points.size() &&
            !is_space(points[end + 1].value) && !is_letter(points[end + 1].value) &&
            !is_number(points[end + 1].value)) ++end;
        if (end < points.size() && !is_space(points[end].value) &&
            !is_letter(points[end].value) && !is_number(points[end].value)) {
            ++end;
            while (end < points.size() && !is_space(points[end].value) &&
                   !is_letter(points[end].value) && !is_number(points[end].value)) ++end;
            while (end < points.size() && is_newline(points[end].value)) ++end;
            emit(position, end);
            position = end;
            continue;
        }
        // Remaining whitespace run.
        end = position + 1;
        while (end < points.size() && is_space(points[end].value)) ++end;
        emit(position, end);
        position = end;
    }
    return output;
}

}  // namespace

HuggingFaceBpeTokenizer HuggingFaceBpeTokenizer::load(
    const std::filesystem::path& vocabulary_json,
    const std::filesystem::path& merges_txt) {
    HuggingFaceBpeTokenizer tokenizer;
    std::vector<int> bytes;
    for (int value = 33; value <= 126; ++value) bytes.push_back(value);
    for (int value = 161; value <= 172; ++value) bytes.push_back(value);
    for (int value = 174; value <= 255; ++value) bytes.push_back(value);
    auto codepoints = bytes;
    std::uint32_t extra = 0;
    for (int value = 0; value < 256; ++value) {
        if (std::find(bytes.begin(), bytes.end(), value) == bytes.end()) {
            bytes.push_back(value);
            codepoints.push_back(static_cast<int>(256U + extra++));
        }
    }
    for (std::size_t index = 0; index < bytes.size(); ++index) {
        const auto encoded = utf8(static_cast<std::uint32_t>(codepoints[index]));
        tokenizer.byte_encoder_[static_cast<std::size_t>(bytes[index])] = encoded;
        tokenizer.byte_decoder_.emplace(encoded, static_cast<std::uint8_t>(bytes[index]));
    }

    std::ifstream vocab_input(vocabulary_json, std::ios::binary);
    if (!vocab_input) throw std::runtime_error("cannot open Hugging Face vocabulary");
    std::ostringstream vocab_buffer;
    vocab_buffer << vocab_input.rdbuf();
    const auto text = vocab_buffer.str();
    std::size_t position = 0;
    if (text.empty() || text[position++] != '{') throw std::runtime_error("vocabulary root must be object");
    while (position < text.size() && text[position] != '}') {
        const auto piece = json_string(text, position);
        if (position >= text.size() || text[position++] != ':') throw std::runtime_error("vocabulary missing ':'");
        const auto begin = position;
        while (position < text.size() && text[position] >= '0' && text[position] <= '9') ++position;
        std::int32_t id = 0;
        const auto parsed = std::from_chars(text.data() + begin, text.data() + position, id);
        if (parsed.ec != std::errc{} || parsed.ptr != text.data() + position || id < 0) {
            throw std::runtime_error("invalid vocabulary token id");
        }
        if (!tokenizer.vocabulary_.emplace(piece, id).second) {
            throw std::runtime_error("duplicate vocabulary piece");
        }
        if (static_cast<std::size_t>(id) >= tokenizer.pieces_.size()) {
            tokenizer.pieces_.resize(static_cast<std::size_t>(id) + 1);
        }
        if (!tokenizer.pieces_[static_cast<std::size_t>(id)].empty()) {
            throw std::runtime_error("duplicate vocabulary id");
        }
        tokenizer.pieces_[static_cast<std::size_t>(id)] = piece;
        if (position < text.size() && text[position] == ',') ++position;
    }
    if (position >= text.size() || text[position] != '}') throw std::runtime_error("unterminated vocabulary");

    std::ifstream merge_input(merges_txt);
    if (!merge_input) throw std::runtime_error("cannot open Hugging Face merges");
    std::string line;
    std::size_t rank = 0;
    while (std::getline(merge_input, line)) {
        if (line.empty() || line[0] == '#') continue;
        const auto separator = line.find(' ');
        if (separator == std::string::npos || separator == 0 || separator + 1 >= line.size()) {
            throw std::runtime_error("malformed Hugging Face merge");
        }
        tokenizer.merge_ranks_.emplace(
            pair_key(std::string_view(line).substr(0, separator),
                     std::string_view(line).substr(separator + 1)), rank++);
    }
    return tokenizer;
}

std::vector<std::int32_t> HuggingFaceBpeTokenizer::encode(std::string_view text) const {
    std::vector<std::int32_t> output;
    const auto encode_ordinary = [&](std::string_view ordinary) {
    for (const auto chunk : pretokenize(ordinary)) {
        std::vector<std::string> symbols;
        symbols.reserve(chunk.size());
        for (const auto character : chunk) {
            symbols.push_back(byte_encoder_[static_cast<unsigned char>(character)]);
        }
        while (symbols.size() >= 2) {
            auto best_rank = std::numeric_limits<std::size_t>::max();
            std::string best_key;
            for (std::size_t index = 0; index + 1 < symbols.size(); ++index) {
                const auto key = pair_key(symbols[index], symbols[index + 1]);
                const auto found = merge_ranks_.find(key);
                if (found != merge_ranks_.end() && found->second < best_rank) {
                    best_rank = found->second;
                    best_key = key;
                }
            }
            if (best_rank == std::numeric_limits<std::size_t>::max()) break;
            std::vector<std::string> merged;
            for (std::size_t index = 0; index < symbols.size();) {
                if (index + 1 < symbols.size() &&
                    pair_key(symbols[index], symbols[index + 1]) == best_key) {
                    merged.push_back(symbols[index] + symbols[index + 1]);
                    index += 2;
                } else {
                    merged.push_back(symbols[index++]);
                }
            }
            symbols = std::move(merged);
        }
        for (const auto& symbol : symbols) {
            const auto found = vocabulary_.find(symbol);
            if (found == vocabulary_.end()) {
                throw std::runtime_error("BPE piece is absent from vocabulary");
            }
            output.push_back(found->second);
        }
    }
    };
    while (!text.empty()) {
        auto special_position = std::string_view::npos;
        const std::string* special_piece = nullptr;
        std::int32_t special_id = -1;
        for (const auto& [piece, id] : special_tokens_) {
            const auto found = text.find(piece);
            if (found < special_position) {
                special_position = found;
                special_piece = &piece;
                special_id = id;
            }
        }
        if (special_piece == nullptr) {
            encode_ordinary(text);
            break;
        }
        encode_ordinary(text.substr(0, special_position));
        output.push_back(special_id);
        text.remove_prefix(special_position + special_piece->size());
    }
    return output;
}

void HuggingFaceBpeTokenizer::add_special_token(std::string token, std::int32_t id) {
    if (token.empty() || id < 0) throw std::invalid_argument("special token is invalid");
    if (vocabulary_.contains(token) || special_tokens_.contains(token) ||
        special_pieces_.contains(id)) {
        throw std::invalid_argument("duplicate special token or ID");
    }
    special_pieces_.emplace(id, token);
    special_tokens_.emplace(std::move(token), id);
}

std::string HuggingFaceBpeTokenizer::decode(const std::vector<std::int32_t>& tokens) const {
    // Special strings are already ordinary UTF-8, whereas byte-level pieces must be
    // inverted through byte_decoder_. Decode each token separately to avoid ambiguity.
    std::string direct_output;
    for (const auto token : tokens) {
        const auto special = special_pieces_.find(token);
        if (special != special_pieces_.end()) {
            direct_output += special->second;
            continue;
        }
        if (token < 0 || static_cast<std::size_t>(token) >= pieces_.size()) {
            throw std::out_of_range("Hugging Face BPE token is outside vocabulary");
        }
        const auto& token_piece = pieces_[static_cast<std::size_t>(token)];
        for (std::size_t position = 0; position < token_piece.size();) {
            const auto lead = static_cast<unsigned char>(token_piece[position]);
            const auto width = lead < 0x80U ? 1U : lead < 0xe0U ? 2U : lead < 0xf0U ? 3U : 4U;
            if (position + width > token_piece.size()) throw std::runtime_error("invalid UTF-8 BPE piece");
            const auto piece = token_piece.substr(position, width);
            const auto found = byte_decoder_.find(piece);
            if (found == byte_decoder_.end()) throw std::runtime_error("BPE codepoint has no byte mapping");
            direct_output.push_back(static_cast<char>(found->second));
            position += width;
        }
    }
    return direct_output;
}

}  // namespace microllm::io
