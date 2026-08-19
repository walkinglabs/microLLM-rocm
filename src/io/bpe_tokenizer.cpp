#include <microllm/io/bpe_tokenizer.h>

#include <algorithm>
#include <map>
#include <sstream>
#include <stdexcept>

namespace microllm::io {
namespace {

std::vector<std::int32_t> bytes(std::string_view text) {
    std::vector<std::int32_t> tokens;
    tokens.reserve(text.size());
    for (const auto character : text) {
        tokens.push_back(static_cast<std::int32_t>(static_cast<unsigned char>(character)));
    }
    return tokens;
}

std::vector<std::int32_t> merge_pair(const std::vector<std::int32_t>& input,
                                     std::pair<std::int32_t, std::int32_t> pair,
                                     std::int32_t replacement) {
    std::vector<std::int32_t> output;
    output.reserve(input.size());
    std::size_t index = 0;
    while (index < input.size()) {
        if (index + 1 < input.size() && input[index] == pair.first &&
            input[index + 1] == pair.second) {
            output.push_back(replacement);
            index += 2;
        } else {
            output.push_back(input[index]);
            ++index;
        }
    }
    return output;
}

}  // namespace

BpeTokenizer::BpeTokenizer() {
    pieces_.reserve(256);
    for (std::int32_t value = 0; value < 256; ++value) {
        pieces_.push_back({static_cast<std::uint8_t>(value)});
    }
}

BpeTokenizer BpeTokenizer::train(std::string_view text, std::size_t target_vocabulary_size) {
    if (text.empty()) throw std::invalid_argument("BPE training text cannot be empty");
    if (target_vocabulary_size < 256) {
        throw std::invalid_argument("byte-level BPE vocabulary cannot be below 256");
    }
    BpeTokenizer tokenizer;
    auto tokens = bytes(text);
    while (tokenizer.pieces_.size() < target_vocabulary_size && tokens.size() >= 2) {
        std::map<std::pair<std::int32_t, std::int32_t>, std::size_t> counts;
        for (std::size_t index = 0; index + 1 < tokens.size(); ++index) {
            ++counts[{tokens[index], tokens[index + 1]}];
        }
        const auto best = std::max_element(
            counts.begin(), counts.end(), [](const auto& left, const auto& right) {
                if (left.second != right.second) return left.second < right.second;
                return left.first > right.first;
            });
        if (best == counts.end() || best->second < 2) break;
        const auto pair = best->first;
        const auto new_token = static_cast<std::int32_t>(tokenizer.pieces_.size());
        auto piece = tokenizer.pieces_[static_cast<std::size_t>(pair.first)];
        const auto& right_piece = tokenizer.pieces_[static_cast<std::size_t>(pair.second)];
        piece.insert(piece.end(), right_piece.begin(), right_piece.end());
        tokenizer.pieces_.push_back(std::move(piece));
        tokenizer.merges_.push_back(pair);
        tokens = merge_pair(tokens, pair, new_token);
    }
    return tokenizer;
}

std::vector<std::int32_t> BpeTokenizer::encode(std::string_view text) const {
    auto tokens = bytes(text);
    for (std::size_t index = 0; index < merges_.size(); ++index) {
        tokens = merge_pair(tokens, merges_[index], static_cast<std::int32_t>(256 + index));
    }
    return tokens;
}

std::string BpeTokenizer::decode(const std::vector<std::int32_t>& tokens) const {
    std::string text;
    for (const auto token : tokens) {
        if (token < 0 || static_cast<std::size_t>(token) >= pieces_.size()) {
            throw std::out_of_range("BPE token is outside the vocabulary");
        }
        const auto& piece = pieces_[static_cast<std::size_t>(token)];
        for (const auto byte : piece) text.push_back(static_cast<char>(byte));
    }
    return text;
}

std::string BpeTokenizer::serialize() const {
    std::ostringstream output;
    output << "MICROLLM_BPE_V1\n";
    for (const auto& [left, right] : merges_) output << left << ' ' << right << '\n';
    return output.str();
}

BpeTokenizer BpeTokenizer::deserialize(std::string_view serialized) {
    std::istringstream input{std::string(serialized)};
    std::string magic;
    std::getline(input, magic);
    if (magic != "MICROLLM_BPE_V1") throw std::invalid_argument("BPE format magic mismatch");
    BpeTokenizer tokenizer;
    std::int32_t left = 0;
    std::int32_t right = 0;
    while (input >> left >> right) {
        if (left < 0 || right < 0 || static_cast<std::size_t>(left) >= tokenizer.pieces_.size() ||
            static_cast<std::size_t>(right) >= tokenizer.pieces_.size()) {
            throw std::invalid_argument("BPE merge references an unavailable token");
        }
        auto piece = tokenizer.pieces_[static_cast<std::size_t>(left)];
        const auto& right_piece = tokenizer.pieces_[static_cast<std::size_t>(right)];
        piece.insert(piece.end(), right_piece.begin(), right_piece.end());
        tokenizer.merges_.emplace_back(left, right);
        tokenizer.pieces_.push_back(std::move(piece));
    }
    if (!input.eof()) throw std::invalid_argument("BPE merge file is malformed");
    return tokenizer;
}

}  // namespace microllm::io
