#include <microllm/io/byte_tokenizer.h>

#include <stdexcept>

namespace microllm::io {

std::vector<std::int32_t> ByteTokenizer::encode(std::string_view text) const {
    std::vector<std::int32_t> tokens;
    tokens.reserve(text.size());
    for (const auto character : text) {
        tokens.push_back(static_cast<std::int32_t>(static_cast<unsigned char>(character)));
    }
    return tokens;
}

std::string ByteTokenizer::decode(const std::vector<std::int32_t>& tokens) const {
    std::string text;
    text.reserve(tokens.size());
    for (const auto token : tokens) {
        if (token < 0 || token >= vocabulary_size()) {
            throw std::out_of_range("byte token is outside [0, 255]");
        }
        text.push_back(static_cast<char>(static_cast<unsigned char>(token)));
    }
    return text;
}

}  // namespace microllm::io
