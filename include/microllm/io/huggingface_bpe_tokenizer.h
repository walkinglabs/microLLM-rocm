#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace microllm::io {

class HuggingFaceBpeTokenizer {
public:
    [[nodiscard]] static HuggingFaceBpeTokenizer load(
        const std::filesystem::path& vocabulary_json,
        const std::filesystem::path& merges_txt);

    [[nodiscard]] std::vector<std::int32_t> encode(std::string_view text) const;
    [[nodiscard]] std::string decode(const std::vector<std::int32_t>& tokens) const;
    void add_special_token(std::string token, std::int32_t id);
    [[nodiscard]] std::size_t vocabulary_size() const noexcept { return pieces_.size(); }

private:
    std::array<std::string, 256> byte_encoder_{};
    std::unordered_map<std::string, std::uint8_t> byte_decoder_;
    std::unordered_map<std::string, std::int32_t> vocabulary_;
    std::vector<std::string> pieces_;
    std::unordered_map<std::string, std::size_t> merge_ranks_;
    std::unordered_map<std::string, std::int32_t> special_tokens_;
    std::unordered_map<std::int32_t, std::string> special_pieces_;
};

}  // namespace microllm::io
