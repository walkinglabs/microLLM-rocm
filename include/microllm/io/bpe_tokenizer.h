#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace microllm::io {

class BpeTokenizer {
public:
    BpeTokenizer();

    [[nodiscard]] static BpeTokenizer train(std::string_view text,
                                            std::size_t target_vocabulary_size);
    [[nodiscard]] std::vector<std::int32_t> encode(std::string_view text) const;
    [[nodiscard]] std::string decode(const std::vector<std::int32_t>& tokens) const;
    [[nodiscard]] std::size_t vocabulary_size() const noexcept { return pieces_.size(); }
    [[nodiscard]] const std::vector<std::pair<std::int32_t, std::int32_t>>& merges() const noexcept {
        return merges_;
    }
    [[nodiscard]] std::string serialize() const;
    [[nodiscard]] static BpeTokenizer deserialize(std::string_view serialized);

private:
    std::vector<std::vector<std::uint8_t>> pieces_;
    std::vector<std::pair<std::int32_t, std::int32_t>> merges_;
};

}  // namespace microllm::io
