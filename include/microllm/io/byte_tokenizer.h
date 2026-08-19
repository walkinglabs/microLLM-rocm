#pragma once

#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

namespace microllm::io {

class ByteTokenizer {
public:
    [[nodiscard]] static constexpr std::int64_t vocabulary_size() noexcept { return 256; }
    [[nodiscard]] std::vector<std::int32_t> encode(std::string_view text) const;
    [[nodiscard]] std::string decode(const std::vector<std::int32_t>& tokens) const;
};

}  // namespace microllm::io
