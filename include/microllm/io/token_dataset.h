#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include <microllm/core/tensor.h>

namespace microllm::io {

struct TokenBatch {
    Tensor inputs;
    Tensor targets;
};

class TokenDataset {
public:
    TokenDataset(std::vector<std::int32_t> tokens, std::int64_t context_length);

    [[nodiscard]] TokenBatch next_batch(std::int64_t batch_size);
    [[nodiscard]] std::uint64_t cursor() const noexcept { return cursor_; }
    void set_cursor(std::uint64_t cursor);
    [[nodiscard]] std::int64_t context_length() const noexcept { return context_length_; }
    [[nodiscard]] std::uint64_t valid_start_count() const noexcept;
    [[nodiscard]] const std::vector<std::int32_t>& tokens() const noexcept { return tokens_; }
    [[nodiscard]] std::string summary() const;

private:
    std::vector<std::int32_t> tokens_;
    std::int64_t context_length_;
    std::uint64_t cursor_ = 0;
};

}  // namespace microllm::io
