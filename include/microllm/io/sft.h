#pragma once

#include <cstdint>
#include <string_view>
#include <vector>

#include <microllm/io/byte_tokenizer.h>
#include <microllm/io/token_dataset.h>

namespace microllm::io {

inline constexpr std::int32_t kIgnoredTarget = -100;

[[nodiscard]] TokenBatch make_sft_batch(const std::vector<std::int32_t>& prompt,
                                        const std::vector<std::int32_t>& response);
[[nodiscard]] TokenBatch make_sft_text_batch(std::string_view prompt,
                                             std::string_view response,
                                             const ByteTokenizer& tokenizer = {});

}  // namespace microllm::io
