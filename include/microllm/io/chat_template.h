#pragma once

#include <string>
#include <vector>

namespace microllm::io {

struct ChatMessage {
    std::string role;
    std::string content;
};

[[nodiscard]] std::string render_qwen2_chat(
    const std::vector<ChatMessage>& messages,
    bool add_generation_prompt = true,
    const std::string& default_system =
        "You are Qwen, created by Alibaba Cloud. You are a helpful assistant.");

[[nodiscard]] std::string render_deepseek_distill_chat(
    const std::vector<ChatMessage>& messages,
    bool add_generation_prompt = true);

}  // namespace microllm::io
