#pragma once

#include <string>
#include <utility>
#include <vector>

namespace microllm::io {

struct ChatMessage {
    std::string role;
    std::string content;
    struct ToolCall {
        std::string name;
        std::string arguments_json;
    };
    std::vector<ToolCall> tool_calls;
    ChatMessage(std::string role_value, std::string content_value,
                std::vector<ToolCall> calls = {})
        : role(std::move(role_value)), content(std::move(content_value)),
          tool_calls(std::move(calls)) {}
};

[[nodiscard]] std::string render_qwen2_chat(
    const std::vector<ChatMessage>& messages,
    bool add_generation_prompt = true,
    const std::string& default_system =
        "You are Qwen, created by Alibaba Cloud. You are a helpful assistant.",
    const std::vector<std::string>& tools_json = {});

[[nodiscard]] std::string render_deepseek_distill_chat(
    const std::vector<ChatMessage>& messages,
    bool add_generation_prompt = true);

}  // namespace microllm::io
