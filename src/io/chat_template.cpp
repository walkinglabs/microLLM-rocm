#include <microllm/io/chat_template.h>

#include <stdexcept>

namespace microllm::io {

std::string render_qwen2_chat(const std::vector<ChatMessage>& messages,
                              bool add_generation_prompt,
                              const std::string& default_system) {
    if (messages.empty()) throw std::invalid_argument("chat messages cannot be empty");
    std::string output;
    if (messages.front().role != "system" && !default_system.empty()) {
        output += "<|im_start|>system\n" + default_system + "<|im_end|>\n";
    }
    for (const auto& message : messages) {
        if (message.role != "system" && message.role != "user" &&
            message.role != "assistant") {
            throw std::invalid_argument("basic Qwen chat supports system/user/assistant only");
        }
        output += "<|im_start|>" + message.role + "\n" + message.content +
                  "<|im_end|>\n";
    }
    if (add_generation_prompt) output += "<|im_start|>assistant\n";
    return output;
}

std::string render_deepseek_distill_chat(
    const std::vector<ChatMessage>& messages, bool add_generation_prompt) {
    if (messages.empty()) throw std::invalid_argument("chat messages cannot be empty");
    std::string system;
    for (const auto& message : messages) {
        if (message.role == "system") system = message.content;
    }
    std::string output = "<｜begin▁of▁sentence｜>" + system;
    for (const auto& message : messages) {
        if (message.role == "system") continue;
        if (message.role == "user") {
            output += "<｜User｜>" + message.content;
        } else if (message.role == "assistant") {
            auto content = message.content;
            const auto think_end = content.rfind("</think>");
            if (think_end != std::string::npos) content.erase(0, think_end + 8);
            output += "<｜Assistant｜>" + content + "<｜end▁of▁sentence｜>";
        } else {
            throw std::invalid_argument(
                "basic DeepSeek Distill chat supports system/user/assistant only");
        }
    }
    if (add_generation_prompt) output += "<｜Assistant｜><think>\n";
    return output;
}

}  // namespace microllm::io
