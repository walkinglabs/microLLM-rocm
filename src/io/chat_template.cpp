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

}  // namespace microllm::io
