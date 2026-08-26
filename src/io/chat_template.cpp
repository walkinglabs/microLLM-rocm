#include <microllm/io/chat_template.h>

#include <stdexcept>
#include <string_view>

namespace microllm::io {
namespace {

std::string json_string(std::string_view value) {
    std::string output{"\""};
    for (const auto character : value) {
        if (character == '"' || character == '\\') output.push_back('\\');
        output.push_back(character);
    }
    output.push_back('"');
    return output;
}

bool json_object_boundary(std::string_view value) {
    const auto first = value.find_first_not_of(" \t\r\n");
    const auto last = value.find_last_not_of(" \t\r\n");
    return first != std::string_view::npos && value[first] == '{' &&
           value[last] == '}';
}

}  // namespace

std::string render_qwen2_chat(const std::vector<ChatMessage>& messages,
                              bool add_generation_prompt,
                              const std::string& default_system,
                              const std::vector<std::string>& tools_json) {
    if (messages.empty()) throw std::invalid_argument("chat messages cannot be empty");
    std::string output;
    const auto append_system = [&](const std::string& content) {
        output += "<|im_start|>system\n" + content;
        if (!tools_json.empty()) {
            output += "\n\n# Tools\n<tools>\n";
            for (const auto& tool : tools_json) {
                if (!json_object_boundary(tool)) {
                    throw std::invalid_argument("Qwen tool schema must be a JSON object");
                }
                output += tool + "\n";
            }
            output += "</tools>\nUse <tool_call> with name and arguments when needed.";
        }
        output += "<|im_end|>\n";
    };
    if (messages.front().role != "system" && !default_system.empty()) {
        append_system(default_system);
    }
    bool awaiting_tool_response = false;
    for (const auto& message : messages) {
        if (message.role != "system" && message.role != "user" &&
            message.role != "assistant" && message.role != "tool") {
            throw std::invalid_argument(
                "Qwen chat supports system/user/assistant/tool roles");
        }
        if (message.role != "assistant" && !message.tool_calls.empty()) {
            throw std::invalid_argument("only assistant messages may contain tool calls");
        }
        if (message.role == "system") {
            append_system(message.content);
            awaiting_tool_response = false;
            continue;
        }
        if (message.role == "tool") {
            if (!awaiting_tool_response || message.content.empty()) {
                throw std::invalid_argument(
                    "Qwen tool response must follow an assistant tool call");
            }
            output += "<|im_start|>user\n<tool_response>\n" +
                      message.content + "\n</tool_response><|im_end|>\n";
            continue;
        }
        output += "<|im_start|>" + message.role + "\n" + message.content;
        if (!message.content.empty() && !message.tool_calls.empty()) output += '\n';
        for (const auto& call : message.tool_calls) {
            if (call.name.empty() || !json_object_boundary(call.arguments_json)) {
                throw std::invalid_argument(
                    "Qwen tool call needs a name and JSON-object arguments");
            }
            output += "<tool_call>\n{\"name\":" + json_string(call.name) +
                      ",\"arguments\":" + call.arguments_json +
                      "}\n</tool_call>\n";
        }
        output += "<|im_end|>\n";
        awaiting_tool_response = !message.tool_calls.empty();
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
