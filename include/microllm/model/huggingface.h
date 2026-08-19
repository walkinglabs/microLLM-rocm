#pragma once

#include <cstdint>
#include <filesystem>
#include <string>

#include <microllm/model/config.h>

namespace microllm::model {

struct HuggingFaceModelConfig {
    ModelConfig model;
    std::string model_type;
    std::string torch_dtype;
    std::int64_t bos_token_id = -1;
    std::int64_t eos_token_id = -1;
};

// First compatibility target: dense Qwen2/Qwen2.5 decoder configs.
[[nodiscard]] HuggingFaceModelConfig load_huggingface_config(
    const std::filesystem::path& path);

}  // namespace microllm::model
