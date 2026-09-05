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

// Dense Qwen2/Qwen2.5 and Qwen3 decoder config compatibility, plus Qwen3 MoE
// (model_type=qwen3_moe) config parsing. Only uniform per-layer MoE
// (decoder_sparse_step=1, empty mlp_only_layers) is supported; MoE weight
// loading and model-level forward are not implemented by this milestone.

[[nodiscard]] HuggingFaceModelConfig load_huggingface_config(
    const std::filesystem::path& path);

}  // namespace microllm::model
