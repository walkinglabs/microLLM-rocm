#pragma once

#include <cstdint>
#include <filesystem>
#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <microllm/autograd/autograd.h>
#include <microllm/model/config.h>
#include <microllm/inference/kv_cache.h>
#include <microllm/io/safetensors.h>

namespace microllm::model {

using NamedValues = std::vector<std::pair<std::string, autograd::Value*>>;

enum class WeightTransform { Identity, Transpose2D };

struct WeightSource {
    std::string name;
    WeightTransform transform = WeightTransform::Identity;
};

using WeightMapping = std::map<std::string, WeightSource>;

struct LoadWeightsOptions {
    bool strict = true;
    WeightMapping mapping;
};

struct LoadWeightsReport {
    std::vector<std::string> loaded;
    std::vector<std::string> missing;
    std::vector<std::string> unexpected;
    std::vector<std::string> incompatible;

    [[nodiscard]] bool complete() const noexcept {
        return missing.empty() && unexpected.empty() && incompatible.empty();
    }
};

[[nodiscard]] WeightMapping qwen_style_weight_mapping(const ModelConfig& config);

class TransformerModel {
public:
    explicit TransformerModel(ModelConfig config, std::uint64_t seed = 1);
    ~TransformerModel();
    TransformerModel(TransformerModel&&) noexcept;
    TransformerModel& operator=(TransformerModel&&) noexcept;
    TransformerModel(const TransformerModel&) = delete;
    TransformerModel& operator=(const TransformerModel&) = delete;

    [[nodiscard]] const ModelConfig& config() const noexcept;
    [[nodiscard]] Device device();
    void to(Device device);
    [[nodiscard]] autograd::Value forward(const Tensor& token_ids);
    [[nodiscard]] Tensor forward_cached(const Tensor& token_id,
                                        inference::KVCache& cache);
    [[nodiscard]] autograd::Value loss(const Tensor& token_ids, const Tensor& targets);
    [[nodiscard]] NamedValues named_parameters();
    [[nodiscard]] std::vector<autograd::Value*> parameters();
    [[nodiscard]] std::uint64_t parameter_count();
    [[nodiscard]] io::StateDict state_dict(Device target = Device::cpu());
    [[nodiscard]] LoadWeightsReport load_state_dict(
        const io::StateDict& state, const LoadWeightsOptions& options = {});
    [[nodiscard]] LoadWeightsReport load_safetensors(
        const std::filesystem::path& path,
        const LoadWeightsOptions& options = {});
    [[nodiscard]] LoadWeightsReport load_safetensors_files(
        const std::vector<std::filesystem::path>& paths,
        const LoadWeightsOptions& options = {});
    [[nodiscard]] LoadWeightsReport load_safetensors_index(
        const std::filesystem::path& index_path,
        const LoadWeightsOptions& options = {});
    void save_safetensors(
        const std::filesystem::path& path,
        const io::SafetensorsSaveOptions& options = {});

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace microllm::model
