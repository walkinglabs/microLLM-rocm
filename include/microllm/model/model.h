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
using Bf16TrainingMirrors =
    std::vector<std::pair<autograd::Value*, Tensor*>>;

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

struct Bf16WeightPreparationReport {
    std::size_t converted_tensors = 0;
    std::uint64_t fp32_bytes_released = 0;
    std::uint64_t bf16_bytes_retained = 0;
};

struct Fp8WeightPreparationReport {
    std::size_t linears_covered = 0;
    std::size_t converted_tensors = 0;
    std::uint64_t fp32_bytes_released = 0;
    std::uint64_t fp8_bytes_retained = 0;
    std::uint64_t scale_bytes_retained = 0;
    std::uint64_t weight_bytes_scanned = 0;
    std::uint64_t device_weight_bytes_scanned = 0;
    std::size_t device_amax_tensors = 0;
    bool host_scale_summary_available = true;
    float minimum_weight_scale = 0.0F;
    float maximum_weight_scale = 0.0F;
};

using Bf16FfnPreparationReport = Bf16WeightPreparationReport;

enum class ParameterInitialization { Random, Uninitialized };

[[nodiscard]] WeightMapping qwen_style_weight_mapping(const ModelConfig& config);

class TransformerModel {
public:
    explicit TransformerModel(
        ModelConfig config, std::uint64_t seed = 1,
        ParameterInitialization initialization = ParameterInitialization::Random);
    ~TransformerModel();
    TransformerModel(TransformerModel&&) noexcept;
    TransformerModel& operator=(TransformerModel&&) noexcept;
    TransformerModel(const TransformerModel&) = delete;
    TransformerModel& operator=(const TransformerModel&) = delete;

    [[nodiscard]] const ModelConfig& config() const noexcept;
    [[nodiscard]] Device device();
    void to(Device device);
    [[nodiscard]] autograd::Value forward(const Tensor& token_ids);
    // Graph-free full-logits inference returning [B,T,V]. Unlike forward(),
    // this never creates autograd nodes and remains valid after BF16 preparation.
    [[nodiscard]] Tensor forward_inference(const Tensor& token_ids);
    // Serving prefill path: processes the full context but projects only the
    // final hidden position, returning [B,1,V] instead of [B,T,V].
    [[nodiscard]] Tensor forward_inference_last_logits(const Tensor& token_ids);
    // Full-sequence BxT prefill that initializes every layer's KV cache and
    // returns last-token logits [B,1,V]. The cache must be empty and fit B/T.
    [[nodiscard]] Tensor forward_prefill_cached(const Tensor& token_ids,
                                                inference::KVCache& cache);
    // Prefills one empty row of a shared batch cache without modifying other
    // rows. A temporary B1 cache is copied into the row as the correctness
    // reference for continuous slot admission.
    [[nodiscard]] Tensor forward_prefill_cached_row(
        const Tensor& token_ids, inference::KVCache& cache, std::int64_t row);
    // Batched form for equal-length prompts entering arbitrary empty rows of a
    // larger shared cache. Returns [A,1,V] logits in active_rows order.
    [[nodiscard]] Tensor forward_prefill_cached_rows(
        const Tensor& token_ids, inference::KVCache& cache,
        const std::vector<std::int64_t>& active_rows);
    [[nodiscard]] Tensor forward_cached(const Tensor& token_id,
                                        inference::KVCache& cache);
    // Correctness-first divergent-row decode. Each Bx1 row may have its own
    // cache position; the first implementation serializes shared-Storage B1
    // views and serves as the oracle for a future positions-aware HIP Kernel.
    [[nodiscard]] Tensor forward_cached_rows(const Tensor& token_ids,
                                             inference::KVCache& cache);
    // Advances only the listed shared-cache rows. token_ids has shape Ax1,
    // active_rows is strictly increasing, and inactive rows remain untouched.
    // This is the compacted correctness path used by continuous serving.
    [[nodiscard]] Tensor forward_cached_active_rows(
        const Tensor& token_ids, inference::KVCache& cache,
        const std::vector<std::int64_t>& active_rows);
    [[nodiscard]] autograd::Value loss(const Tensor& token_ids, const Tensor& targets);
    [[nodiscard]] NamedValues named_parameters();
    [[nodiscard]] std::vector<autograd::Value*> parameters();
    [[nodiscard]] std::uint64_t parameter_count();
    // One-way inference preparation: each FFN FP32 weight is replaced by BF16.
    // No persistent FP32 copy remains inside the model after this call.
    [[nodiscard]] Bf16FfnPreparationReport prepare_bf16_ffn_inference();
    [[nodiscard]] bool bf16_ffn_inference_prepared() const noexcept;
    [[nodiscard]] Bf16WeightPreparationReport prepare_bf16_attention_inference();
    [[nodiscard]] bool bf16_attention_inference_prepared() const noexcept;
    // One-way inference preparation for every Linear. FP32 Embedding/Norm and
    // a tied output head remain unchanged. Weight scale is either fixed or
    // computed independently from each Linear Tensor; activation scale remains fixed.
    [[nodiscard]] Fp8WeightPreparationReport prepare_fp8_inference_weights();
    [[nodiscard]] bool fp8_inference_weights_prepared() const noexcept;
    // Creates persistent BF16 forward mirrors for every Linear FP32 master.
    // Mirrors are derived runtime state and must be prepared after loading/restoring.
    [[nodiscard]] Bf16TrainingMirrors prepare_bf16_training_mirrors();
    [[nodiscard]] bool bf16_training_mirrors_prepared() const noexcept;
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
    [[nodiscard]] Tensor forward_inference_impl(const Tensor& token_ids,
                                                bool last_logits_only);
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace microllm::model
