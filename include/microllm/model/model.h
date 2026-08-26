#pragma once

#include <cstddef>
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
// External source name -> internal target parameter. A strict loader consumes
// the extra source only after proving its serialized values exactly equal the
// primary source for that tied target.
using WeightAliases = std::map<std::string, std::string>;

struct LoadWeightsOptions {
    bool strict = true;
    WeightMapping mapping;
    WeightAliases aliases;
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

struct Int8WeightPreparationReport {
    std::size_t linears_covered = 0;
    std::uint64_t fp32_bytes_released = 0;
    std::uint64_t int8_bytes_retained = 0;
    std::uint64_t scale_bytes_retained = 0;
    std::uint64_t device_weight_bytes_scanned = 0;
    std::size_t device_amax_tensors = 0;
};
enum class Int8WeightScaleMode { TensorAmax, OutputColumnAmax };
enum class Int8WeightScope {
    AllLinear, FfnOnly, AttentionOnly, AttentionQkvOnly,
    AttentionOutputOnly, OutputHeadOnly
};

using Bf16FfnPreparationReport = Bf16WeightPreparationReport;

struct Bf16FfnArenaStats {
    std::size_t entries = 0;
    std::size_t hits = 0;
    std::size_t misses = 0;
    std::size_t eligible_calls = 0;
    std::size_t bypassed_calls = 0;
    std::int64_t minimum_rows = 1;
    std::uint64_t capacity_bytes = 0;
};

using Bf16QkvArenaStats = Bf16FfnArenaStats;
struct Bf16GroupedQkvPrewarmReport {
    std::int64_t rows = 0;
    std::size_t blocks = 0;
    double total_ms = 0.0;
    double kernel_setup_ms = 0.0;
    double argument_setup_ms = 0.0;
    bool already_warm = false;
};
struct AttentionCoreArenaStats {
    std::size_t entries = 0;
    std::size_t hits = 0;
    std::size_t misses = 0;
    std::size_t eligible_calls = 0;
    std::size_t bypassed_calls = 0;
    std::int64_t minimum_sequence = 512;
    std::uint64_t capacity_bytes = 0;
};

enum class ParameterInitialization { Random, Uninitialized };

[[nodiscard]] WeightMapping qwen_style_weight_mapping(const ModelConfig& config);
[[nodiscard]] WeightAliases qwen3_tied_weight_aliases(const ModelConfig& config);

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
    // Explicit research policy: the listed Transformer blocks keep their FFN
    // weights and activations in FP32 while every other FFN becomes BF16.
    // Indices must be unique and in range. The operation remains one-way.
    [[nodiscard]] Bf16FfnPreparationReport prepare_bf16_ffn_inference(
        const std::vector<std::int64_t>& fp32_layers);
    [[nodiscard]] bool bf16_ffn_inference_prepared() const noexcept;
    // Opt-in graph-free inference workspace. One stable backing allocation is
    // cached per (device, flattened row count) and reused across all blocks.
    // A model instance requires external synchronization while this is enabled.
    void set_bf16_ffn_arena_enabled(bool enabled,
                                    std::int64_t minimum_rows = 1);
    [[nodiscard]] bool bf16_ffn_arena_enabled() const noexcept;
    [[nodiscard]] Bf16FfnArenaStats bf16_ffn_arena_stats() const noexcept;
    void set_bf16_ffn_norm_fusion_enabled(bool enabled);
    [[nodiscard]] bool bf16_ffn_norm_fusion_enabled() const noexcept;
    [[nodiscard]] Bf16WeightPreparationReport prepare_bf16_attention_inference();
    [[nodiscard]] bool bf16_attention_inference_prepared() const noexcept;
    void set_bf16_qkv_arena_enabled(bool enabled,
                                    std::int64_t minimum_rows = 512);
    [[nodiscard]] bool bf16_qkv_arena_enabled() const noexcept;
    [[nodiscard]] Bf16QkvArenaStats bf16_qkv_arena_stats() const noexcept;
    void set_bf16_attention_norm_fusion_enabled(bool enabled);
    [[nodiscard]] bool bf16_attention_norm_fusion_enabled() const noexcept;
    [[nodiscard]] Bf16GroupedQkvPrewarmReport
    prewarm_bf16_grouped_qkv(std::int64_t rows);
    void set_attention_core_arena_enabled(
        bool enabled, std::int64_t minimum_sequence = 512);
    [[nodiscard]] bool attention_core_arena_enabled() const noexcept;
    [[nodiscard]] AttentionCoreArenaStats attention_core_arena_stats() const noexcept;
    // Explicit research route for uniform cached decode. Zero disables it.
    // Divergent/positions-aware decode retains its dedicated reference path.
    void set_cached_attention_split_sequence(
        std::int64_t splits, std::int64_t minimum_sequence = 512);
    [[nodiscard]] std::int64_t cached_attention_split_sequence_splits()
        const noexcept;
    [[nodiscard]] std::int64_t cached_attention_split_minimum_sequence()
        const noexcept;
    void set_cached_attention_materialized_scores(
        bool enabled, std::int64_t minimum_sequence = 512);
    [[nodiscard]] bool cached_attention_materialized_scores_enabled()
        const noexcept;
    [[nodiscard]] std::int64_t cached_attention_materialized_minimum_sequence()
        const noexcept;
    // Explicit research route for uniform cached decode. Zero disables it.
    // It is mutually exclusive with full split-sequence and materialized routes.
    void set_cached_attention_split_pv(
        std::int64_t splits, std::int64_t minimum_sequence = 512);
    [[nodiscard]] std::int64_t cached_attention_split_pv_splits() const noexcept;
    [[nodiscard]] std::int64_t cached_attention_split_pv_minimum_sequence()
        const noexcept;
    // One-way inference preparation for every Linear. FP32 Embedding/Norm and
    // a tied output head remain unchanged. Weight scale is either fixed or
    // computed independently from each Linear Tensor; activation scale remains fixed.
    [[nodiscard]] Fp8WeightPreparationReport prepare_fp8_inference_weights();
    [[nodiscard]] bool fp8_inference_weights_prepared() const noexcept;
    [[nodiscard]] Int8WeightPreparationReport prepare_int8_inference_weights(
        Int8WeightScaleMode scale_mode = Int8WeightScaleMode::TensorAmax,
        Int8WeightScope scope = Int8WeightScope::AllLinear);
    [[nodiscard]] bool int8_inference_weights_prepared() const noexcept;
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
