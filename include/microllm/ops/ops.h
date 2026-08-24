#pragma once

#include <cstddef>
#include <cstdint>
#include <compare>
#include <filesystem>
#include <memory>
#include <string>
#include <vector>

#include <microllm/core/tensor.h>
#include <microllm/ops/context.h>

namespace microllm::ops {

enum class MatmulImplementation { Auto, Readable, HipBLASLt };
enum class AdamWImplementation { Auto, Scalar, Vectorized };
enum class BiasGradientImplementation { Auto, ScalarColumns, CooperativeRows };
enum class CausalSoftmaxImplementation { Auto, Rows128 };

struct AdamWMultiTensorEntry {
    Tensor* parameter = nullptr;
    const Tensor* gradient = nullptr;
    Tensor* first_moment = nullptr;
    Tensor* second_moment = nullptr;
    Tensor* bf16_mirror = nullptr;
};

struct AdamWMultiTensorStats {
    std::size_t tensors = 0;
    std::size_t blocks = 0;
    std::size_t descriptor_bytes = 0;
    std::size_t block_map_bytes = 0;
};

class AdamWMultiTensorWorkspace {
public:
    AdamWMultiTensorWorkspace() = default;
    AdamWMultiTensorWorkspace(std::vector<std::int64_t> element_counts,
                              Device device);

private:
    struct Impl;
    std::shared_ptr<Impl> impl_;

    friend void adamw_update_multi_(
        AdamWMultiTensorWorkspace&, const std::vector<AdamWMultiTensorEntry>&,
        float, float, float, float, float, float, float, const OpContext&);
    friend AdamWMultiTensorStats adamw_multi_tensor_workspace_stats(
        const AdamWMultiTensorWorkspace&) noexcept;
};

// Stable device-owned step and bias-correction scalars for explicit HIP Graph
// replay. The caller must keep this object alive until every captured launch
// completes and synchronize its step back before checkpointing or returning
// to the ordinary host-stepped optimizer path.
class AdamWGraphStepState {
public:
    AdamWGraphStepState() = default;
    explicit AdamWGraphStepState(Device device,
                                 std::uint64_t initial_step = 0);
    AdamWGraphStepState(const AdamWGraphStepState&) = delete;
    AdamWGraphStepState& operator=(const AdamWGraphStepState&) = delete;
    AdamWGraphStepState(AdamWGraphStepState&&) noexcept = default;
    AdamWGraphStepState& operator=(AdamWGraphStepState&&) noexcept = default;

    [[nodiscard]] bool defined() const noexcept;
    [[nodiscard]] Device device() const noexcept;
    [[nodiscard]] std::uint64_t synchronized_step() const;

private:
    Tensor step_;
    Tensor corrections_;

    friend void adamw_graph_advance_(AdamWGraphStepState&, float, float,
                                     const OpContext&);
    friend void adamw_update_graph_(
        Tensor&, const Tensor&, Tensor&, Tensor&, Tensor*,
        const AdamWGraphStepState&, float, float, float, float, float,
        const OpContext&, AdamWImplementation);
    friend void adamw_update_bf16_moments_graph_(
        Tensor&, const Tensor&, Tensor&, Tensor&, Tensor*,
        const AdamWGraphStepState&, float, float, float, float, float,
        const OpContext&);
};

struct MatmulTuningKey {
    std::int64_t rows = 0;
    std::int64_t inner = 0;
    std::int64_t columns = 0;
    DType dtype = DType::Float32;
    bool transpose_left = false;
    bool transpose_right = false;
    std::vector<std::int64_t> left_strides;
    std::vector<std::int64_t> right_strides;
    std::string architecture;
    int hip_runtime_version = 0;
    int hip_driver_version = 0;
    int hipblaslt_version = 0;
    OpMode mode = OpMode::Unspecified;
    std::size_t workspace_limit = 0;

    auto operator<=>(const MatmulTuningKey&) const = default;
};

struct MatmulTuningCacheLoadReport {
    std::size_t parsed_entries = 0;
    std::size_t loaded_entries = 0;
    std::size_t stale_entries = 0;
};

struct AdamWTuningKey {
    std::int64_t elements = 0;
    DType parameter_dtype = DType::Float32;
    DType gradient_dtype = DType::Float32;
    DType first_moment_dtype = DType::Float32;
    DType second_moment_dtype = DType::Float32;
    bool bf16_mirror = false;
    bool parameter_aligned16 = false;
    bool gradient_aligned16 = false;
    bool first_moment_aligned16 = false;
    bool second_moment_aligned16 = false;
    std::string architecture;
    int hip_runtime_version = 0;
    int hip_driver_version = 0;
    OpMode mode = OpMode::Unspecified;

    auto operator<=>(const AdamWTuningKey&) const = default;
};

struct AdamWTuningCacheLoadReport {
    std::size_t parsed_entries = 0;
    std::size_t loaded_entries = 0;
    std::size_t stale_entries = 0;
};

struct Bf16PlanCacheStats {
    std::size_t entries = 0;
    std::size_t hits = 0;
    std::size_t misses = 0;
};

struct Bf16GroupedQkvKey {
    std::int64_t rows = 0;
    std::int64_t inner = 0;
    std::int64_t query_columns = 0;
    std::int64_t key_columns = 0;
    std::int64_t value_columns = 0;
    std::string architecture;
    int hip_runtime_version = 0;
    int hip_driver_version = 0;
    int hipblaslt_version = 0;

    auto operator<=>(const Bf16GroupedQkvKey&) const = default;
};

struct Bf16GroupedQkvStats {
    std::size_t registered_entries = 0;
    std::size_t algorithm_entries = 0;
    std::size_t algorithm_hits = 0;
    std::size_t algorithm_misses = 0;
    std::size_t kernel_entries = 0;
    std::size_t kernel_hits = 0;
    std::size_t kernel_misses = 0;
    std::size_t plan_entries = 0;
    std::size_t plan_hits = 0;
    std::size_t plan_misses = 0;
    std::size_t dispatches = 0;
    std::size_t retained_query_key_dispatches = 0;
    double kernel_setup_ms = 0.0;
    double argument_setup_ms = 0.0;
};

struct Bf16GroupedGateUpKey {
    std::int64_t rows = 0;
    std::int64_t inner = 0;
    std::int64_t columns = 0;
    std::string architecture;
    int hip_runtime_version = 0;
    int hip_driver_version = 0;
    int hipblaslt_version = 0;

    auto operator<=>(const Bf16GroupedGateUpKey&) const = default;
};

using Bf16GroupedGateUpStats = Bf16GroupedQkvStats;

// Exact hipBLASLt descriptor/environment identity for an explicitly accepted
// FP32 solution index. Leading batch dimensions are flattened exactly as the
// backend sees them; only contiguous operands are supported by this registry.
struct Fp32MatmulSolutionKey {
    std::int64_t batches = 0;
    std::int64_t left_rows = 0;
    std::int64_t left_columns = 0;
    std::int64_t right_rows = 0;
    std::int64_t right_columns = 0;
    std::int64_t output_rows = 0;
    std::int64_t output_columns = 0;
    std::int64_t left_batch_stride = 0;
    std::int64_t right_batch_stride = 0;
    std::int64_t output_batch_stride = 0;
    bool transpose_left = false;
    bool transpose_right = false;
    std::uint32_t alpha_bits = 0;
    std::string architecture;
    int hip_runtime_version = 0;
    int hip_driver_version = 0;
    int hipblaslt_version = 0;
    OpMode mode = OpMode::Unspecified;
    std::size_t workspace_limit = 0;

    auto operator<=>(const Fp32MatmulSolutionKey&) const = default;
};

struct Fp32MatmulSolutionStats {
    std::size_t registered_entries = 0;
    std::size_t cached_algorithms = 0;
    std::size_t registry_hits = 0;
    std::size_t registry_misses = 0;
    std::size_t cache_hits = 0;
    std::size_t cache_misses = 0;
    std::size_t dispatches = 0;
};

struct AttentionLayoutPlanCacheStats {
    std::size_t entries = 0;
    std::size_t hits = 0;
    std::size_t misses = 0;
};

struct Fp8DispatchStats {
    std::size_t native_shapes = 0;
    std::size_t software_fallback_shapes = 0;
    std::size_t software_fallback_calls = 0;
    std::size_t outer_row_fallback_calls = 0;
    // -1 unknown, 0 rejected by the installed runtime, 1 native execution observed.
    int outer_row_native_status = -1;
    std::size_t output_column_scale_calls = 0;
    // -1 unknown, 0 rejected by the installed runtime, 1 native execution observed.
    int output_column_native_status = -1;
};

struct Fp8DynamicQuantStats {
    std::size_t tensor_calls = 0;
    std::size_t row_calls = 0;
    std::uint64_t tensor_elements = 0;
    std::uint64_t row_elements = 0;
    std::size_t column_calls = 0;
    std::uint64_t column_elements = 0;
    std::size_t clipped_tensor_calls = 0;
};

struct TensorPair {
    Tensor first;
    Tensor second;
};

struct TensorTriple {
    Tensor first;
    Tensor second;
    Tensor third;
};

enum class Fp8ScaleMode { Scalar, OuterRow, OuterColumn };

struct ScaledTensor {
    Tensor values;
    Tensor scale;
    float scale_value = 1.0F;
    bool host_scale_available = true;
    Fp8ScaleMode scale_mode = Fp8ScaleMode::Scalar;
};

struct Bf16FfnDiagnostics {
    Tensor input_bf16;
    Tensor gate;
    Tensor up;
    Tensor activated;
    Tensor output;
};

// Caller-owned storage for one BF16 FFN invocation. The fallback Tensor keeps
// FP32-output shapes unsupported by the installed hipBLASLt usable without a
// hidden allocation. Every Tensor is validated by bf16_ffn_out_.
struct Bf16FfnWorkspace {
    Tensor input_bf16;
    Tensor gate;
    Tensor up;
    Tensor activated;
    Tensor output_fallback_bf16;
};

struct Bf16QkvWorkspace {
    Tensor input_bf16;
    Tensor query_fallback_bf16;
    Tensor key_fallback_bf16;
    Tensor value_fallback_bf16;
};

struct CausalGqaAttentionWorkspace {
    Tensor scaled_query;
    Tensor expanded_kv;
    Tensor probabilities;
};

[[nodiscard]] ScaledTensor quantize_fp8(const Tensor& input, DType fp8_dtype,
                                        float scale, const OpContext& context = {});
[[nodiscard]] ScaledTensor quantize_fp8_with_scale(
    const Tensor& input, DType fp8_dtype, float scale_value,
    const Tensor& scale_tensor, const OpContext& context = {});
// Computes one scale for the complete input Tensor on its current device.
// On HIP the returned scale Tensor is authoritative and no scalar is copied to host.
[[nodiscard]] ScaledTensor quantize_fp8_dynamic(
    const Tensor& input, DType fp8_dtype, float minimum_scale,
    const OpContext& context = {}, float maximum_fraction = 1.0F);
[[nodiscard]] ScaledTensor quantize_fp8_rows_dynamic(
    const Tensor& input, DType fp8_dtype, float minimum_scale,
    const OpContext& context = {});
// Computes one device-resident scale per column of a contiguous rank-two Tensor.
// This is intended for [input, output] Linear weights.
[[nodiscard]] ScaledTensor quantize_fp8_columns_dynamic(
    const Tensor& input, DType fp8_dtype, float minimum_scale,
    const OpContext& context = {});
[[nodiscard]] Tensor dequantize_fp8(const ScaledTensor& input, DType output_dtype,
                                    const OpContext& context = {});
[[nodiscard]] Tensor fp8_matmul(const ScaledTensor& left, const ScaledTensor& right,
                                DType output_dtype = DType::BFloat16,
                                const OpContext& context = {});
[[nodiscard]] Tensor cast(const Tensor& input, DType output_dtype,
                          const OpContext& context = {});
void cast_out_(const Tensor& input, Tensor& output, const OpContext& context = {});
void cast_transpose_2d_out_(const Tensor& input, Tensor& output,
                            const OpContext& context = {});
[[nodiscard]] Tensor bf16_matmul(const Tensor& left_fp32, const Tensor& right_bf16,
                                 const OpContext& context = {});
[[nodiscard]] Tensor bf16_matmul_output(const Tensor& left_bf16,
                                        const Tensor& right_bf16,
                                        DType output_dtype,
                                        const OpContext& context = {});
// Writes into caller Storage. A shape-compatible BF16 fallback is required for
// FP32 output so runtime-specific direct-output rejection remains allocation-free.
void bf16_matmul_output_out_(Tensor& output, const Tensor& left_bf16,
                             const Tensor& right_bf16,
                             Tensor& output_fallback_bf16,
                             const OpContext& context = {});
// Runs gate/up -> SwiGLU -> down as one continuous BF16 activation island.
// Input and output stay FP32 at the residual boundary; all three weights must
// already be BF16 so the hot path never creates a hidden persistent copy.
[[nodiscard]] Tensor bf16_ffn(const Tensor& input_fp32,
                              const Tensor& gate_weight_bf16,
                              const Tensor& up_weight_bf16,
                              const Tensor& down_weight_bf16,
                              const OpContext& context = {});
void bf16_ffn_out_(Tensor& output_fp32, Bf16FfnWorkspace& workspace,
                   const Tensor& input_fp32,
                   const Tensor& gate_weight_bf16,
                   const Tensor& up_weight_bf16,
                   const Tensor& down_weight_bf16,
                   const OpContext& context = {});
// Diagnostic-only variant exposing the existing activation island boundaries.
// It executes the same kernels as bf16_ffn and does not copy values to the host.
[[nodiscard]] Bf16FfnDiagnostics bf16_ffn_diagnostics(
    const Tensor& input_fp32, const Tensor& gate_weight_bf16,
    const Tensor& up_weight_bf16, const Tensor& down_weight_bf16,
    const OpContext& context = {});
// Casts the shared FP32 activation once, then submits three BF16-weight
// projections with FP32 outputs. Intended for Q/K/V inference projections.
[[nodiscard]] TensorTriple bf16_qkv_projection(
    const Tensor& input_fp32, const Tensor& query_weight_bf16,
    const Tensor& key_weight_bf16, const Tensor& value_weight_bf16,
    const OpContext& context = {});
// Casts one shared FP32 activation once for the equal-shape gate/up projections.
[[nodiscard]] TensorPair bf16_gate_up_projection(
    const Tensor& input_fp32, const Tensor& gate_weight_bf16,
    const Tensor& up_weight_bf16, const OpContext& context = {});
// Returns true only when query/key remain in the caller's BF16 fallback tensors.
// Value and all non-grouped fallbacks are always materialized in the FP32 outputs.
bool bf16_qkv_projection_out_(
    Tensor& query_output_fp32, Tensor& key_output_fp32,
    Tensor& value_output_fp32, Bf16QkvWorkspace& workspace,
    const Tensor& input_fp32, const Tensor& query_weight_bf16,
    const Tensor& key_weight_bf16, const Tensor& value_weight_bf16,
    const OpContext& context = {}, bool retain_query_key_bf16 = false);

void fill_(Tensor& tensor, float value, const OpContext& context = {});
void adamw_update_(Tensor& parameter, const Tensor& gradient,
                   Tensor& first_moment, Tensor& second_moment,
                   float learning_rate, float beta1, float beta2,
                   float epsilon, float weight_decay,
                   float first_correction, float second_correction,
                   const OpContext& context = {},
                   AdamWImplementation implementation = AdamWImplementation::Auto);
void adamw_update_bf16_mirror_(Tensor& parameter, const Tensor& gradient,
                               Tensor& first_moment, Tensor& second_moment,
                               Tensor& bf16_mirror, float learning_rate,
                               float beta1, float beta2, float epsilon,
                               float weight_decay, float first_correction,
                               float second_correction,
                               const OpContext& context = {},
                               AdamWImplementation implementation =
                                   AdamWImplementation::Auto);
// Experimental reduced-state primitive. Moments are rounded to BF16 before
// they are used for the parameter update, matching the persisted state.
void adamw_update_bf16_moments_(
    Tensor& parameter, const Tensor& gradient, Tensor& first_moment_bf16,
    Tensor& second_moment_bf16, Tensor* parameter_bf16_mirror,
    float learning_rate, float beta1, float beta2, float epsilon,
    float weight_decay, float first_correction, float second_correction,
    const OpContext& context = {});
// Experimental primitive. Entries may use matching FP32 or BF16 moments;
// BF16 recurrence is rounded before the parameter update, as in the
// single-tensor reduced-state primitive.
void adamw_update_multi_(
    AdamWMultiTensorWorkspace& workspace,
    const std::vector<AdamWMultiTensorEntry>& entries,
    float learning_rate, float beta1, float beta2, float epsilon,
    float weight_decay, float first_correction, float second_correction,
    const OpContext& context = {});
// Graph-replayable AdamW primitives. advance must be enqueued exactly once
// before all updates in the captured optimizer region.
void adamw_graph_advance_(AdamWGraphStepState& state, float beta1,
                          float beta2, const OpContext& context = {});
void adamw_update_graph_(
    Tensor& parameter, const Tensor& gradient, Tensor& first_moment,
    Tensor& second_moment, Tensor* bf16_mirror,
    const AdamWGraphStepState& graph_state, float learning_rate,
    float beta1, float beta2, float epsilon, float weight_decay,
    const OpContext& context = {},
    AdamWImplementation implementation = AdamWImplementation::Auto);
void adamw_update_bf16_moments_graph_(
    Tensor& parameter, const Tensor& gradient, Tensor& first_moment_bf16,
    Tensor& second_moment_bf16, Tensor* parameter_bf16_mirror,
    const AdamWGraphStepState& graph_state, float learning_rate,
    float beta1, float beta2, float epsilon, float weight_decay,
    const OpContext& context = {});
[[nodiscard]] AdamWMultiTensorStats adamw_multi_tensor_workspace_stats(
    const AdamWMultiTensorWorkspace& workspace) noexcept;
[[nodiscard]] AdamWTuningKey make_adamw_tuning_key(
    const Tensor& parameter, const Tensor& gradient,
    const Tensor& first_moment, const Tensor& second_moment,
    const Tensor* bf16_mirror = nullptr, const OpContext& context = {});
[[nodiscard]] AdamWImplementation choose_adamw_implementation(
    const Tensor& parameter, const Tensor& gradient,
    const Tensor& first_moment, const Tensor& second_moment,
    const Tensor* bf16_mirror = nullptr, const OpContext& context = {});
void register_adamw_implementation(const AdamWTuningKey& key,
                                   AdamWImplementation implementation);
void clear_adamw_implementation_registry();
[[nodiscard]] std::size_t adamw_registered_implementation_count() noexcept;
void save_adamw_tuning_cache(const std::filesystem::path& path);
[[nodiscard]] AdamWTuningCacheLoadReport load_adamw_tuning_cache(
    const std::filesystem::path& path, Device device,
    bool replace_existing = true);

[[nodiscard]] Tensor add(const Tensor& left, const Tensor& right, const OpContext& context = {});
void add_in_place_(Tensor& destination, const Tensor& source,
                   const OpContext& context = {});
[[nodiscard]] Tensor add_bias(const Tensor& input, const Tensor& bias,
                              const OpContext& context = {});
[[nodiscard]] Tensor bias_gradient(const Tensor& gradient,
                                   const OpContext& context = {});
[[nodiscard]] Tensor bias_gradient_with_implementation(
    const Tensor& gradient, BiasGradientImplementation implementation,
    const OpContext& context = {});
[[nodiscard]] Tensor multiply(const Tensor& left, const Tensor& right,
                              const OpContext& context = {});
[[nodiscard]] Tensor scale(const Tensor& input, float factor, const OpContext& context = {});
[[nodiscard]] Tensor matmul(const Tensor& left, const Tensor& right,
                            const OpContext& context = {});
[[nodiscard]] bool hipblaslt_available() noexcept;
[[nodiscard]] int hipblaslt_version() noexcept;
[[nodiscard]] Bf16PlanCacheStats bf16_plan_cache_stats() noexcept;
void clear_bf16_plan_cache() noexcept;
[[nodiscard]] Bf16GroupedQkvKey make_bf16_grouped_qkv_key(
    std::int64_t rows, std::int64_t inner,
    std::int64_t query_columns, std::int64_t key_columns,
    std::int64_t value_columns, Device device);
void register_bf16_grouped_qkv_algorithm(
    const Bf16GroupedQkvKey& key, int solution_index);
void clear_bf16_grouped_qkv_registry() noexcept;
[[nodiscard]] Bf16GroupedQkvStats bf16_grouped_qkv_stats() noexcept;
[[nodiscard]] Bf16GroupedGateUpKey make_bf16_grouped_gate_up_key(
    std::int64_t rows, std::int64_t inner,
    std::int64_t columns, Device device);
void register_bf16_grouped_gate_up_algorithm(
    const Bf16GroupedGateUpKey& key, int solution_index);
void clear_bf16_grouped_gate_up_registry() noexcept;
[[nodiscard]] Bf16GroupedGateUpStats
bf16_grouped_gate_up_stats() noexcept;
void enable_attention_layout_plan_cache(bool enabled) noexcept;
[[nodiscard]] bool attention_layout_plan_cache_enabled() noexcept;
[[nodiscard]] AttentionLayoutPlanCacheStats
attention_layout_plan_cache_stats() noexcept;
void clear_attention_layout_plan_cache() noexcept;
void enable_attention_gemm_scale_fusion(bool enabled) noexcept;
[[nodiscard]] bool attention_gemm_scale_fusion_enabled() noexcept;
void enable_attention_paired_gqa_repeat(bool enabled) noexcept;
[[nodiscard]] bool attention_paired_gqa_repeat_enabled() noexcept;
void enable_attention_gqa_value_broadcast(bool enabled) noexcept;
[[nodiscard]] bool attention_gqa_value_broadcast_enabled() noexcept;
void enable_attention_gqa_forward_value_broadcast(bool enabled) noexcept;
[[nodiscard]] bool attention_gqa_forward_value_broadcast_enabled() noexcept;
void enable_inference_bthd_attention(bool enabled) noexcept;
[[nodiscard]] bool inference_bthd_attention_enabled() noexcept;
void enable_inference_bthd_bf16_qk(bool enabled) noexcept;
[[nodiscard]] bool inference_bthd_bf16_qk_enabled() noexcept;
void register_bf16_algorithm(std::int64_t rows, std::int64_t inner,
                             std::int64_t columns, DType output_dtype,
                             int solution_index);
void clear_bf16_algorithm_registry() noexcept;
[[nodiscard]] std::size_t bf16_registered_algorithm_count() noexcept;
[[nodiscard]] Fp32MatmulSolutionKey make_fp32_matmul_solution_key(
    const Shape& left_shape, const Shape& right_shape, Device device,
    bool transpose_left = false, bool transpose_right = false,
    const OpContext& context = {}, float alpha = 1.0F);
// Registration is deliberately explicit and thread-local. A solution index is
// never promoted by an operator benchmark alone; callers must first pass their
// complete-model correctness and performance gate on the exact environment.
void register_fp32_matmul_solution(
    const Fp32MatmulSolutionKey& key, int solution_index);
void clear_fp32_matmul_solution_registry() noexcept;
[[nodiscard]] Fp32MatmulSolutionStats
fp32_matmul_solution_stats() noexcept;
[[nodiscard]] Fp8DispatchStats fp8_dispatch_stats() noexcept;
void clear_fp8_dispatch_registry() noexcept;
[[nodiscard]] Fp8DynamicQuantStats fp8_dynamic_quant_stats() noexcept;
void clear_fp8_dynamic_quant_stats() noexcept;
[[nodiscard]] MatmulImplementation choose_matmul_implementation(
    const Tensor& left, const Tensor& right);
[[nodiscard]] MatmulImplementation choose_matmul_implementation(
    const Tensor& left, const Tensor& right,
    bool transpose_left, bool transpose_right);
[[nodiscard]] MatmulImplementation choose_matmul_implementation(
    const Tensor& left, const Tensor& right,
    bool transpose_left, bool transpose_right, const OpContext& context);
[[nodiscard]] MatmulTuningKey make_matmul_tuning_key(
    const Tensor& left, const Tensor& right,
    bool transpose_left = false, bool transpose_right = false,
    const OpContext& context = {});
void register_matmul_implementation(const MatmulTuningKey& key,
                                    MatmulImplementation implementation);
void clear_matmul_implementation_registry();
[[nodiscard]] std::size_t matmul_registered_implementation_count() noexcept;
void save_matmul_tuning_cache(const std::filesystem::path& path);
[[nodiscard]] MatmulTuningCacheLoadReport load_matmul_tuning_cache(
    const std::filesystem::path& path, Device device,
    bool replace_existing = true);
[[nodiscard]] Tensor matmul_with_implementation(
    const Tensor& left, const Tensor& right, MatmulImplementation implementation,
    const OpContext& context = {});
// Computes op(left) @ op(right) without materializing either transpose.  This
// first optimization contract is deliberately limited to contiguous 2D inputs.
[[nodiscard]] Tensor matmul_with_implementation(
    const Tensor& left, const Tensor& right, MatmulImplementation implementation,
    bool transpose_left, bool transpose_right, const OpContext& context = {});
// Writes op(left) @ op(right) into caller-owned Storage. The HIPBLASLt path is
// capture-safe when inputs/output/workspace outlive the captured graph.
void matmul_out_(Tensor& output, const Tensor& left, const Tensor& right,
                 MatmulImplementation implementation,
                 bool transpose_left = false, bool transpose_right = false,
                 const OpContext& context = {});
// Computes factor * (op(left) @ op(right)). hipBLASLt applies factor through
// the GEMM alpha parameter; readable/CPU keeps an explicit composed reference.
[[nodiscard]] Tensor matmul_scaled_with_implementation(
    const Tensor& left, const Tensor& right, float factor,
    MatmulImplementation implementation, bool transpose_left = false,
    bool transpose_right = false, const OpContext& context = {});
[[nodiscard]] Tensor embedding(const Tensor& weight, const Tensor& indices,
                               const OpContext& context = {});
[[nodiscard]] Tensor softmax(const Tensor& input, std::int64_t dim = -1,
                             const OpContext& context = {});
[[nodiscard]] Tensor rms_norm(const Tensor& input, const Tensor& weight,
                              float epsilon = 1.0e-5F, const OpContext& context = {});
// Returns {left + right, rms_norm(left + right, weight)} in one HIP launch.
[[nodiscard]] TensorPair add_rms_norm(const Tensor& left, const Tensor& right,
                                      const Tensor& weight, float epsilon = 1.0e-5F,
                                      const OpContext& context = {});
[[nodiscard]] Tensor silu(const Tensor& input, const OpContext& context = {});
[[nodiscard]] Tensor swiglu(const Tensor& gate, const Tensor& up,
                            const OpContext& context = {});
void swiglu_out_(Tensor& output, const Tensor& gate, const Tensor& up,
                 const OpContext& context = {});
[[nodiscard]] Tensor rope(const Tensor& input, std::int64_t sequence_dim = 1,
                          std::int64_t position_offset = 0, float base = 10000.0F,
                          const OpContext& context = {});
[[nodiscard]] Tensor rope_split_half(const Tensor& input,
                                     std::int64_t sequence_dim = 1,
                                     std::int64_t position_offset = 0,
                                     float base = 10000.0F,
                                     const OpContext& context = {});
// Fuses a Q/K projection bias with split-half RoPE. Input is [B,H,T,D], bias is [H*D].
[[nodiscard]] Tensor rope_split_half_bias(const Tensor& input, const Tensor& bias,
                                          std::int64_t position_offset = 0,
                                          float base = 10000.0F,
                                          const OpContext& context = {});
// Reads a contiguous FP32 or BF16 projection in [B,T,H,D] order, adds
// [H*D] bias and writes
// the rotated result directly in the [B,H,T,D] order consumed by Attention.
// The layout conversion is part of the operator: no transpose materialization is
// required before the call.
[[nodiscard]] Tensor rope_split_half_bias_bthd(
    const Tensor& input, const Tensor& bias,
    std::int64_t position_offset = 0, float base = 10000.0F,
    const OpContext& context = {});
// Decode-only RoPE for [A,H,1,D]. positions[A] supplies one absolute
// position per active request row.
[[nodiscard]] Tensor rope_positions(const Tensor& input, const Tensor& positions,
                                    float base = 10000.0F,
                                    const OpContext& context = {});
[[nodiscard]] Tensor rope_split_half_positions(
    const Tensor& input, const Tensor& positions, float base = 10000.0F,
    const OpContext& context = {});
[[nodiscard]] Tensor rope_split_half_bias_positions(
    const Tensor& input, const Tensor& bias, const Tensor& positions,
    float base = 10000.0F, const OpContext& context = {});
[[nodiscard]] Tensor cross_entropy(const Tensor& logits, const Tensor& targets,
                                   const OpContext& context = {});

// Correctness-first backward primitives.  These are ordinary engine operators rather
// than autograd nodes so the graph engine can use the same CPU reference and HIP
// device-native implementations.
[[nodiscard]] Tensor reduce_sum(const Tensor& input, const OpContext& context = {});
[[nodiscard]] Tensor broadcast_scalar(const Tensor& scalar, Shape shape,
                                      const OpContext& context = {});
[[nodiscard]] Tensor embedding_backward(const Tensor& gradient, const Tensor& indices,
                                        std::int64_t vocabulary,
                                        const OpContext& context = {});
void embedding_backward_add_(Tensor& weight_gradient, const Tensor& gradient,
                             const Tensor& indices,
                             const OpContext& context = {});
[[nodiscard]] Tensor softmax_backward(const Tensor& output, const Tensor& gradient,
                                      const OpContext& context = {});
[[nodiscard]] TensorPair rms_norm_backward(const Tensor& input, const Tensor& weight,
                                           const Tensor& gradient, float epsilon = 1.0e-5F,
                                           const OpContext& context = {});
[[nodiscard]] Tensor silu_backward(const Tensor& input, const Tensor& gradient,
                                   const OpContext& context = {});
[[nodiscard]] TensorPair swiglu_backward(const Tensor& gate, const Tensor& up,
                                         const Tensor& gradient,
                                         const OpContext& context = {});
[[nodiscard]] Tensor rope_backward(const Tensor& gradient, std::int64_t sequence_dim = 1,
                                   std::int64_t position_offset = 0,
                                   float base = 10000.0F,
                                   const OpContext& context = {});
[[nodiscard]] Tensor rope_split_half_backward(
    const Tensor& gradient, std::int64_t sequence_dim = 1,
    std::int64_t position_offset = 0, float base = 10000.0F,
    const OpContext& context = {});
// Inverse of rope_split_half_bias_bthd's layout and rotation. The incoming
// gradient is [B,H,T,D]; the returned projection/bias-input gradient is the
// contiguous [B,T,H,D] tensor expected by the preceding reshape.
[[nodiscard]] Tensor rope_split_half_bias_bthd_backward(
    const Tensor& gradient, std::int64_t position_offset = 0,
    float base = 10000.0F, const OpContext& context = {});
[[nodiscard]] Tensor cross_entropy_backward(const Tensor& logits, const Tensor& targets,
                                            const Tensor& loss_gradient,
                                            const OpContext& context = {});
[[nodiscard]] Tensor causal_softmax(const Tensor& scores, const OpContext& context = {});
[[nodiscard]] Tensor causal_softmax_with_implementation(
    const Tensor& scores, CausalSoftmaxImplementation implementation,
    const OpContext& context = {});
[[nodiscard]] Tensor causal_softmax_backward(const Tensor& output, const Tensor& gradient,
                                             const OpContext& context = {});
// Multiplies causal probabilities [B,H,T,T] by values already stored in
// projection order [B,T,H,D], and writes context in the same [B,T,H,D]
// order. The hipBLASLt path represents interleaved heads with a wide leading
// dimension instead of materializing BHTD inputs and outputs.
[[nodiscard]] Tensor attention_probability_value_bthd(
    const Tensor& probabilities, const Tensor& value,
    const OpContext& context = {});
// GQA variant: broadcasts each V[B,T,KV,D] head across `repeats` query heads
// through a zero batch stride and writes context [B,T,H,D] directly.
[[nodiscard]] Tensor attention_probability_value_gqa_bthd(
    const Tensor& probabilities, const Tensor& value,
    std::int64_t repeats, const OpContext& context = {});
// Computes dP[B,H,T,T] from dO/V in interleaved [B,T,H,D] order.
[[nodiscard]] Tensor attention_probability_gradient_bthd(
    const Tensor& output_gradient, const Tensor& value,
    const OpContext& context = {});
// GQA dP variant using zero batch stride for V[B,T,KV,D].
[[nodiscard]] Tensor attention_probability_gradient_gqa_bthd(
    const Tensor& output_gradient, const Tensor& value,
    std::int64_t repeats, const OpContext& context = {});
// Computes dV[B,T,H,D] = transpose(P) @ dO without materializing either
// interleaved head matrix.
[[nodiscard]] Tensor attention_value_gradient_bthd(
    const Tensor& probabilities, const Tensor& output_gradient,
    const OpContext& context = {});
// Full-sequence causal Attention without materializing repeated K/V heads or
// the T×T score/probability tensors. Shapes are Q[B,H,T,D], K/V[B,KV,T,D].
[[nodiscard]] Tensor causal_gqa_attention(const Tensor& query, const Tensor& key,
                                          const Tensor& value,
                                          std::int64_t repeats, float scale,
                                          const OpContext& context = {});
void causal_gqa_attention_out_(
    Tensor& output, CausalGqaAttentionWorkspace& workspace,
    const Tensor& query, const Tensor& key, const Tensor& value,
    std::int64_t repeats, float scale, const OpContext& context = {});
// Long-sequence training helper: returns {context, causal probabilities}.
[[nodiscard]] TensorPair causal_gqa_attention_saved(
    const Tensor& query, const Tensor& key, const Tensor& value,
    std::int64_t repeats, float scale, const OpContext& context = {});
// Returns FP32 gradients for {query, key, value}; probabilities are recomputed.
[[nodiscard]] TensorTriple causal_gqa_attention_backward(
    const Tensor& query, const Tensor& key, const Tensor& value,
    const Tensor& output_gradient, std::int64_t repeats, float scale,
    const OpContext& context = {});
[[nodiscard]] TensorTriple causal_gqa_attention_backward_saved(
    const Tensor& query, const Tensor& key, const Tensor& value,
    const Tensor& probabilities, const Tensor& output_gradient,
    std::int64_t repeats, float scale, const OpContext& context = {});
// Training layout variant: Q/K stay [B,H,T,D], while V and context use the
// projection/output-linear order [B,T,KV-or-H,D].
[[nodiscard]] Tensor causal_gqa_attention_bthd(
    const Tensor& query, const Tensor& key, const Tensor& value,
    std::int64_t repeats, float scale, const OpContext& context = {});
[[nodiscard]] TensorPair causal_gqa_attention_bthd_saved(
    const Tensor& query, const Tensor& key, const Tensor& value,
    std::int64_t repeats, float scale, const OpContext& context = {});
[[nodiscard]] TensorTriple causal_gqa_attention_bthd_backward(
    const Tensor& query, const Tensor& key, const Tensor& value,
    const Tensor& output_gradient, std::int64_t repeats, float scale,
    const OpContext& context = {});
[[nodiscard]] TensorTriple causal_gqa_attention_bthd_backward_saved(
    const Tensor& query, const Tensor& key, const Tensor& value,
    const Tensor& probabilities, const Tensor& output_gradient,
    std::int64_t repeats, float scale, const OpContext& context = {});
[[nodiscard]] Tensor repeat_interleave(const Tensor& input, std::int64_t dim,
                                       std::int64_t repeats,
                                       const OpContext& context = {});
// Fuses the exact BF16->FP32 conversion with repeat_interleave. The result is
// numerically identical to repeat_interleave(input.cast(FP32), dim, repeats).
[[nodiscard]] Tensor repeat_interleave_bf16_to_float(
    const Tensor& input, std::int64_t dim, std::int64_t repeats,
    const OpContext& context = {});
[[nodiscard]] Tensor repeat_interleave_backward(const Tensor& gradient,
                                                const Shape& input_shape,
                                                std::int64_t dim, std::int64_t repeats,
                                                const OpContext& context = {});
// Expands K[B,KV,T,D] and V[B,T,KV,D] to H=KV*repeats in one pass.
[[nodiscard]] TensorPair repeat_gqa_kv_bthd(
    const Tensor& key, const Tensor& value, std::int64_t repeats,
    const OpContext& context = {});
// Reduces expanded dK[B,H,T,D] and dV[B,T,H,D] to KV heads in one pass.
[[nodiscard]] TensorPair repeat_gqa_kv_bthd_backward(
    const Tensor& key_gradient, const Tensor& value_gradient,
    std::int64_t repeats, const OpContext& context = {});

// Cached decoding primitives. Cache storage is [B, kv_heads, capacity, width]
// while its logical Tensor shape exposes only the initialized prefix. Query/current
// values are FP32; cache storage may be FP32 or BF16 and accumulation stays FP32.
void kv_cache_store_(Tensor& cache, const Tensor& current, std::int64_t position,
                     const OpContext& context = {});
void kv_cache_store_pair_(Tensor& key_cache, Tensor& value_cache,
                          const Tensor& current_key, const Tensor& current_value,
                          std::int64_t position, const OpContext& context = {});
// Stores A active rows into arbitrary cache rows/positions. positions[A] and
// cache_rows[A] are contiguous Int32 tensors on the same device.
void kv_cache_store_pair_positions_(
    Tensor& key_cache, Tensor& value_cache, const Tensor& current_key,
    const Tensor& current_value, const Tensor& positions,
    const Tensor& cache_rows, const OpContext& context = {});
[[nodiscard]] Tensor cached_gqa_attention(const Tensor& query, const Tensor& key_cache,
                                          const Tensor& value_cache,
                                          std::int64_t repeats, float scale,
                                          const OpContext& context = {});
[[nodiscard]] Tensor cached_gqa_attention_positions(
    const Tensor& query, const Tensor& key_cache, const Tensor& value_cache,
    const Tensor& positions, const Tensor& cache_rows,
    std::int64_t repeats, float scale, const OpContext& context = {});
// Returns one int32 index with a smallest-index tie rule.  The result is -1 when
// any input is non-finite, allowing asynchronous device execution to stay visible.
[[nodiscard]] Tensor argmax(const Tensor& input, const OpContext& context = {});
void argmax_out_(const Tensor& input, Tensor& output,
                 const OpContext& context = {});
// Reduces only the last dimension and preserves every leading dimension.
// Each row follows argmax's smallest-index and non-finite-to--1 contract.
[[nodiscard]] Tensor argmax_last_dim(const Tensor& input,
                                     const OpContext& context = {});
void argmax_last_dim_out_(const Tensor& input, Tensor& output,
                          const OpContext& context = {});

}  // namespace microllm::ops
