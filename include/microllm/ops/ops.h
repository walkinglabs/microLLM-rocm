#pragma once

#include <cstddef>
#include <cstdint>

#include <microllm/core/tensor.h>
#include <microllm/ops/context.h>

namespace microllm::ops {

enum class MatmulImplementation { Auto, Readable, HipBLASLt };

struct Bf16PlanCacheStats {
    std::size_t entries = 0;
    std::size_t hits = 0;
    std::size_t misses = 0;
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

struct ScaledTensor {
    Tensor values;
    Tensor scale;
    float scale_value = 1.0F;
};

[[nodiscard]] ScaledTensor quantize_fp8(const Tensor& input, DType fp8_dtype,
                                        float scale, const OpContext& context = {});
[[nodiscard]] Tensor dequantize_fp8(const ScaledTensor& input, DType output_dtype,
                                    const OpContext& context = {});
[[nodiscard]] Tensor fp8_matmul(const ScaledTensor& left, const ScaledTensor& right,
                                DType output_dtype = DType::BFloat16,
                                const OpContext& context = {});
[[nodiscard]] Tensor cast(const Tensor& input, DType output_dtype,
                          const OpContext& context = {});
[[nodiscard]] Tensor bf16_matmul(const Tensor& left_fp32, const Tensor& right_bf16,
                                 const OpContext& context = {});
[[nodiscard]] Tensor bf16_matmul_output(const Tensor& left_bf16,
                                        const Tensor& right_bf16,
                                        DType output_dtype,
                                        const OpContext& context = {});
// Runs gate/up -> SwiGLU -> down as one continuous BF16 activation island.
// Input and output stay FP32 at the residual boundary; all three weights must
// already be BF16 so the hot path never creates a hidden persistent copy.
[[nodiscard]] Tensor bf16_ffn(const Tensor& input_fp32,
                              const Tensor& gate_weight_bf16,
                              const Tensor& up_weight_bf16,
                              const Tensor& down_weight_bf16,
                              const OpContext& context = {});
// Casts the shared FP32 activation once, then submits three BF16-weight
// projections with FP32 outputs. Intended for Q/K/V inference projections.
[[nodiscard]] TensorTriple bf16_qkv_projection(
    const Tensor& input_fp32, const Tensor& query_weight_bf16,
    const Tensor& key_weight_bf16, const Tensor& value_weight_bf16,
    const OpContext& context = {});

void fill_(Tensor& tensor, float value, const OpContext& context = {});
void adamw_update_(Tensor& parameter, const Tensor& gradient,
                   Tensor& first_moment, Tensor& second_moment,
                   float learning_rate, float beta1, float beta2,
                   float epsilon, float weight_decay,
                   float first_correction, float second_correction,
                   const OpContext& context = {});

[[nodiscard]] Tensor add(const Tensor& left, const Tensor& right, const OpContext& context = {});
[[nodiscard]] Tensor add_bias(const Tensor& input, const Tensor& bias,
                              const OpContext& context = {});
[[nodiscard]] Tensor bias_gradient(const Tensor& gradient,
                                   const OpContext& context = {});
[[nodiscard]] Tensor multiply(const Tensor& left, const Tensor& right,
                              const OpContext& context = {});
[[nodiscard]] Tensor scale(const Tensor& input, float factor, const OpContext& context = {});
[[nodiscard]] Tensor matmul(const Tensor& left, const Tensor& right,
                            const OpContext& context = {});
[[nodiscard]] bool hipblaslt_available() noexcept;
[[nodiscard]] Bf16PlanCacheStats bf16_plan_cache_stats() noexcept;
void clear_bf16_plan_cache() noexcept;
[[nodiscard]] MatmulImplementation choose_matmul_implementation(
    const Tensor& left, const Tensor& right);
void register_matmul_implementation(std::int64_t rows, std::int64_t inner,
                                    std::int64_t columns,
                                    MatmulImplementation implementation);
void clear_matmul_implementation_registry();
[[nodiscard]] Tensor matmul_with_implementation(
    const Tensor& left, const Tensor& right, MatmulImplementation implementation,
    const OpContext& context = {});
// Computes op(left) @ op(right) without materializing either transpose.  This
// first optimization contract is deliberately limited to contiguous 2D inputs.
[[nodiscard]] Tensor matmul_with_implementation(
    const Tensor& left, const Tensor& right, MatmulImplementation implementation,
    bool transpose_left, bool transpose_right, const OpContext& context = {});
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
[[nodiscard]] Tensor cross_entropy_backward(const Tensor& logits, const Tensor& targets,
                                            const Tensor& loss_gradient,
                                            const OpContext& context = {});
[[nodiscard]] Tensor causal_softmax(const Tensor& scores, const OpContext& context = {});
[[nodiscard]] Tensor causal_softmax_backward(const Tensor& output, const Tensor& gradient,
                                             const OpContext& context = {});
[[nodiscard]] Tensor repeat_interleave(const Tensor& input, std::int64_t dim,
                                       std::int64_t repeats,
                                       const OpContext& context = {});
[[nodiscard]] Tensor repeat_interleave_backward(const Tensor& gradient,
                                                const Shape& input_shape,
                                                std::int64_t dim, std::int64_t repeats,
                                                const OpContext& context = {});

// Cached decoding primitives.  Cache storage is [1, kv_heads, capacity, width]
// while its logical Tensor shape exposes only the initialized prefix.
void kv_cache_store_(Tensor& cache, const Tensor& current, std::int64_t position,
                     const OpContext& context = {});
void kv_cache_store_pair_(Tensor& key_cache, Tensor& value_cache,
                          const Tensor& current_key, const Tensor& current_value,
                          std::int64_t position, const OpContext& context = {});
[[nodiscard]] Tensor cached_gqa_attention(const Tensor& query, const Tensor& key_cache,
                                          const Tensor& value_cache,
                                          std::int64_t repeats, float scale,
                                          const OpContext& context = {});
// Returns one int32 index with a smallest-index tie rule.  The result is -1 when
// any input is non-finite, allowing asynchronous device execution to stay visible.
[[nodiscard]] Tensor argmax(const Tensor& input, const OpContext& context = {});

}  // namespace microllm::ops
