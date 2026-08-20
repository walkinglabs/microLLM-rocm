#pragma once

#include <cstdint>

#include <microllm/base/dtype.h>

namespace microllm::ops::hip {

void launch_fill(float* output, std::int64_t elements, float value, void* stream = nullptr);
void launch_adamw_update(float* parameter, const float* gradient,
                         float* first_moment, float* second_moment,
                         std::int64_t elements, float learning_rate,
                         float beta1, float beta2, float epsilon,
                         float weight_decay, float first_correction,
                         float second_correction, void* stream = nullptr);
void launch_add(const float* left, const float* right, float* output,
                std::int64_t elements, void* stream = nullptr);
void launch_add_bias(const float* input, const float* bias, float* output,
                     std::int64_t elements, std::int64_t width, void* stream = nullptr);
void launch_bias_gradient(const float* gradient, float* output,
                          std::int64_t rows, std::int64_t width, void* stream = nullptr);
void launch_multiply(const float* left, const float* right, float* output,
                     std::int64_t elements, void* stream = nullptr);
void launch_scale(const float* input, float* output, std::int64_t elements, float factor,
                  void* stream = nullptr);
void launch_fill_typed(void* output, DType dtype, std::int64_t elements, float value,
                       void* stream = nullptr);
void launch_add_typed(const void* left, const void* right, void* output, DType dtype,
                      std::int64_t elements, void* stream = nullptr);
void launch_multiply_typed(const void* left, const void* right, void* output, DType dtype,
                           std::int64_t elements, void* stream = nullptr);
void launch_scale_typed(const void* input, void* output, DType dtype,
                        std::int64_t elements, float factor, void* stream = nullptr);
void launch_matmul_typed(const void* left, const void* right, void* output, DType dtype,
                         std::int64_t batches, std::int64_t rows, std::int64_t inner,
                         std::int64_t columns, void* stream = nullptr);
void launch_matmul_transposed_typed(
    const void* left, const void* right, void* output, DType dtype,
    std::int64_t left_rows, std::int64_t left_columns,
    std::int64_t right_rows, std::int64_t right_columns,
    bool transpose_left, bool transpose_right, void* stream = nullptr);
void launch_silu_typed(const void* input, void* output, DType dtype,
                       std::int64_t elements, void* stream = nullptr);
void launch_swiglu_typed(const void* gate, const void* up, void* output, DType dtype,
                         std::int64_t elements, void* stream = nullptr);
void launch_quantize_fp8(const void* input, DType input_dtype, void* output,
                         DType fp8_dtype, std::int64_t elements, float inverse_scale,
                         void* stream = nullptr);
void launch_dequantize_fp8(const void* input, DType fp8_dtype, void* output,
                           DType output_dtype, std::int64_t elements, float scale,
                           void* stream = nullptr);
void launch_cast(const void* input, DType input_dtype, void* output,
                 DType output_dtype, std::int64_t elements,
                 void* stream = nullptr);
void launch_matmul(const float* left, const float* right, float* output,
                   std::int64_t batches, std::int64_t rows, std::int64_t inner,
                   std::int64_t columns, void* stream = nullptr);
void launch_embedding(const float* weight, const std::int32_t* indices, float* output,
                      std::int64_t tokens, std::int64_t vocabulary, std::int64_t width,
                      void* stream = nullptr);
void launch_softmax(const float* input, float* output, std::int64_t rows,
                    std::int64_t width, void* stream = nullptr);
void launch_rms_norm(const float* input, const float* weight, float* output,
                     std::int64_t rows, std::int64_t width, float epsilon,
                     void* stream = nullptr);
void launch_add_rms_norm(const float* left, const float* right, const float* weight,
                         float* sum, float* normalized, std::int64_t rows,
                         std::int64_t width, float epsilon, void* stream = nullptr);
void launch_silu(const float* input, float* output, std::int64_t elements,
                 void* stream = nullptr);
void launch_swiglu(const float* gate, const float* up, float* output,
                   std::int64_t elements, void* stream = nullptr);
void launch_rope(const float* input, float* output, std::int64_t elements,
                 std::int64_t head_width, std::int64_t sequence_size,
                 std::int64_t sequence_stride, std::int64_t position_offset, float base,
                 void* stream = nullptr);
void launch_rope_split_half(const float* input, float* output,
                            std::int64_t elements, std::int64_t head_width,
                            std::int64_t sequence_size,
                            std::int64_t sequence_stride,
                            std::int64_t position_offset, float base,
                            void* stream = nullptr);
void launch_rope_split_half_bias(const float* input, const float* bias, float* output,
                                 std::int64_t elements, std::int64_t heads,
                                 std::int64_t sequence_size, std::int64_t head_width,
                                 std::int64_t position_offset, float base,
                                 void* stream = nullptr);
void launch_cross_entropy(const float* logits, const std::int32_t* targets, float* output,
                          float* row_data, std::int64_t rows, std::int64_t classes,
                          void* stream = nullptr);
void launch_reduce_sum(const float* input, float* output, std::int64_t elements,
                       void* stream = nullptr);
void launch_broadcast_scalar(const float* scalar, float* output, std::int64_t elements,
                             void* stream = nullptr);
void launch_embedding_backward(const float* gradient, const std::int32_t* indices,
                               float* weight_gradient, std::int64_t tokens,
                               std::int64_t vocabulary, std::int64_t width,
                               void* stream = nullptr);
void launch_softmax_backward(const float* output, const float* gradient,
                             float* input_gradient, std::int64_t rows,
                             std::int64_t width, void* stream = nullptr);
void launch_rms_norm_backward(const float* input, const float* weight,
                              const float* gradient, float* input_gradient,
                              float* weight_gradient, float* row_inverse_rms,
                              std::int64_t rows,
                              std::int64_t width, float epsilon, void* stream = nullptr);
void launch_silu_backward(const float* input, const float* gradient,
                          float* input_gradient, std::int64_t elements,
                          void* stream = nullptr);
void launch_swiglu_backward(const float* gate, const float* up, const float* gradient,
                            float* gate_gradient, float* up_gradient,
                            std::int64_t elements, void* stream = nullptr);
void launch_rope_backward(const float* gradient, float* input_gradient,
                          std::int64_t elements, std::int64_t head_width,
                          std::int64_t sequence_size, std::int64_t sequence_stride,
                          std::int64_t position_offset, float base,
                          void* stream = nullptr);
void launch_rope_split_half_backward(
    const float* gradient, float* input_gradient, std::int64_t elements,
    std::int64_t head_width, std::int64_t sequence_size,
    std::int64_t sequence_stride, std::int64_t position_offset, float base,
    void* stream = nullptr);
void launch_cross_entropy_backward(const float* logits, const std::int32_t* targets,
                                   const float* loss_gradient, float* logits_gradient,
                                   float* row_stats, float* factor,
                                   std::int64_t rows, std::int64_t classes,
                                   void* stream = nullptr);
void launch_causal_softmax(const float* scores, float* output, std::int64_t rows,
                           std::int64_t sequence, void* stream = nullptr);
void launch_causal_softmax_backward(const float* output, const float* gradient,
                                    float* input_gradient, std::int64_t rows,
                                    std::int64_t sequence, void* stream = nullptr);
void launch_repeat_interleave(const float* input, float* output,
                              std::int64_t output_elements, std::int64_t repeated_width,
                              std::int64_t inner, std::int64_t repeats,
                              void* stream = nullptr);
void launch_repeat_interleave_backward(const float* gradient, float* input_gradient,
                                       std::int64_t input_elements,
                                       std::int64_t input_width, std::int64_t inner,
                                       std::int64_t repeats, void* stream = nullptr);
void launch_kv_cache_store(const float* current, float* cache,
                           std::int64_t heads, std::int64_t capacity,
                           std::int64_t width, std::int64_t position,
                           void* stream = nullptr);
void launch_cached_attention_scores(
    const float* query, const float* key_cache, float* scores,
    std::int64_t heads, std::int64_t kv_heads, std::int64_t sequence,
    std::int64_t cache_head_stride, std::int64_t width,
    std::int64_t repeats, float scale, void* stream = nullptr);
void launch_cached_attention_context(
    const float* probabilities, const float* value_cache, float* output,
    std::int64_t heads, std::int64_t kv_heads, std::int64_t sequence,
    std::int64_t cache_head_stride, std::int64_t width,
    std::int64_t repeats, void* stream = nullptr);
void launch_cached_attention_fused(
    const float* query, const float* key_cache, const float* value_cache,
    float* output, std::int64_t heads, std::int64_t sequence,
    std::int64_t cache_head_stride, std::int64_t width,
    std::int64_t repeats, float scale, void* stream = nullptr);
void launch_argmax(const float* input, std::int32_t* output,
                   std::int64_t elements, void* stream = nullptr);
void launch_argmax_two_stage(const float* input, float* partials,
                             std::int32_t* output, std::int64_t elements,
                             std::int64_t blocks, void* stream = nullptr);

}  // namespace microllm::ops::hip
