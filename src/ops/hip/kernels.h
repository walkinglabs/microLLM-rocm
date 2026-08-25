#pragma once

#include <cstddef>
#include <cstdint>

#include <microllm/base/dtype.h>

namespace microllm::ops::hip {

struct AdamWMultiTensorDescriptor {
    float* parameter = nullptr;
    const float* gradient = nullptr;
    void* first_moment = nullptr;
    void* second_moment = nullptr;
    void* bf16_mirror = nullptr;
    std::int64_t elements = 0;
    std::int64_t first_block = 0;
    std::uint32_t bf16_moments = 0;
};

void* create_adamw_descriptor_staging(std::size_t bytes);
void destroy_adamw_descriptor_staging(void* staging) noexcept;
void* acquire_adamw_descriptor_staging(void* staging);
void mark_adamw_descriptor_staging_in_use(
    void* staging, void* stream = nullptr);

void launch_fill(float* output, std::int64_t elements, float value, void* stream = nullptr);
void launch_adamw_update(float* parameter, const float* gradient,
                         float* first_moment, float* second_moment,
                         void* bf16_mirror,
                         std::int64_t elements, float learning_rate,
                         float beta1, float beta2, float epsilon,
                         float weight_decay, float first_correction,
                         float second_correction, void* stream = nullptr);
void launch_adamw_update_vectorized(float* parameter, const float* gradient,
                                    float* first_moment, float* second_moment,
                                    void* bf16_mirror, std::int64_t elements,
                                    float learning_rate, float beta1, float beta2,
                                    float epsilon, float weight_decay,
                                    float first_correction, float second_correction,
                                    void* stream = nullptr);
void launch_adamw_update_multi(
    const AdamWMultiTensorDescriptor* descriptors,
    const std::int32_t* block_to_tensor, std::int64_t blocks,
    float learning_rate, float beta1, float beta2, float epsilon,
    float weight_decay, float first_correction, float second_correction,
    void* stream = nullptr);
void launch_adamw_update_multi_graph(
    const AdamWMultiTensorDescriptor* descriptors,
    const std::int32_t* block_to_tensor, std::int64_t blocks,
    float learning_rate, float beta1, float beta2, float epsilon,
    float weight_decay, const float* corrections,
    void* stream = nullptr);
void launch_adamw_update_bf16_moments(
    float* parameter, const float* gradient, void* first_moment_bf16,
    void* second_moment_bf16, void* parameter_bf16_mirror,
    std::int64_t elements, float learning_rate, float beta1, float beta2,
    float epsilon, float weight_decay, float first_correction,
    float second_correction, void* stream = nullptr);
void launch_adamw_graph_advance(std::int32_t* step, float* corrections,
                                float beta1, float beta2,
                                void* stream = nullptr);
void launch_adamw_update_graph(
    float* parameter, const float* gradient, float* first_moment,
    float* second_moment, void* bf16_mirror, std::int64_t elements,
    float learning_rate, float beta1, float beta2, float epsilon,
    float weight_decay, const float* corrections, bool vectorized,
    void* stream = nullptr);
void launch_adamw_update_bf16_moments_graph(
    float* parameter, const float* gradient, void* first_moment_bf16,
    void* second_moment_bf16, void* parameter_bf16_mirror,
    std::int64_t elements, float learning_rate, float beta1, float beta2,
    float epsilon, float weight_decay, const float* corrections,
    void* stream = nullptr);
void launch_add(const float* left, const float* right, float* output,
                std::int64_t elements, void* stream = nullptr);
void launch_add_bias(const float* input, const float* bias, float* output,
                     std::int64_t elements, std::int64_t width, void* stream = nullptr);
void launch_add_bias_bf16(const void* input_bf16, const float* bias,
                          void* output_bf16, std::int64_t elements,
                          std::int64_t width, void* stream = nullptr);
void launch_bias_gradient(const float* gradient, float* output,
                          std::int64_t rows, std::int64_t width, void* stream = nullptr);
void launch_bias_gradient_cooperative(const float* gradient, float* output,
                                      std::int64_t rows, std::int64_t width,
                                      void* stream = nullptr);
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
                         std::int64_t elements, void* stream = nullptr,
                         bool vectorized = false);
void launch_quantize_fp8(const void* input, DType input_dtype, void* output,
                         DType fp8_dtype, std::int64_t elements, float inverse_scale,
                         void* stream = nullptr);
void launch_quantize_fp8_dynamic(const void* input, DType input_dtype,
                                 void* output, DType fp8_dtype,
                                 float* scale, float* partial_maxima,
                                 std::int64_t partial_count,
                                 std::int64_t elements,
                                 float minimum_scale,
                                 float maximum_fraction,
                                 void* stream = nullptr);
void launch_quantize_fp8_rows_dynamic(
    const void* input, DType input_dtype, void* output, DType fp8_dtype,
    float* scales, std::int64_t rows, std::int64_t columns,
    float minimum_scale, void* stream = nullptr);
void launch_quantize_fp8_columns_dynamic(
    const void* input, DType input_dtype, void* output, DType fp8_dtype,
    float* scales, std::int64_t rows, std::int64_t columns,
    float minimum_scale, void* stream = nullptr);
void launch_dequantize_fp8(const void* input, DType fp8_dtype, void* output,
                           DType output_dtype, std::int64_t elements, float scale,
                           void* stream = nullptr);
void launch_dequantize_fp8_device_scale(
    const void* input, DType fp8_dtype, void* output, DType output_dtype,
    std::int64_t elements, const float* scale, void* stream = nullptr);
void launch_dequantize_fp8_row_scales(
    const void* input, DType fp8_dtype, void* output, DType output_dtype,
    std::int64_t rows, std::int64_t columns, const float* scales,
    void* stream = nullptr);
void launch_dequantize_fp8_column_scales(
    const void* input, DType fp8_dtype, void* output, DType output_dtype,
    std::int64_t rows, std::int64_t columns, const float* scales,
    void* stream = nullptr);
void launch_scale_columns_by_first(
    void* values, DType dtype, std::int64_t rows, std::int64_t columns,
    const float* scales, void* stream = nullptr);
void launch_cast(const void* input, DType input_dtype, void* output,
                 DType output_dtype, std::int64_t elements,
                 void* stream = nullptr);
void launch_cast_transpose_2d(const void* input, DType input_dtype,
                              void* output, DType output_dtype,
                              std::int64_t rows, std::int64_t columns,
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
void launch_rms_norm_bf16_output(
    const float* input, const float* weight, void* output_bf16,
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
void launch_rope_split_half_bias_bthd(
    const void* input, DType input_dtype, const float* bias, float* output,
    std::int64_t batches, std::int64_t sequence_size, std::int64_t heads,
    std::int64_t head_width, std::int64_t position_offset, float base,
    void* stream = nullptr);
void launch_rope_split_half_bias_bthd_bf16(
    const void* input_bf16, const float* bias, void* output_bf16,
    std::int64_t batches, std::int64_t sequence_size, std::int64_t heads,
    std::int64_t head_width, std::int64_t position_offset, float base,
    void* stream = nullptr);
void launch_rope_positions(const float* input, const std::int32_t* positions,
                           float* output, std::int64_t batches,
                           std::int64_t heads, std::int64_t head_width,
                           float base, void* stream = nullptr);
void launch_rope_split_half_positions(
    const float* input, const std::int32_t* positions, float* output,
    std::int64_t batches, std::int64_t heads, std::int64_t head_width,
    float base, void* stream = nullptr);
void launch_rope_split_half_bias_positions(
    const float* input, const float* bias, const std::int32_t* positions,
    float* output, std::int64_t batches, std::int64_t heads,
    std::int64_t head_width, float base, void* stream = nullptr);
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
void launch_rope_split_half_bias_bthd_backward(
    const float* gradient, float* input_gradient,
    std::int64_t batches, std::int64_t sequence_size, std::int64_t heads,
    std::int64_t head_width, std::int64_t position_offset, float base,
    void* stream = nullptr);
void launch_cross_entropy_backward(const float* logits, const std::int32_t* targets,
                                   const float* loss_gradient, float* logits_gradient,
                                   float* row_stats, float* factor,
                                   std::int64_t rows, std::int64_t classes,
                                   void* stream = nullptr);
void launch_causal_softmax(const float* scores, float* output, std::int64_t rows,
                           std::int64_t sequence, void* stream = nullptr,
                           bool use_128_threads = false);
void launch_causal_softmax_backward(const float* output, const float* gradient,
                                    float* input_gradient, std::int64_t rows,
                                    std::int64_t sequence, void* stream = nullptr);
void launch_causal_gqa_attention(
    const float* query, const float* key, const float* value, float* output,
    float* saved_probabilities,
    std::int64_t batches, std::int64_t heads, std::int64_t kv_heads,
    std::int64_t sequence, std::int64_t width, std::int64_t repeats,
    float scale, void* stream = nullptr);
void launch_causal_gqa_attention_backward(
    const float* query, const float* key, const float* value,
    const float* output_gradient, float* query_gradient,
    float* key_gradient, float* value_gradient,
    std::int64_t batches, std::int64_t heads, std::int64_t kv_heads,
    std::int64_t sequence, std::int64_t width, std::int64_t repeats,
    float scale, void* stream = nullptr);
void launch_causal_gqa_attention_backward_rows(
    const float* query, const float* key, const float* value,
    const float* output_gradient, float* query_gradient,
    float* probabilities, float* scaled_score_gradients,
    std::int64_t batches, std::int64_t heads, std::int64_t kv_heads,
    std::int64_t sequence, std::int64_t width, std::int64_t repeats,
    float scale, void* stream = nullptr);
void launch_causal_gqa_attention_backward_saved_rows(
    const float* key, const float* value, const float* output_gradient,
    const float* probabilities, float* query_gradient,
    float* scaled_score_gradients,
    std::int64_t batches, std::int64_t heads, std::int64_t kv_heads,
    std::int64_t sequence, std::int64_t width, std::int64_t repeats,
    float scale, void* stream = nullptr);
void launch_rocwmma_online_gqa_attention_bthd(
    const void* query_bf16, const void* key_bf16, const void* value_bf16,
    float* output, std::int64_t batches, std::int64_t heads,
    std::int64_t kv_heads, std::int64_t sequence, std::int64_t width,
    float scale, void* stream = nullptr);
void launch_repeat_interleave(const float* input, float* output,
                              std::int64_t output_elements, std::int64_t repeated_width,
                              std::int64_t inner, std::int64_t repeats,
                              void* stream = nullptr);
void launch_repeat_interleave_bf16_to_float(
    const void* input, float* output, std::int64_t output_elements,
    std::int64_t repeated_width, std::int64_t inner,
    std::int64_t repeats, void* stream = nullptr);
void launch_repeat_interleave_backward(const float* gradient, float* input_gradient,
                                       std::int64_t input_elements,
                                       std::int64_t input_width, std::int64_t inner,
                                       std::int64_t repeats, void* stream = nullptr);
void launch_repeat_gqa_kv_bthd(
    const float* key, const float* value, float* expanded_key,
    float* expanded_value, std::int64_t batches, std::int64_t kv_heads,
    std::int64_t sequence, std::int64_t width, std::int64_t repeats,
    void* stream = nullptr);
void launch_repeat_gqa_kv_bthd_backward(
    const float* key_gradient, const float* value_gradient,
    float* reduced_key_gradient, float* reduced_value_gradient,
    std::int64_t batches, std::int64_t kv_heads,
    std::int64_t sequence, std::int64_t width, std::int64_t repeats,
    void* stream = nullptr);
void launch_kv_cache_store(const float* current, void* cache, DType cache_dtype,
                           std::int64_t batches, std::int64_t heads,
                           std::int64_t capacity,
                           std::int64_t width, std::int64_t position,
                           void* stream = nullptr);
void launch_kv_cache_store_pair(const float* current_key, const float* current_value,
                                void* key_cache, void* value_cache,
                                DType cache_dtype,
                                std::int64_t batches, std::int64_t heads,
                                std::int64_t capacity,
                                std::int64_t width, std::int64_t position,
                                void* stream = nullptr);
void launch_kv_cache_store_pair_positions(
    const float* current_key, const float* current_value,
    const std::int32_t* positions, const std::int32_t* cache_rows,
    void* key_cache, void* value_cache, DType cache_dtype,
    std::int64_t active_batches, std::int64_t cache_batches,
    std::int64_t heads, std::int64_t capacity,
    std::int64_t logical_prefix, std::int64_t width,
    void* stream = nullptr);
void launch_cached_attention_scores(
    const float* query, const void* key_cache, DType cache_dtype, float* scores,
    std::int64_t batches, std::int64_t heads, std::int64_t kv_heads,
    std::int64_t sequence, std::int64_t cache_batch_stride,
    std::int64_t cache_head_stride, std::int64_t width,
    std::int64_t repeats, float scale, void* stream = nullptr);
void launch_cached_attention_context(
    const float* probabilities, const void* value_cache, DType cache_dtype,
    float* output,
    std::int64_t batches, std::int64_t heads, std::int64_t kv_heads,
    std::int64_t sequence, std::int64_t cache_batch_stride,
    std::int64_t cache_head_stride, std::int64_t width,
    std::int64_t repeats, void* stream = nullptr);
void launch_cached_attention_fused(
    const float* query, const void* key_cache, const void* value_cache,
    DType cache_dtype, float* output, std::int64_t batches, std::int64_t heads,
    std::int64_t sequence, std::int64_t cache_batch_stride,
    std::int64_t cache_head_stride, std::int64_t width,
    std::int64_t repeats, float scale, void* stream = nullptr);
void launch_cached_attention_fused_positions(
    const float* query, const void* key_cache, const void* value_cache,
    const std::int32_t* positions, const std::int32_t* cache_rows,
    DType cache_dtype, float* output, std::int64_t active_batches,
    std::int64_t cache_batches, std::int64_t heads,
    std::int64_t logical_prefix, std::int64_t cache_batch_stride,
    std::int64_t cache_head_stride, std::int64_t width,
    std::int64_t repeats, float scale, void* stream = nullptr);
void launch_cached_attention_scores_positions(
    const float* query, const void* key_cache, const std::int32_t* positions,
    const std::int32_t* cache_rows, DType cache_dtype, float* scores,
    std::int64_t active_batches, std::int64_t cache_batches,
    std::int64_t heads, std::int64_t logical_prefix,
    std::int64_t cache_batch_stride, std::int64_t cache_head_stride,
    std::int64_t width, std::int64_t repeats, float scale,
    void* stream = nullptr);
void launch_cached_attention_context_positions(
    const float* probabilities, const void* value_cache,
    const std::int32_t* positions, const std::int32_t* cache_rows,
    DType cache_dtype, float* output, std::int64_t active_batches,
    std::int64_t cache_batches, std::int64_t heads,
    std::int64_t logical_prefix, std::int64_t cache_batch_stride,
    std::int64_t cache_head_stride, std::int64_t width,
    std::int64_t repeats, void* stream = nullptr);
void launch_argmax(const float* input, std::int32_t* output,
                   std::int64_t elements, void* stream = nullptr);
void launch_argmax_last_dim(const float* input, std::int32_t* output,
                            std::int64_t rows, std::int64_t classes,
                            void* stream = nullptr);
void launch_argmax_two_stage(const float* input, float* partials,
                             std::int32_t* output, std::int64_t elements,
                             std::int64_t blocks, void* stream = nullptr);

}  // namespace microllm::ops::hip
