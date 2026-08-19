#pragma once

#include <cstdint>

namespace microllm::ops::hip {

void launch_fill(float* output, std::int64_t elements, float value, void* stream = nullptr);
void launch_add(const float* left, const float* right, float* output,
                std::int64_t elements, void* stream = nullptr);
void launch_multiply(const float* left, const float* right, float* output,
                     std::int64_t elements, void* stream = nullptr);
void launch_scale(const float* input, float* output, std::int64_t elements, float factor,
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
void launch_silu(const float* input, float* output, std::int64_t elements,
                 void* stream = nullptr);
void launch_swiglu(const float* gate, const float* up, float* output,
                   std::int64_t elements, void* stream = nullptr);
void launch_rope(const float* input, float* output, std::int64_t elements,
                 std::int64_t head_width, std::int64_t sequence_size,
                 std::int64_t sequence_stride, std::int64_t position_offset, float base,
                 void* stream = nullptr);
void launch_cross_entropy(const float* logits, const std::int32_t* targets, float* output,
                          std::int64_t rows, std::int64_t classes,
                          void* stream = nullptr);

}  // namespace microllm::ops::hip
