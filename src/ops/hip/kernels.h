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

}  // namespace microllm::ops::hip
