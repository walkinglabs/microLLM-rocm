#pragma once

#include <cstdint>

#include <microllm/core/tensor.h>

namespace microllm::ops {

void fill_(Tensor& tensor, float value);

[[nodiscard]] Tensor add(const Tensor& left, const Tensor& right);
[[nodiscard]] Tensor multiply(const Tensor& left, const Tensor& right);
[[nodiscard]] Tensor scale(const Tensor& input, float factor);
[[nodiscard]] Tensor matmul(const Tensor& left, const Tensor& right);
[[nodiscard]] Tensor embedding(const Tensor& weight, const Tensor& indices);
[[nodiscard]] Tensor softmax(const Tensor& input, std::int64_t dim = -1);
[[nodiscard]] Tensor rms_norm(const Tensor& input, const Tensor& weight,
                              float epsilon = 1.0e-5F);
[[nodiscard]] Tensor silu(const Tensor& input);
[[nodiscard]] Tensor swiglu(const Tensor& gate, const Tensor& up);
[[nodiscard]] Tensor rope(const Tensor& input, std::int64_t sequence_dim = 1,
                          std::int64_t position_offset = 0, float base = 10000.0F);
[[nodiscard]] Tensor cross_entropy(const Tensor& logits, const Tensor& targets);

}  // namespace microllm::ops
