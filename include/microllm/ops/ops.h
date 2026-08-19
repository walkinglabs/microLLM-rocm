#pragma once

#include <cstdint>

#include <microllm/core/tensor.h>
#include <microllm/ops/context.h>

namespace microllm::ops {

enum class MatmulImplementation { Auto, Readable, HipBLASLt };

void fill_(Tensor& tensor, float value, const OpContext& context = {});

[[nodiscard]] Tensor add(const Tensor& left, const Tensor& right, const OpContext& context = {});
[[nodiscard]] Tensor multiply(const Tensor& left, const Tensor& right,
                              const OpContext& context = {});
[[nodiscard]] Tensor scale(const Tensor& input, float factor, const OpContext& context = {});
[[nodiscard]] Tensor matmul(const Tensor& left, const Tensor& right,
                            const OpContext& context = {});
[[nodiscard]] bool hipblaslt_available() noexcept;
[[nodiscard]] MatmulImplementation choose_matmul_implementation(
    const Tensor& left, const Tensor& right);
void register_matmul_implementation(std::int64_t rows, std::int64_t inner,
                                    std::int64_t columns,
                                    MatmulImplementation implementation);
void clear_matmul_implementation_registry();
[[nodiscard]] Tensor matmul_with_implementation(
    const Tensor& left, const Tensor& right, MatmulImplementation implementation,
    const OpContext& context = {});
[[nodiscard]] Tensor embedding(const Tensor& weight, const Tensor& indices,
                               const OpContext& context = {});
[[nodiscard]] Tensor softmax(const Tensor& input, std::int64_t dim = -1,
                             const OpContext& context = {});
[[nodiscard]] Tensor rms_norm(const Tensor& input, const Tensor& weight,
                              float epsilon = 1.0e-5F, const OpContext& context = {});
[[nodiscard]] Tensor silu(const Tensor& input, const OpContext& context = {});
[[nodiscard]] Tensor swiglu(const Tensor& gate, const Tensor& up,
                            const OpContext& context = {});
[[nodiscard]] Tensor rope(const Tensor& input, std::int64_t sequence_dim = 1,
                          std::int64_t position_offset = 0, float base = 10000.0F,
                          const OpContext& context = {});
[[nodiscard]] Tensor cross_entropy(const Tensor& logits, const Tensor& targets,
                                   const OpContext& context = {});

}  // namespace microllm::ops
