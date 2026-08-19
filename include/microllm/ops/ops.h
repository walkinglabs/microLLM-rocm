#pragma once

#include <cstdint>

#include <microllm/core/tensor.h>
#include <microllm/ops/context.h>

namespace microllm::ops {

enum class MatmulImplementation { Auto, Readable, HipBLASLt };

struct TensorPair {
    Tensor first;
    Tensor second;
};

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

}  // namespace microllm::ops
