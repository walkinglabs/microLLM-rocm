#pragma once

#include <memory>

#include <microllm/core/tensor.h>

namespace microllm::autograd {

class Value {
public:
    struct Node;

    Value() = default;
    explicit Value(Tensor data, bool requires_grad = false);

    [[nodiscard]] bool defined() const noexcept;
    [[nodiscard]] bool requires_grad() const noexcept;
    [[nodiscard]] const Tensor& data() const;
    [[nodiscard]] Tensor& mutable_data();
    [[nodiscard]] bool has_grad() const noexcept;
    [[nodiscard]] const Tensor& grad() const;
    void set_grad(Tensor gradient);
    void zero_grad();
    void backward() const;
    void backward(const Tensor& gradient) const;
    [[nodiscard]] Value detach() const;

private:
    explicit Value(std::shared_ptr<Node> node);
    std::shared_ptr<Node> node_;

    friend struct ValueAccess;

    friend Value add(const Value&, const Value&);
    friend Value multiply(const Value&, const Value&);
    friend Value scale(const Value&, float);
    friend Value matmul(const Value&, const Value&);
    friend Value sum(const Value&);
    friend Value mean(const Value&);
    friend Value reshape(const Value&, Shape);
    friend Value transpose(const Value&, std::int64_t, std::int64_t);
    friend Value embedding(const Value&, const Tensor&);
    friend Value softmax(const Value&, std::int64_t);
    friend Value rms_norm(const Value&, const Value&, float);
    friend Value silu(const Value&);
    friend Value swiglu(const Value&, const Value&);
    friend Value rope(const Value&, std::int64_t, std::int64_t, float);
    friend Value cross_entropy(const Value&, const Tensor&);
    friend Value contiguous(const Value&);
    friend Value causal_softmax(const Value&);
    friend Value repeat_interleave(const Value&, std::int64_t, std::int64_t);
};

[[nodiscard]] Value add(const Value& left, const Value& right);
[[nodiscard]] Value multiply(const Value& left, const Value& right);
[[nodiscard]] Value scale(const Value& input, float factor);
[[nodiscard]] Value matmul(const Value& left, const Value& right);
[[nodiscard]] Value sum(const Value& input);
[[nodiscard]] Value mean(const Value& input);
[[nodiscard]] Value reshape(const Value& input, Shape shape);
[[nodiscard]] Value transpose(const Value& input, std::int64_t dim0, std::int64_t dim1);
[[nodiscard]] Value embedding(const Value& weight, const Tensor& indices);
[[nodiscard]] Value softmax(const Value& input, std::int64_t dim = -1);
[[nodiscard]] Value rms_norm(const Value& input, const Value& weight,
                             float epsilon = 1.0e-5F);
[[nodiscard]] Value silu(const Value& input);
[[nodiscard]] Value swiglu(const Value& gate, const Value& up);
[[nodiscard]] Value rope(const Value& input, std::int64_t sequence_dim = 1,
                         std::int64_t position_offset = 0, float base = 10000.0F);
[[nodiscard]] Value cross_entropy(const Value& logits, const Tensor& targets);
[[nodiscard]] Value contiguous(const Value& input);
[[nodiscard]] Value causal_softmax(const Value& scores);
[[nodiscard]] Value repeat_interleave(const Value& input, std::int64_t dim,
                                      std::int64_t repeats);

}  // namespace microllm::autograd
