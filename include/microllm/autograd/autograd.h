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
};

[[nodiscard]] Value add(const Value& left, const Value& right);
[[nodiscard]] Value multiply(const Value& left, const Value& right);
[[nodiscard]] Value scale(const Value& input, float factor);
[[nodiscard]] Value matmul(const Value& left, const Value& right);
[[nodiscard]] Value sum(const Value& input);
[[nodiscard]] Value mean(const Value& input);

}  // namespace microllm::autograd
