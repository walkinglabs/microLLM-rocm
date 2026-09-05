#pragma once

#include <cstddef>
#include <functional>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <microllm/core/tensor.h>

namespace microllm::autograd {

struct ValueTriple;
struct MoeRouterResult;

struct GraphNodeInfo {
    std::size_t id = 0;
    std::string operation;
    Shape shape;
    bool requires_grad = false;
    std::vector<std::size_t> parents;
};

struct GraphSnapshot {
    // Nodes are in topological order: every parent appears before its child.
    std::vector<GraphNodeInfo> nodes;
    std::size_t root_id = 0;
    std::size_t edge_count = 0;
};

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
    // Explicit caller-owned leaf gradient destination. Contributions are
    // accumulated in-place even when another owner retains the Storage.
    void bind_grad_buffer(Tensor gradient, bool zero = true);
    void unbind_grad_buffer() noexcept;
    [[nodiscard]] bool grad_buffer_bound() const noexcept;
    void set_gradient_ready_hook(std::function<void()> hook);
    void clear_gradient_ready_hook() noexcept;
    void zero_grad();
    void backward() const;
    void backward(const Tensor& gradient) const;
    [[nodiscard]] Value detach() const;

private:
    explicit Value(std::shared_ptr<Node> node);
    std::shared_ptr<Node> node_;

    friend struct ValueAccess;

    friend Value add(const Value&, const Value&);
    friend Value add_bias(const Value&, const Value&);
    friend Value multiply(const Value&, const Value&);
    friend Value scale(const Value&, float);
    friend Value matmul(const Value&, const Value&);
    friend Value matmul(const Value&, const Value&, bool, bool);
    friend Value fp8_matmul(const Value&, const Value&, float, float, DType,
                            DType);
    friend Value bf16_matmul(const Value&, const Value&);
    friend Value bf16_matmul(const Value&, const Value&, const Tensor&);
    friend std::pair<Value, Value> bf16_gate_up_projection(
        const Value&, const Value&, const Tensor&, const Value&,
        const Tensor&);
    friend ValueTriple bf16_qkv_projection(
        const Value&, const Value&, const Tensor&, const Value&,
        const Tensor&, const Value&, const Tensor&);
    friend Value sum(const Value&);
    friend Value mean(const Value&);
    friend Value reshape(const Value&, Shape);
    friend Value transpose(const Value&, std::int64_t, std::int64_t);
    friend Value embedding(const Value&, const Tensor&);
    friend Value softmax(const Value&, std::int64_t);
    friend Value rms_norm(const Value&, const Value&, float);
    friend std::pair<Value, Value> add_rms_norm(
        const Value&, const Value&, const Value&, float);
    friend Value silu(const Value&);
    friend Value swiglu(const Value&, const Value&);
    friend Value rope(const Value&, std::int64_t, std::int64_t, float);
    friend Value rope_split_half(const Value&, std::int64_t, std::int64_t, float);
    friend Value rope_split_half_bias(const Value&, const Value&, std::int64_t, float);
    friend Value rope_split_half_bias_bthd(const Value&, const Value&, std::int64_t,
                                           float);
    friend Value cross_entropy(const Value&, const Tensor&);
    friend MoeRouterResult moe_router_top_k(const Value&, std::int64_t, bool);
    friend Value moe_expert_ffn(const Value&, const Tensor&, const Value&,
                                const Value&, const Value&);
    friend Value moe_combine(const Value&, const Tensor&, const Value&);
    friend std::pair<Value, Value> moe_split_gate_up(const Value&);
    friend Value contiguous(const Value&);
    friend Value causal_softmax(const Value&);
    friend Value causal_gqa_attention(const Value&, const Value&, const Value&,
                                      std::int64_t, float);
    friend Value causal_gqa_attention_bthd(const Value&, const Value&, const Value&,
                                            std::int64_t, float);
    friend Value repeat_interleave(const Value&, std::int64_t, std::int64_t);
    friend GraphSnapshot inspect_graph(const Value&);
};

struct ValueTriple {
    Value first;
    Value second;
    Value third;
};

// Top-k selection is not differentiable: indices are a plain Tensor, never a
// Value, exactly like embedding()'s index argument.
struct MoeRouterResult {
    Tensor indices;
    Value weights;
};

[[nodiscard]] Value add(const Value& left, const Value& right);
[[nodiscard]] Value add_bias(const Value& input, const Value& bias);
[[nodiscard]] Value multiply(const Value& left, const Value& right);
[[nodiscard]] Value scale(const Value& input, float factor);
[[nodiscard]] Value matmul(const Value& left, const Value& right);
// Two-dimensional transpose-aware GEMM.  The flags change how each input is
// read; they do not allocate a transposed Tensor.
[[nodiscard]] Value matmul(const Value& left, const Value& right,
                           bool transpose_left, bool transpose_right);
// FP8 forward with FP32 master inputs and straight-through FP32 gradients.
[[nodiscard]] Value fp8_matmul(const Value& left, const Value& right,
                               float left_scale, float right_scale,
                               DType left_fp8_dtype = DType::Float8E4M3FNUZ,
                               DType right_fp8_dtype = DType::Float8E4M3FNUZ);
// BF16 rounded forward with FP32 master/straight-through gradients.
[[nodiscard]] Value bf16_matmul(const Value& left, const Value& right);
// Uses a caller-owned BF16 mirror for forward while gradients target the FP32 master.
[[nodiscard]] Value bf16_matmul(const Value& left, const Value& right_master,
                                const Tensor& right_bf16_mirror);
[[nodiscard]] std::pair<Value, Value> bf16_gate_up_projection(
    const Value& input, const Value& gate_master,
    const Tensor& gate_bf16_mirror, const Value& up_master,
    const Tensor& up_bf16_mirror);
[[nodiscard]] ValueTriple bf16_qkv_projection(
    const Value& input, const Value& query_master,
    const Tensor& query_bf16_mirror, const Value& key_master,
    const Tensor& key_bf16_mirror, const Value& value_master,
    const Tensor& value_bf16_mirror);
[[nodiscard]] Value sum(const Value& input);
[[nodiscard]] Value mean(const Value& input);
[[nodiscard]] Value reshape(const Value& input, Shape shape);
[[nodiscard]] Value transpose(const Value& input, std::int64_t dim0, std::int64_t dim1);
[[nodiscard]] Value embedding(const Value& weight, const Tensor& indices);
[[nodiscard]] Value softmax(const Value& input, std::int64_t dim = -1);
[[nodiscard]] Value rms_norm(const Value& input, const Value& weight,
                             float epsilon = 1.0e-5F);
[[nodiscard]] std::pair<Value, Value> add_rms_norm(
    const Value& left, const Value& right, const Value& weight,
    float epsilon = 1.0e-5F);
[[nodiscard]] Value silu(const Value& input);
[[nodiscard]] Value swiglu(const Value& gate, const Value& up);
[[nodiscard]] Value rope(const Value& input, std::int64_t sequence_dim = 1,
                         std::int64_t position_offset = 0, float base = 10000.0F);
[[nodiscard]] Value rope_split_half(const Value& input,
                                    std::int64_t sequence_dim = 1,
                                    std::int64_t position_offset = 0,
                                    float base = 10000.0F);
[[nodiscard]] Value rope_split_half_bias(const Value& input, const Value& bias,
                                         std::int64_t position_offset = 0,
                                         float base = 10000.0F);
[[nodiscard]] Value rope_split_half_bias_bthd(
    const Value& input, const Value& bias,
    std::int64_t position_offset = 0, float base = 10000.0F);
[[nodiscard]] Value cross_entropy(const Value& logits, const Tensor& targets);
// MoE routing. Contract: docs/OPERATOR_CONTRACTS.zh-CN.md, "MoE 路由". indices is
// always a plain (non-differentiable) Tensor, matching embedding()'s convention.
[[nodiscard]] MoeRouterResult moe_router_top_k(const Value& logits, std::int64_t k,
                                               bool norm_topk_prob);
[[nodiscard]] Value moe_expert_ffn(const Value& input, const Tensor& expert_indices,
                                   const Value& gate_weight, const Value& up_weight,
                                   const Value& down_weight);
[[nodiscard]] Value moe_combine(const Value& expert_output, const Tensor& expert_indices,
                                const Value& expert_weights);
// Checkpoint-format adapter (M6): splits Hugging Face Qwen3MoeExperts'
// fused gate_up_proj [num_experts, 2*ffn_dim, dim] into the two
// [num_experts, dim, ffn_dim] Values moe_expert_ffn expects. Not a routing
// primitive -- see ops.h's moe_split_gate_up for the exact layout contract.
[[nodiscard]] std::pair<Value, Value> moe_split_gate_up(const Value& gate_up_proj);
[[nodiscard]] Value contiguous(const Value& input);
[[nodiscard]] Value causal_softmax(const Value& scores);
[[nodiscard]] Value causal_gqa_attention(const Value& query, const Value& key,
                                         const Value& value,
                                         std::int64_t repeats, float scale);
[[nodiscard]] Value causal_gqa_attention_bthd(
    const Value& query, const Value& key, const Value& value,
    std::int64_t repeats, float scale);
[[nodiscard]] Value repeat_interleave(const Value& input, std::int64_t dim,
                                      std::int64_t repeats);
[[nodiscard]] GraphSnapshot inspect_graph(const Value& root);

}  // namespace microllm::autograd
