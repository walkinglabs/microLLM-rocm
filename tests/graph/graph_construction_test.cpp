#include <algorithm>
#include <string>
#include <vector>

#include <gtest/gtest.h>
#include <microllm/autograd/autograd.h>

namespace microllm::autograd {
namespace {

void expect_root(const Value& value, const std::string& operation,
                 std::size_t parent_count, const Shape& shape) {
    const auto graph = inspect_graph(value);
    ASSERT_LT(graph.root_id, graph.nodes.size());
    const auto& root = graph.nodes[graph.root_id];
    EXPECT_EQ(root.operation, operation);
    EXPECT_EQ(root.parents.size(), parent_count);
    EXPECT_EQ(root.shape, shape);
    for (const auto& node : graph.nodes) {
        EXPECT_EQ(node.id, static_cast<std::size_t>(&node - graph.nodes.data()));
        for (const auto parent : node.parents) EXPECT_LT(parent, node.id);
    }
}

}  // namespace

TEST(GraphConstructionTest, RecordsParentsSharedEdgesAndTopologicalOrder) {
    const Value a(Tensor::from_vector({1, 2}, {2}), true);
    const Value b(Tensor::from_vector({3, 4}, {2}), true);
    const auto product = multiply(a, b);
    const auto branched = add(product, a);
    const auto loss = sum(branched);
    const auto graph = inspect_graph(loss);

    ASSERT_EQ(graph.nodes.size(), 5U);
    EXPECT_EQ(graph.edge_count, 5U);
    EXPECT_EQ(graph.root_id, 4U);
    EXPECT_EQ(graph.nodes[0].operation, "leaf");
    EXPECT_EQ(graph.nodes[1].operation, "leaf");
    EXPECT_EQ(graph.nodes[2].operation, "multiply");
    EXPECT_EQ(graph.nodes[3].operation, "add");
    EXPECT_EQ(graph.nodes[4].operation, "sum");
    EXPECT_EQ(graph.nodes[3].parents, (std::vector<std::size_t>{2, 0}));
    for (const auto& node : graph.nodes) {
        for (const auto parent : node.parents) EXPECT_LT(parent, node.id);
    }
}

TEST(GraphConstructionTest, DropsParentEdgesWhenNothingRequiresGradient) {
    const Value left(Tensor::from_vector({1, 2}, {2}));
    const Value right(Tensor::from_vector({3, 4}, {2}));
    const auto result = add(left, right);
    const auto graph = inspect_graph(result);
    ASSERT_EQ(graph.nodes.size(), 1U);
    EXPECT_EQ(graph.edge_count, 0U);
    EXPECT_EQ(graph.nodes[0].operation, "add");
    EXPECT_FALSE(graph.nodes[0].requires_grad);
    EXPECT_TRUE(graph.nodes[0].parents.empty());
}

TEST(GraphConstructionTest, EveryPublicGraphOperationHasAVisibleRootContract) {
    const Value vector(Tensor::from_vector({-1, 0, 1, 2}, {2, 2}), true);
    const Value other(Tensor::from_vector({2, 3, 4, 5}, {2, 2}), true);
    expect_root(add(vector, other), "add", 2, {2, 2});
    expect_root(multiply(vector, other), "multiply", 2, {2, 2});
    expect_root(scale(vector, 0.5F), "scale", 1, {2, 2});
    expect_root(matmul(vector, other), "matmul", 2, {2, 2});
    const auto vector_mirror = vector.data().cast(DType::BFloat16);
    const auto other_mirror = other.data().cast(DType::BFloat16);
    expect_root(bf16_matmul(vector, other, other_mirror),
                "bf16_matmul", 2, {2, 2});
    const auto gate_up = bf16_gate_up_projection(
        vector, vector, vector_mirror, other, other_mirror);
    expect_root(gate_up.first, "bf16_gate_projection", 2, {2, 2});
    expect_root(gate_up.second, "bf16_up_projection", 2, {2, 2});
    const auto qkv = bf16_qkv_projection(
        vector, vector, vector_mirror, other, other_mirror,
        vector, vector_mirror);
    expect_root(qkv.first, "bf16_query_projection", 2, {2, 2});
    expect_root(qkv.second, "bf16_key_projection", 2, {2, 2});
    expect_root(qkv.third, "bf16_value_projection", 2, {2, 2});
    expect_root(sum(vector), "sum", 1, {});

    const auto mean_graph = inspect_graph(mean(vector));
    ASSERT_EQ(mean_graph.nodes.size(), 3U);
    EXPECT_EQ(mean_graph.nodes[1].operation, "sum");
    EXPECT_EQ(mean_graph.nodes[2].operation, "scale");

    expect_root(reshape(vector, {4}), "reshape", 1, {4});
    expect_root(transpose(vector, 0, 1), "transpose", 1, {2, 2});
    expect_root(softmax(vector), "softmax", 1, {2, 2});
    expect_root(silu(vector), "silu", 1, {2, 2});

    const Value norm_weight(Tensor::from_vector({1, 2}, {2}), true);
    expect_root(rms_norm(vector, norm_weight), "rms_norm", 2, {2, 2});
    const auto fused_norm = add_rms_norm(vector, other, norm_weight);
    expect_root(fused_norm.first, "add_rms_norm_sum", 2, {2, 2});
    expect_root(fused_norm.second, "add_rms_norm", 2, {2, 2});
    expect_root(swiglu(vector, other), "swiglu", 2, {2, 2});

    const Value rope_input(
        Tensor::from_vector({1, 0, 0, 1, 1, 0, 0, 1}, {1, 2, 1, 4}), true);
    expect_root(rope(rope_input), "rope", 1, {1, 2, 1, 4});

    const Value embedding_weight(
        Tensor::from_vector({0, 1, 2, 3, 4, 5}, {3, 2}), true);
    expect_root(embedding(embedding_weight, Tensor::from_int32_vector({2, 0}, {2})),
                "embedding", 1, {2, 2});

    const Value logits(Tensor::from_vector({2, 1, 0, 0, 1, 2}, {2, 3}), true);
    expect_root(cross_entropy(logits, Tensor::from_int32_vector({0, 2}, {2})),
                "cross_entropy", 1, {});

    const Value non_contiguous(
        Tensor::from_vector({0, 1, 2, 3, 4, 5}, {2, 3}).transpose(0, 1), true);
    expect_root(contiguous(non_contiguous), "contiguous", 1, {3, 2});

    const Value scores(
        Tensor::from_vector({1, 2, 3, 4, 5, 6, 7, 8, 9}, {1, 3, 3}), true);
    expect_root(causal_softmax(scores), "causal_softmax", 1, {1, 3, 3});
    expect_root(repeat_interleave(vector, 0, 2), "repeat_interleave", 1, {4, 2});
}

TEST(GraphConstructionTest, UndefinedRootCannotBeInspected) {
    EXPECT_THROW((void)inspect_graph(Value()), std::invalid_argument);
}

}  // namespace microllm::autograd
