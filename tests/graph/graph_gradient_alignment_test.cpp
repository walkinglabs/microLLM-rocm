#include <vector>

#include <gtest/gtest.h>
#include <microllm/autograd/autograd.h>

namespace microllm::autograd {
namespace {

void expect_near(const Tensor& tensor, const std::vector<float>& expected,
                 float tolerance = 1.0e-6F) {
    const auto actual = tensor.to_vector();
    ASSERT_EQ(actual.size(), expected.size());
    for (std::size_t index = 0; index < actual.size(); ++index) {
        EXPECT_NEAR(actual[index], expected[index], tolerance) << "index=" << index;
    }
}

}  // namespace

TEST(GraphGradientAlignmentTest, CompositeForwardAndBothLeafGradientsMatchHandOracle) {
    Value a(Tensor::from_vector({1, 2, 3, 4}, {2, 2}), true);
    Value b(Tensor::from_vector({5, 6, 7, 8}, {2, 2}), true);
    const auto output = add(multiply(a, b), scale(a, 2.0F));
    expect_near(output.data(), {7, 16, 27, 40});
    const auto loss = mean(output);
    expect_near(loss.data(), {22.5F});
    loss.backward();
    expect_near(a.grad(), {1.75F, 2.0F, 2.25F, 2.5F});
    expect_near(b.grad(), {0.25F, 0.5F, 0.75F, 1.0F});
}

TEST(GraphGradientAlignmentTest, SharedLeafAccumulatesAndRepeatedBackwardIsStable) {
    Value input(Tensor::from_vector({1, 2, 3}, {3}), true);
    const auto loss = sum(add(multiply(input, input), input));
    loss.backward();
    expect_near(input.grad(), {3, 5, 7});
    loss.backward();
    expect_near(input.grad(), {6, 10, 14});
}

TEST(GraphGradientAlignmentTest,
     AddRmsNormMatchesComposedForwardAndAllBranchedGradients) {
    const auto left_data = Tensor::from_vector(
        {1, 2, 3, -1, -2, -3}, {2, 3});
    const auto right_data = Tensor::from_vector(
        {0.5F, -0.5F, 1, 2, 1, 0}, {2, 3});
    const auto weight_data = Tensor::from_vector({1, 0.5F, 2}, {3});
    const Value sum_seed(Tensor::from_vector(
        {1, -1, 2, -2, 3, -3}, {2, 3}));
    const Value norm_seed(Tensor::from_vector(
        {0.5F, 2, -1, 1.5F, -0.5F, 3}, {2, 3}));

    Value fused_left(left_data, true);
    Value fused_right(right_data, true);
    Value fused_weight(weight_data, true);
    const auto fused = add_rms_norm(
        fused_left, fused_right, fused_weight);
    const auto fused_loss = add(
        sum(multiply(fused.first, sum_seed)),
        sum(multiply(fused.second, norm_seed)));
    fused_loss.backward();

    Value reference_left(left_data, true);
    Value reference_right(right_data, true);
    Value reference_weight(weight_data, true);
    const auto reference_sum = add(reference_left, reference_right);
    const auto reference_norm = rms_norm(
        reference_sum, reference_weight);
    const auto reference_loss = add(
        sum(multiply(reference_sum, sum_seed)),
        sum(multiply(reference_norm, norm_seed)));
    reference_loss.backward();

    expect_near(fused.first.data(), reference_sum.data().to_vector());
    expect_near(fused.second.data(), reference_norm.data().to_vector());
    expect_near(fused_left.grad(), reference_left.grad().to_vector());
    expect_near(fused_right.grad(), reference_right.grad().to_vector());
    expect_near(fused_weight.grad(), reference_weight.grad().to_vector());
    const auto graph = inspect_graph(fused.second);
    EXPECT_EQ(graph.nodes[graph.root_id].operation, "add_rms_norm");
}

TEST(GraphGradientAlignmentTest,
     SharedBf16ProjectionCastsMatchComposedOutputsAndAllGradients) {
    const auto input_data = Tensor::from_vector(
        {1, -2, 3, 0.5F, 2, -1}, {2, 3});
    const auto gate_data = Tensor::from_vector(
        {1, 0.5F, -1, 2, 0.25F, -0.75F}, {3, 2});
    const auto up_data = Tensor::from_vector(
        {-0.5F, 1.5F, 2, -1, 0.75F, 0.25F}, {3, 2});
    const auto gate_mirror = gate_data.cast(DType::BFloat16);
    const auto up_mirror = up_data.cast(DType::BFloat16);
    const Value gate_seed(Tensor::from_vector({1, -2, 0.5F, 3}, {2, 2}));
    const Value up_seed(Tensor::from_vector({-1, 0.25F, 2, -0.5F}, {2, 2}));

    Value fused_input(input_data, true);
    Value fused_gate(gate_data, true);
    Value fused_up(up_data, true);
    const auto fused = bf16_gate_up_projection(
        fused_input, fused_gate, gate_mirror, fused_up, up_mirror);
    add(sum(multiply(fused.first, gate_seed)),
        sum(multiply(fused.second, up_seed))).backward();

    Value reference_input(input_data, true);
    Value reference_gate(gate_data, true);
    Value reference_up(up_data, true);
    const auto reference_gate_output = bf16_matmul(
        reference_input, reference_gate, gate_mirror);
    const auto reference_up_output = bf16_matmul(
        reference_input, reference_up, up_mirror);
    add(sum(multiply(reference_gate_output, gate_seed)),
        sum(multiply(reference_up_output, up_seed))).backward();

    expect_near(fused.first.data(), reference_gate_output.data().to_vector());
    expect_near(fused.second.data(), reference_up_output.data().to_vector());
    expect_near(fused_input.grad(), reference_input.grad().to_vector(), 2.0e-5F);
    expect_near(fused_gate.grad(), reference_gate.grad().to_vector(), 2.0e-5F);
    expect_near(fused_up.grad(), reference_up.grad().to_vector(), 2.0e-5F);

    const auto key_data = Tensor::from_vector({1, -0.5F, 2}, {3, 1});
    const auto value_data = Tensor::from_vector({-1, 0.25F, 0.5F}, {3, 1});
    const auto key_mirror = key_data.cast(DType::BFloat16);
    const auto value_mirror = value_data.cast(DType::BFloat16);
    Value triple_input(input_data, true);
    Value triple_query(gate_data, true);
    Value triple_key(key_data, true);
    Value triple_value(value_data, true);
    const auto triple = bf16_qkv_projection(
        triple_input, triple_query, gate_mirror, triple_key, key_mirror,
        triple_value, value_mirror);
    const Value key_seed(Tensor::from_vector({1.5F, -2}, {2, 1}));
    const Value value_seed(Tensor::from_vector({-0.25F, 3}, {2, 1}));
    add(add(sum(multiply(triple.first, gate_seed)),
            sum(multiply(triple.second, key_seed))),
        sum(multiply(triple.third, value_seed))).backward();

    Value triple_reference_input(input_data, true);
    Value triple_reference_query(gate_data, true);
    Value triple_reference_key(key_data, true);
    Value triple_reference_value(value_data, true);
    const auto query_output = bf16_matmul(
        triple_reference_input, triple_reference_query, gate_mirror);
    const auto key_output = bf16_matmul(
        triple_reference_input, triple_reference_key, key_mirror);
    const auto value_output = bf16_matmul(
        triple_reference_input, triple_reference_value, value_mirror);
    add(add(sum(multiply(query_output, gate_seed)),
            sum(multiply(key_output, key_seed))),
        sum(multiply(value_output, value_seed))).backward();
    expect_near(triple.first.data(), query_output.data().to_vector());
    expect_near(triple.second.data(), key_output.data().to_vector());
    expect_near(triple.third.data(), value_output.data().to_vector());
    expect_near(triple_input.grad(), triple_reference_input.grad().to_vector(),
                2.0e-5F);
    expect_near(triple_query.grad(), triple_reference_query.grad().to_vector(),
                2.0e-5F);
    expect_near(triple_key.grad(), triple_reference_key.grad().to_vector(),
                2.0e-5F);
    expect_near(triple_value.grad(), triple_reference_value.grad().to_vector(),
                2.0e-5F);
}

TEST(GraphGradientAlignmentTest, ViewGraphRestoresLogicalGradientOrder) {
    Value input(Tensor::from_vector({0, 1, 2, 3, 4, 5}, {2, 3}), true);
    const auto view = transpose(input, 0, 1);
    const auto packed = contiguous(view);
    const Value weights(Tensor::from_vector({1, 2, 3, 4, 5, 6}, {3, 2}));
    sum(multiply(packed, weights)).backward();
    expect_near(input.grad(), {1, 3, 5, 2, 4, 6});
}

TEST(GraphGradientAlignmentTest, TiedHeadMatchesHandForwardAndBothGradientsWithoutTransposeNode) {
    Value hidden(Tensor::from_vector({1, 2, 3, 4, 5, 6}, {2, 3}), true);
    Value weight(Tensor::from_vector(
        {1, 0, -1, 2, 1, 0, -2, 0.5F, 1, 3, -1, 2}, {4, 3}), true);
    const Value seed(Tensor::from_vector(
        {1, -1, 0.5F, 2, -2, 1, 3, -0.5F}, {2, 4}));
    const auto logits = matmul(hidden, weight, false, true);
    expect_near(logits.data(), {-2, 4, 2, 7, -2, 13, 0.5F, 19});
    const auto graph = inspect_graph(logits);
    for (const auto& node : graph.nodes) {
        EXPECT_NE(node.operation, "transpose");
        EXPECT_NE(node.operation, "contiguous");
    }
    sum(multiply(logits, seed)).backward();
    expect_near(hidden.grad(), {4, -2.75F, 3.5F, -7.5F, 3, 4});
    expect_near(weight.grad(), {-7, -8, -9, 3, 3, 3,
                                12.5F, 16, 19.5F, 0, 1.5F, 3});
}

TEST(GraphGradientAlignmentTest, BthdBiasRopeMatchesComposedGraphWithoutLayoutNodes) {
    const auto input_data = Tensor::from_vector(
        {1, 2, 3, 4, 5, 6, 7, 8,
         -1, -2, -3, -4, -5, -6, -7, -8}, {1, 2, 2, 4});
    const auto bias_data = Tensor::from_vector(
        {0.1F, 0.2F, 0.3F, 0.4F, -0.1F, -0.2F, -0.3F, -0.4F}, {8});
    const Value seed(Tensor::from_vector(
        {1, -1, 2, -2, 3, -3, 4, -4,
         -1, 1, -2, 2, -3, 3, -4, 4}, {1, 2, 2, 4}));

    Value fused_input(input_data, true);
    Value fused_bias(bias_data, true);
    const auto fused = rope_split_half_bias_bthd(fused_input, fused_bias);
    const auto snapshot = inspect_graph(fused);
    ASSERT_EQ(snapshot.nodes.size(), 3U);
    EXPECT_EQ(snapshot.nodes[snapshot.root_id].operation,
              "rope_split_half_bias_bthd");
    for (const auto& node : snapshot.nodes) {
        EXPECT_NE(node.operation, "transpose");
        EXPECT_NE(node.operation, "contiguous");
    }
    sum(multiply(fused, seed)).backward();

    Value reference_input(input_data, true);
    Value reference_bias(bias_data, true);
    const auto arranged = transpose(
        reshape(add_bias(reshape(reference_input, {2, 8}), reference_bias),
                {1, 2, 2, 4}),
        1, 2);
    const auto reference = rope_split_half(arranged, 2);
    sum(multiply(reference, seed)).backward();

    expect_near(fused.data(), reference.data().to_vector(), 3.0e-5F);
    expect_near(fused_input.grad(), reference_input.grad().to_vector(), 3.0e-5F);
    expect_near(fused_bias.grad(), reference_bias.grad().to_vector(), 3.0e-5F);
}

TEST(GraphGradientAlignmentTest, BthdCausalGqaMatchesComposedGraphAndAllGradients) {
    const auto query_data = Tensor::from_vector(
        {0.5F, -1, 1.5F, 0.25F, -0.5F, 1,
         0.75F, -0.25F, 1, 0.5F, -1, 0.25F}, {1, 2, 3, 2});
    const auto key_data = Tensor::from_vector(
        {0.5F, 1, -0.5F, 0.25F, 1.5F, -1}, {1, 1, 3, 2});
    const auto value_data = Tensor::from_vector(
        {1, 2, 3, 4, 5, 6}, {1, 3, 1, 2});
    const Value seed(Tensor::from_vector(
        {1, -1, 0.5F, -0.5F, 2, -2,
         1.5F, -1.5F, 3, -3, 2.5F, -2.5F}, {1, 3, 2, 2}));

    Value fused_query(query_data, true);
    Value fused_key(key_data, true);
    Value fused_value(value_data, true);
    const auto fused = causal_gqa_attention_bthd(
        fused_query, fused_key, fused_value, 2, 0.5F);
    const auto snapshot = inspect_graph(fused);
    ASSERT_EQ(snapshot.nodes.size(), 4U);
    for (const auto& node : snapshot.nodes) {
        EXPECT_NE(node.operation, "transpose");
        EXPECT_NE(node.operation, "contiguous");
    }
    sum(multiply(fused, seed)).backward();

    Value reference_query(query_data, true);
    Value reference_key(key_data, true);
    Value reference_value(value_data, true);
    const auto reference = contiguous(transpose(causal_gqa_attention(
        reference_query, reference_key,
        contiguous(transpose(reference_value, 1, 2)), 2, 0.5F), 1, 2));
    sum(multiply(reference, seed)).backward();

    expect_near(fused.data(), reference.data().to_vector(), 3.0e-5F);
    expect_near(fused_query.grad(), reference_query.grad().to_vector(), 3.0e-5F);
    expect_near(fused_key.grad(), reference_key.grad().to_vector(), 3.0e-5F);
    expect_near(fused_value.grad(), reference_value.grad().to_vector(), 3.0e-5F);
}

TEST(GraphGradientAlignmentTest, RejectsBadBackwardSeedShape) {
    Value input(Tensor::from_vector({1, 2, 3}, {3}), true);
    const auto output = multiply(input, input);
    EXPECT_THROW(output.backward(), std::invalid_argument);
    EXPECT_THROW(output.backward(Tensor::from_vector({1, 2}, {2})), std::invalid_argument);
}

}  // namespace microllm::autograd
