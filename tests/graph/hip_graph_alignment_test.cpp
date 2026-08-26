#include <string>
#include <utility>
#include <vector>

#include <gtest/gtest.h>
#include <microllm/model/model.h>
#include <microllm/runtime/runtime.h>

namespace microllm::autograd {
namespace {

void require_graph_gpu() {
    if (runtime::hip_device_count() == 0) GTEST_SKIP() << "No visible HIP device";
}

void expect_graph_near(const std::vector<float>& actual,
                       const std::vector<float>& expected, float tolerance) {
    ASSERT_EQ(actual.size(), expected.size());
    for (std::size_t index = 0; index < actual.size(); ++index) {
        EXPECT_NEAR(actual[index], expected[index], tolerance) << "index=" << index;
    }
}

}  // namespace

TEST(HipGraphAlignmentTest, FullTransformerForwardAndBackwardMatchCpuWithoutHostTransfers) {
    require_graph_gpu();
    const model::ModelConfig config{.vocabulary_size = 8,
                                    .dimension = 8,
                                    .layers = 1,
                                    .heads = 2,
                                    .kv_heads = 1,
                                    .ffn_dimension = 16,
                                    .max_sequence_length = 4,
                                    .rope_base = 10000.0F,
                                    .tie_embeddings = false};
    const auto tokens = Tensor::from_int32_vector({0, 1, 2, 3}, {1, 4});
    const auto targets = Tensor::from_int32_vector({1, 2, 3, 0}, {1, 4});

    model::TransformerModel cpu_model(config, 113);
    const auto cpu_loss = cpu_model.loss(tokens, targets);
    const auto cpu_graph = inspect_graph(cpu_loss);
    ASSERT_GT(cpu_graph.nodes.size(), 20U);
    EXPECT_EQ(cpu_graph.nodes[cpu_graph.root_id].operation, "cross_entropy");
    cpu_loss.backward();

    model::TransformerModel hip_model(config, 113);
    hip_model.to(Device::hip());
    const auto hip_tokens = tokens.to(Device::hip());
    const auto hip_targets = targets.to(Device::hip());
    runtime::reset_transfer_stats();
    const auto hip_loss = hip_model.loss(hip_tokens, hip_targets);
    const auto hip_graph = inspect_graph(hip_loss);
    EXPECT_EQ(hip_graph.nodes.size(), cpu_graph.nodes.size());
    EXPECT_EQ(hip_graph.edge_count, cpu_graph.edge_count);
    hip_loss.backward();
    runtime::synchronize(Device::hip());
    const auto transfers = runtime::transfer_stats();
    EXPECT_EQ(transfers.host_to_device_calls, 0U);
    EXPECT_EQ(transfers.device_to_host_calls, 0U);

    expect_graph_near(hip_loss.data().to_vector(), cpu_loss.data().to_vector(), 3.0e-4F);
    const auto cpu_parameters = cpu_model.named_parameters();
    const auto hip_parameters = hip_model.named_parameters();
    ASSERT_EQ(cpu_parameters.size(), hip_parameters.size());
    for (std::size_t index = 0; index < cpu_parameters.size(); ++index) {
        EXPECT_EQ(cpu_parameters[index].first, hip_parameters[index].first);
        ASSERT_TRUE(cpu_parameters[index].second->has_grad()) << cpu_parameters[index].first;
        ASSERT_TRUE(hip_parameters[index].second->has_grad()) << hip_parameters[index].first;
        expect_graph_near(hip_parameters[index].second->grad().to_vector(),
                          cpu_parameters[index].second->grad().to_vector(), 2.0e-3F);
    }
}

TEST(HipGraphAlignmentTest,
     AddRmsNormMatchesBranchedHipGraphAndStaysDeviceNative) {
    require_graph_gpu();
    const auto gpu = Device::hip(0);
    const auto left_data = Tensor::from_vector(
        {1, 2, 3, -1, -2, -3}, {2, 3}).to(gpu);
    const auto right_data = Tensor::from_vector(
        {0.5F, -0.5F, 1, 2, 1, 0}, {2, 3}).to(gpu);
    const auto weight_data = Tensor::from_vector({1, 0.5F, 2}, {3}).to(gpu);
    const Value sum_seed(Tensor::from_vector(
        {1, -1, 2, -2, 3, -3}, {2, 3}).to(gpu));
    const Value norm_seed(Tensor::from_vector(
        {0.5F, 2, -1, 1.5F, -0.5F, 3}, {2, 3}).to(gpu));

    Value baseline_left(left_data, true);
    Value baseline_right(right_data, true);
    Value baseline_weight(weight_data, true);
    const auto baseline_sum = add(baseline_left, baseline_right);
    const auto baseline_normalized = rms_norm(baseline_sum, baseline_weight);
    const auto baseline_loss = add(
        sum(multiply(baseline_sum, sum_seed)),
        sum(multiply(baseline_normalized, norm_seed)));
    baseline_loss.backward();
    runtime::synchronize(gpu);
    const auto expected_sum = baseline_sum.data().to_vector();
    const auto expected_normalized = baseline_normalized.data().to_vector();
    const auto expected_loss = baseline_loss.data().to_vector();
    const auto expected_left_gradient = baseline_left.grad().to_vector();
    const auto expected_right_gradient = baseline_right.grad().to_vector();
    const auto expected_weight_gradient = baseline_weight.grad().to_vector();

    Value fused_left(left_data, true);
    Value fused_right(right_data, true);
    Value fused_weight(weight_data, true);
    runtime::reset_transfer_stats();
    const auto fused = add_rms_norm(fused_left, fused_right, fused_weight);
    const auto fused_loss = add(
        sum(multiply(fused.first, sum_seed)),
        sum(multiply(fused.second, norm_seed)));
    fused_loss.backward();
    runtime::synchronize(gpu);
    const auto transfers = runtime::transfer_stats();

    EXPECT_EQ(transfers.host_to_device_calls, 0U);
    EXPECT_EQ(transfers.device_to_host_calls, 0U);
    expect_graph_near(fused_loss.data().to_vector(), expected_loss, 3.0e-5F);
    expect_graph_near(fused.first.data().to_vector(), expected_sum, 3.0e-5F);
    expect_graph_near(fused.second.data().to_vector(), expected_normalized, 3.0e-5F);
    expect_graph_near(fused_left.grad().to_vector(), expected_left_gradient, 3.0e-5F);
    expect_graph_near(fused_right.grad().to_vector(), expected_right_gradient, 3.0e-5F);
    expect_graph_near(fused_weight.grad().to_vector(), expected_weight_gradient, 3.0e-5F);
}

TEST(HipGraphAlignmentTest,
     SharedBf16ProjectionCastsMatchCpuAndStayDeviceNative) {
    require_graph_gpu();
    const auto input_data = Tensor::from_vector(
        {1, -2, 3, 0.5F, 2, -1}, {2, 3});
    const auto query_data = Tensor::from_vector(
        {1, 0.5F, -1, 2, 0.25F, -0.75F}, {3, 2});
    const auto gate_data = Tensor::from_vector(
        {0.5F, -1, 2, 0.25F, 1.5F, -0.5F}, {3, 2});
    const auto up_data = Tensor::from_vector(
        {-0.5F, 1.5F, 2, -1, 0.75F, 0.25F}, {3, 2});
    const auto key_data = Tensor::from_vector({1, -0.5F, 2}, {3, 1});
    const auto value_data = Tensor::from_vector({-1, 0.25F, 0.5F}, {3, 1});
    const Value pair_seed(Tensor::from_vector({1, -2, 0.5F, 3}, {2, 2}));
    const Value key_seed(Tensor::from_vector({1.5F, -2}, {2, 1}));
    const Value value_seed(Tensor::from_vector({-0.25F, 3}, {2, 1}));

    Value cpu_input(input_data, true);
    Value cpu_query(query_data, true);
    Value cpu_gate(gate_data, true);
    Value cpu_up(up_data, true);
    Value cpu_key(key_data, true);
    Value cpu_value(value_data, true);
    const auto cpu_pair = bf16_gate_up_projection(
        cpu_input, cpu_gate, gate_data.cast(DType::BFloat16),
        cpu_up, up_data.cast(DType::BFloat16));
    const auto cpu_qkv = bf16_qkv_projection(
        cpu_input, cpu_query, query_data.cast(DType::BFloat16),
        cpu_key, key_data.cast(DType::BFloat16), cpu_value,
        value_data.cast(DType::BFloat16));
    const auto cpu_loss = add(
        add(sum(multiply(cpu_pair.first, pair_seed)),
            sum(multiply(cpu_pair.second, pair_seed))),
        add(add(sum(multiply(cpu_qkv.first, pair_seed)),
                sum(multiply(cpu_qkv.second, key_seed))),
            sum(multiply(cpu_qkv.third, value_seed))));
    cpu_loss.backward();

    const auto gpu = Device::hip(0);
    Value hip_input(input_data.to(gpu), true);
    Value hip_query(query_data.to(gpu), true);
    Value hip_gate(gate_data.to(gpu), true);
    Value hip_up(up_data.to(gpu), true);
    Value hip_key(key_data.to(gpu), true);
    Value hip_value(value_data.to(gpu), true);
    const Value hip_pair_seed(pair_seed.data().to(gpu));
    const Value hip_key_seed(key_seed.data().to(gpu));
    const Value hip_value_seed(value_seed.data().to(gpu));
    const auto hip_query_mirror = hip_query.data().cast(DType::BFloat16);
    const auto hip_gate_mirror = hip_gate.data().cast(DType::BFloat16);
    const auto hip_up_mirror = hip_up.data().cast(DType::BFloat16);
    const auto hip_key_mirror = hip_key.data().cast(DType::BFloat16);
    const auto hip_value_mirror = hip_value.data().cast(DType::BFloat16);
    runtime::reset_transfer_stats();
    const auto hip_pair = bf16_gate_up_projection(
        hip_input, hip_gate, hip_gate_mirror, hip_up, hip_up_mirror);
    const auto hip_qkv = bf16_qkv_projection(
        hip_input, hip_query, hip_query_mirror, hip_key, hip_key_mirror,
        hip_value, hip_value_mirror);
    const auto hip_loss = add(
        add(sum(multiply(hip_pair.first, hip_pair_seed)),
            sum(multiply(hip_pair.second, hip_pair_seed))),
        add(add(sum(multiply(hip_qkv.first, hip_pair_seed)),
                sum(multiply(hip_qkv.second, hip_key_seed))),
            sum(multiply(hip_qkv.third, hip_value_seed))));
    hip_loss.backward();
    runtime::synchronize(gpu);
    const auto transfers = runtime::transfer_stats();

    EXPECT_EQ(transfers.host_to_device_calls, 0U);
    EXPECT_EQ(transfers.device_to_host_calls, 0U);
    expect_graph_near(hip_loss.data().to_vector(), cpu_loss.data().to_vector(),
                      3.0e-4F);
    for (const auto& [hip_leaf, cpu_leaf] : std::vector<std::pair<Value*, Value*>>{
             {&hip_input, &cpu_input}, {&hip_query, &cpu_query},
             {&hip_gate, &cpu_gate}, {&hip_up, &cpu_up},
             {&hip_key, &cpu_key}, {&hip_value, &cpu_value}}) {
        expect_graph_near(hip_leaf->grad().to_vector(),
                          cpu_leaf->grad().to_vector(), 2.0e-3F);
    }
}

TEST(HipGraphAlignmentTest, DeferredScopedStreamRestoresCompleteInferenceLogits) {
    require_graph_gpu();
    const model::ModelConfig config{.vocabulary_size = 16,
                                    .dimension = 16,
                                    .layers = 1,
                                    .heads = 4,
                                    .kv_heads = 2,
                                    .ffn_dimension = 32,
                                    .max_sequence_length = 4,
                                    .rope_base = 10000.0F,
                                    .tie_embeddings = false};
    const auto gpu = Device::hip(0);
    const auto tokens = Tensor::from_int32_vector({1, 2, 3, 4}, {1, 4}).to(gpu);
    model::TransformerModel legacy(config, 173);
    legacy.to(gpu);
    const auto expected = legacy.forward_inference(tokens);
    runtime::synchronize(gpu);
    const auto expected_values = expected.to_vector();

    model::TransformerModel candidate(config, 173);
    candidate.to(gpu);
    for (int repetition = 0; repetition < 3; ++repetition) {
        runtime::Stream stream(gpu);
        Tensor actual;
        std::size_t deferred_blocks = 0;
        std::size_t deferred_bytes = 0;
        runtime::reset_transfer_stats();
        {
            runtime::ScopedDeferredHipStream scope(stream);
            actual = candidate.forward_inference(tokens);
            deferred_blocks = scope.pending_blocks();
            deferred_bytes = scope.pending_bytes();
            scope.finish();
        }
        const auto transfers = runtime::transfer_stats();
        EXPECT_EQ(transfers.host_to_device_calls, 0U);
        EXPECT_EQ(transfers.device_to_host_calls, 0U);
        EXPECT_EQ(transfers.device_to_device_calls, 0U);
        EXPECT_GT(deferred_blocks, 20U);
        EXPECT_GT(deferred_bytes, 0U);
        expect_graph_near(actual.to_vector(), expected_values, 1.0e-5F);
    }
}

TEST(HipGraphAlignmentTest, DeferredScopedStreamMatchesForwardBackwardGradients) {
    require_graph_gpu();
    const model::ModelConfig config{.vocabulary_size = 8,
                                    .dimension = 8,
                                    .layers = 1,
                                    .heads = 2,
                                    .kv_heads = 1,
                                    .ffn_dimension = 16,
                                    .max_sequence_length = 4,
                                    .rope_base = 10000.0F,
                                    .tie_embeddings = false};
    const auto gpu = Device::hip(0);
    const auto tokens = Tensor::from_int32_vector({0, 1, 2, 3}, {1, 4}).to(gpu);
    const auto targets = Tensor::from_int32_vector({1, 2, 3, 0}, {1, 4}).to(gpu);
    std::vector<std::pair<std::string, std::vector<float>>> expected_gradients;
    std::vector<float> expected_loss;
    {
        model::TransformerModel legacy(config, 179);
        legacy.to(gpu);
        auto loss = legacy.loss(tokens, targets);
        loss.backward();
        runtime::synchronize(gpu);
        expected_loss = loss.data().to_vector();
        for (const auto& [name, parameter] : legacy.named_parameters()) {
            ASSERT_TRUE(parameter->has_grad()) << name;
            expected_gradients.emplace_back(name, parameter->grad().to_vector());
        }
    }

    model::TransformerModel candidate(config, 179);
    candidate.to(gpu);
    runtime::Stream stream(gpu);
    Tensor candidate_loss;
    std::size_t deferred_blocks = 0;
    std::size_t deferred_bytes = 0;
    runtime::reset_transfer_stats();
    {
        runtime::ScopedDeferredHipStream scope(stream);
        {
            auto loss = candidate.loss(tokens, targets);
            loss.backward();
            candidate_loss = loss.data();
        }
        deferred_blocks = scope.pending_blocks();
        deferred_bytes = scope.pending_bytes();
        scope.finish();
    }
    const auto transfers = runtime::transfer_stats();
    EXPECT_EQ(transfers.host_to_device_calls, 0U);
    EXPECT_EQ(transfers.device_to_host_calls, 0U);
    EXPECT_EQ(transfers.device_to_device_calls, 0U);
    EXPECT_GT(deferred_blocks, 20U);
    EXPECT_GT(deferred_bytes, 0U);
    expect_graph_near(candidate_loss.to_vector(), expected_loss, 1.0e-5F);
    const auto parameters = candidate.named_parameters();
    ASSERT_EQ(parameters.size(), expected_gradients.size());
    for (std::size_t index = 0; index < parameters.size(); ++index) {
        EXPECT_EQ(parameters[index].first, expected_gradients[index].first);
        ASSERT_TRUE(parameters[index].second->has_grad()) << parameters[index].first;
        expect_graph_near(parameters[index].second->grad().to_vector(),
                          expected_gradients[index].second, 1.0e-5F);
    }
}

TEST(HipGraphAlignmentTest, Bf16LinearInferenceMatchesCpuAndStaysDeviceNative) {
    require_graph_gpu();
    const model::ModelConfig config{.vocabulary_size = 16,
                                    .dimension = 128,
                                    .layers = 1,
                                    .heads = 4,
                                    .kv_heads = 2,
                                    .ffn_dimension = 256,
                                    .max_sequence_length = 4,
                                    .rope_base = 10000.0F,
                                    .tie_embeddings = false};
    const auto tokens = Tensor::from_int32_vector({1, 2, 3, 4}, {1, 4});
    model::TransformerModel cpu(config, 127);
    const auto cpu_report = cpu.prepare_bf16_ffn_inference();
    ASSERT_EQ(cpu_report.converted_tensors, 3U);
    ASSERT_EQ(cpu.prepare_bf16_attention_inference().converted_tensors, 4U);
    const auto expected = cpu.forward_inference(tokens).to_vector();

    model::TransformerModel hip(config, 127);
    hip.to(Device::hip(0));
    const auto hip_report = hip.prepare_bf16_ffn_inference();
    EXPECT_EQ(hip_report.fp32_bytes_released, cpu_report.fp32_bytes_released);
    EXPECT_EQ(hip.prepare_bf16_attention_inference().converted_tensors, 4U);
    const auto device_tokens = tokens.to(Device::hip(0));
    runtime::reset_transfer_stats();
    const auto actual = hip.forward_inference(device_tokens);
    runtime::synchronize(Device::hip(0));
    const auto transfers = runtime::transfer_stats();
    EXPECT_EQ(transfers.host_to_device_calls, 0U);
    EXPECT_EQ(transfers.device_to_host_calls, 0U);
    EXPECT_EQ(actual.device(), Device::hip(0));
    expect_graph_near(actual.to_vector(), expected, 5.0e-2F);
    hip.set_bf16_ffn_arena_enabled(true, 5);
    const auto bypassed = hip.forward_inference(device_tokens);
    runtime::synchronize(Device::hip(0));
    EXPECT_EQ(bypassed.to_vector(), actual.to_vector());
    EXPECT_EQ(hip.bf16_ffn_arena_stats().entries, 0U);
    EXPECT_EQ(hip.bf16_ffn_arena_stats().bypassed_calls, 1U);
    hip.set_bf16_ffn_arena_enabled(true);
    runtime::reset_transfer_stats();
    const auto arena_first = hip.forward_inference(device_tokens);
    const auto arena_second = hip.forward_inference(device_tokens);
    runtime::synchronize(Device::hip(0));
    const auto arena_transfers = runtime::transfer_stats();
    EXPECT_EQ(arena_transfers.host_to_device_calls, 0U);
    EXPECT_EQ(arena_transfers.device_to_host_calls, 0U);
    EXPECT_EQ(arena_first.to_vector(), actual.to_vector());
    EXPECT_EQ(arena_second.to_vector(), actual.to_vector());
    const auto arena_stats = hip.bf16_ffn_arena_stats();
    EXPECT_EQ(arena_stats.entries, 1U);
    EXPECT_EQ(arena_stats.misses, 1U);
    EXPECT_EQ(arena_stats.hits, 1U);
    EXPECT_GT(arena_stats.capacity_bytes, 0U);
    hip.set_bf16_qkv_arena_enabled(true, 5);
    const auto qkv_bypassed = hip.forward_inference(device_tokens);
    runtime::synchronize(Device::hip(0));
    EXPECT_EQ(qkv_bypassed.to_vector(), actual.to_vector());
    EXPECT_EQ(hip.bf16_qkv_arena_stats().entries, 0U);
    EXPECT_EQ(hip.bf16_qkv_arena_stats().bypassed_calls, 1U);
    hip.set_bf16_qkv_arena_enabled(true, 1);
    runtime::reset_transfer_stats();
    const auto qkv_first = hip.forward_inference(device_tokens);
    const auto qkv_second = hip.forward_inference(device_tokens);
    runtime::synchronize(Device::hip(0));
    const auto qkv_transfers = runtime::transfer_stats();
    EXPECT_EQ(qkv_transfers.host_to_device_calls, 0U);
    EXPECT_EQ(qkv_transfers.device_to_host_calls, 0U);
    EXPECT_EQ(qkv_first.to_vector(), actual.to_vector());
    EXPECT_EQ(qkv_second.to_vector(), actual.to_vector());
    const auto qkv_stats = hip.bf16_qkv_arena_stats();
    EXPECT_EQ(qkv_stats.entries, 1U);
    EXPECT_EQ(qkv_stats.misses, 1U);
    EXPECT_EQ(qkv_stats.hits, 1U);
    EXPECT_GT(qkv_stats.capacity_bytes, 0U);
    hip.set_attention_core_arena_enabled(true, 5);
    const auto core_bypassed = hip.forward_inference(device_tokens);
    runtime::synchronize(Device::hip(0));
    EXPECT_EQ(core_bypassed.to_vector(), actual.to_vector());
    EXPECT_EQ(hip.attention_core_arena_stats().entries, 0U);
    EXPECT_EQ(hip.attention_core_arena_stats().bypassed_calls, 1U);
    hip.set_attention_core_arena_enabled(true, 1);
    runtime::reset_transfer_stats();
    const auto core_first = hip.forward_inference(device_tokens);
    const auto core_second = hip.forward_inference(device_tokens);
    runtime::synchronize(Device::hip(0));
    const auto core_transfers = runtime::transfer_stats();
    EXPECT_EQ(core_transfers.host_to_device_calls, 0U);
    EXPECT_EQ(core_transfers.device_to_host_calls, 0U);
    EXPECT_EQ(core_first.to_vector(), actual.to_vector());
    EXPECT_EQ(core_second.to_vector(), actual.to_vector());
    const auto core_stats = hip.attention_core_arena_stats();
    EXPECT_EQ(core_stats.entries, 1U);
    EXPECT_EQ(core_stats.misses, 1U);
    EXPECT_EQ(core_stats.hits, 1U);
    EXPECT_GT(core_stats.capacity_bytes, 0U);
}

TEST(HipGraphAlignmentTest,
     ExplicitGradientBufferPreservesHipAddressAndMatchesCpu) {
    require_graph_gpu();
    const auto gpu = Device::hip(0);
    const auto input_data = Tensor::from_vector({1, 2, 3}, {3});

    Value cpu_input(input_data, true);
    sum(add(scale(cpu_input, 2.0F), scale(cpu_input, 3.0F))).backward();

    Tensor owner({3}, DType::Float32, gpu);
    const auto expected_address = owner.storage().data();
    auto external_storage = Storage::from_external(
        expected_address, owner.storage().num_bytes(), gpu);
    auto external_buffer = Tensor::from_storage(
        external_storage, {3}, {1}, 0, DType::Float32);
    Value hip_input(input_data.to(gpu), true);
    hip_input.bind_grad_buffer(external_buffer);

    runtime::reset_transfer_stats();
    const auto loss = sum(add(scale(hip_input, 2.0F),
                              scale(hip_input, 3.0F)));
    loss.backward();
    runtime::synchronize(gpu);
    const auto transfers = runtime::transfer_stats();
    EXPECT_EQ(transfers.host_to_device_calls, 0U);
    EXPECT_EQ(transfers.device_to_host_calls, 0U);
    ASSERT_TRUE(hip_input.grad_buffer_bound());
    EXPECT_EQ(hip_input.grad().storage().data(), expected_address);
    expect_graph_near(hip_input.grad().to_vector(),
                      cpu_input.grad().to_vector(), 1.0e-6F);

    hip_input.zero_grad();
    runtime::synchronize(gpu);
    EXPECT_EQ(hip_input.grad().storage().data(), expected_address);
    EXPECT_EQ(hip_input.grad().to_vector(), (std::vector<float>{0, 0, 0}));
    loss.backward();
    runtime::synchronize(gpu);
    EXPECT_EQ(hip_input.grad().storage().data(), expected_address);
    expect_graph_near(hip_input.grad().to_vector(),
                      cpu_input.grad().to_vector(), 1.0e-6F);
}

TEST(HipGraphAlignmentTest,
     Int8PreparedSingleTokenModelMatchesCpuAndStaysDeviceNative) {
    require_graph_gpu();
    const model::ModelConfig config{.vocabulary_size = 16,
                                    .dimension = 8,
                                    .layers = 1,
                                    .heads = 2,
                                    .kv_heads = 1,
                                    .ffn_dimension = 16,
                                    .max_sequence_length = 4,
                                    .rope_base = 10000.0F,
                                    .tie_embeddings = false};
    const auto tokens = Tensor::from_int32_vector({3}, {1, 1});
    model::TransformerModel cpu(config, 991);
    const auto fp32 = cpu.forward_inference(tokens).to_vector();
    const auto cpu_report = cpu.prepare_int8_inference_weights();
    EXPECT_EQ(cpu_report.linears_covered, 8U);
    const auto expected = cpu.forward_inference(tokens).to_vector();

    model::TransformerModel hip(config, 991);
    hip.to(Device::hip(0));
    runtime::reset_transfer_stats();
    const auto hip_report = hip.prepare_int8_inference_weights();
    runtime::synchronize(Device::hip(0));
    const auto preparation_transfers = runtime::transfer_stats();
    EXPECT_EQ(preparation_transfers.host_to_device_calls, 0U);
    EXPECT_EQ(preparation_transfers.device_to_host_calls, 0U);
    EXPECT_EQ(hip_report.device_amax_tensors, 8U);
    EXPECT_GT(hip_report.device_weight_bytes_scanned, 0U);
    EXPECT_EQ(hip_report.int8_bytes_retained,
              cpu_report.int8_bytes_retained);
    const auto device_tokens = tokens.to(Device::hip(0));
    runtime::reset_transfer_stats();
    const auto actual = hip.forward_inference(device_tokens);
    runtime::synchronize(Device::hip(0));
    const auto transfers = runtime::transfer_stats();
    EXPECT_EQ(transfers.host_to_device_calls, 0U);
    EXPECT_EQ(transfers.device_to_host_calls, 0U);
    expect_graph_near(actual.to_vector(), expected, 2.0e-3F);
    expect_graph_near(actual.to_vector(), fp32, 0.12F);
}

}  // namespace microllm::autograd
