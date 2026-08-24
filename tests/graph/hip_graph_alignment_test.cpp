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
}

}  // namespace microllm::autograd
