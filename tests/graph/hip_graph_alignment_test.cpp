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
}

}  // namespace microllm::autograd
