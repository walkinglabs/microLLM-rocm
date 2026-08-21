#include <cstdint>
#include <vector>

#include <gtest/gtest.h>
#include <microllm/inference/generator.h>
#include <microllm/model/model.h>
#include <microllm/runtime/runtime.h>

namespace microllm::inference {
namespace {

model::ModelConfig hip_shape_matrix_config() {
    return {.vocabulary_size = 32,
            .dimension = 16,
            .layers = 2,
            .heads = 4,
            .kv_heads = 2,
            .ffn_dimension = 32,
            .max_sequence_length = 132,
            .rope_base = 10000.0F,
            .tie_embeddings = false};
}

std::vector<std::vector<std::int32_t>> hip_prompts(std::int64_t context,
                                                    std::int64_t batch) {
    std::vector<std::vector<std::int32_t>> result(
        static_cast<std::size_t>(batch),
        std::vector<std::int32_t>(static_cast<std::size_t>(context)));
    for (std::int64_t row = 0; row < batch; ++row) {
        for (std::int64_t token = 0; token < context; ++token) {
            result[static_cast<std::size_t>(row)][static_cast<std::size_t>(token)] =
                static_cast<std::int32_t>((row * 13 + token * 5 + 3) % 32);
        }
    }
    return result;
}

}  // namespace

TEST(HipInferenceShapeMatrixTest, CpuTokensMatchAcrossBoundaryContextBatchAndCacheDtype) {
    if (runtime::hip_device_count() == 0) GTEST_SKIP() << "No visible HIP device";
    const auto config = hip_shape_matrix_config();
    for (const auto dtype : {DType::Float32, DType::BFloat16}) {
        model::TransformerModel cpu(config, 137);
        model::TransformerModel hip(config, 137);
        hip.to(Device::hip(0));
        for (const auto context : {1, 7, 16, 31, 32, 33, 63, 64, 65, 127, 128}) {
            for (const auto batch : {1, 2, 4, 8}) {
                const auto input = hip_prompts(context, batch);
                const GenerationConfig generation{
                    .max_new_tokens = 3,
                    .temperature = 0.0F,
                    .top_k = 1,
                    .seed = 31,
                    .kv_cache_dtype = dtype,
                    .kv_cache_layer_dtypes = {},
                    .stop_tokens = {}};
                const auto expected = generate_batch(cpu, input, generation);
                const auto actual = generate_batch(hip, input, generation);
                EXPECT_EQ(actual, expected)
                    << "context=" << context << " batch=" << batch
                    << " dtype=" << dtype_name(dtype);
            }
        }
    }
}

TEST(HipInferenceShapeMatrixTest, LastLogitsMatchFullRowsWithoutPayloadTransfer) {
    if (runtime::hip_device_count() == 0) GTEST_SKIP() << "No visible HIP device";
    const auto config = hip_shape_matrix_config();
    const auto input = hip_prompts(7, 2);
    std::vector<std::int32_t> flat;
    for (const auto& row : input) flat.insert(flat.end(), row.begin(), row.end());
    const auto tokens = Tensor::from_int32_vector(flat, {2, 7});
    model::TransformerModel cpu(config, 149);
    const auto full = cpu.forward_inference(tokens).to_vector();
    std::vector<float> expected;
    for (std::int64_t batch = 0; batch < 2; ++batch) {
        const auto offset = static_cast<std::size_t>(
            (batch * 7 + 6) * config.vocabulary_size);
        expected.insert(expected.end(), full.begin() + offset,
                        full.begin() + offset + config.vocabulary_size);
    }

    model::TransformerModel hip(config, 149);
    hip.to(Device::hip(0));
    runtime::reset_transfer_stats();
    const auto last = hip.forward_inference_last_logits(tokens.to(Device::hip(0)));
    runtime::synchronize(Device::hip(0));
    EXPECT_EQ(last.shape(), (Shape{2, 1, config.vocabulary_size}));
    EXPECT_EQ(runtime::transfer_stats().device_to_host_calls, 0U);
    const auto actual = last.to_vector();
    EXPECT_EQ(runtime::transfer_stats().device_to_host_bytes,
              expected.size() * sizeof(float));
    ASSERT_EQ(actual.size(), expected.size());
    for (std::size_t index = 0; index < actual.size(); ++index) {
        EXPECT_NEAR(actual[index], expected[index], 2.0e-4F) << "index=" << index;
    }
}

TEST(HipInferenceShapeMatrixTest, DivergentRowsMatchCpuWithoutPayloadD2H) {
    if (runtime::hip_device_count() == 0) GTEST_SKIP() << "No visible HIP device";
    auto config = hip_shape_matrix_config();
    config.max_sequence_length = 8;
    for (const auto dtype : {DType::Float32, DType::BFloat16}) {
        model::TransformerModel cpu(config, 173);
        model::TransformerModel hip(config, 173);
        hip.to(Device::hip(0));
        KVCache cpu_cache(config.layers, config.max_sequence_length, 2, dtype);
        KVCache hip_cache(config.layers, config.max_sequence_length, 2, dtype);
        const auto prefix = Tensor::from_int32_vector(
            {1, 2, 3, 4, 3, 2}, {2, 3});
        (void)cpu.forward_prefill_cached(prefix, cpu_cache);
        (void)hip.forward_prefill_cached(prefix.to(Device::hip(0)), hip_cache);
        cpu_cache.reset_row(0);
        hip_cache.reset_row(0);
        const auto tokens = Tensor::from_int32_vector({7, 8}, {2, 1});
        const auto device_tokens = tokens.to(Device::hip(0));
        const auto expected = cpu.forward_cached_rows(tokens, cpu_cache).to_vector();
        runtime::reset_transfer_stats();
        const auto device_output = hip.forward_cached_rows(device_tokens, hip_cache);
        runtime::synchronize(Device::hip(0));
        EXPECT_EQ(runtime::transfer_stats().device_to_host_calls, 0U);
        const auto actual = device_output.to_vector();
        const auto tolerance = dtype == DType::Float32 ? 2.0e-4F : 5.0e-2F;
        ASSERT_EQ(actual.size(), expected.size());
        for (std::size_t index = 0; index < actual.size(); ++index) {
            EXPECT_NEAR(actual[index], expected[index], tolerance)
                << "dtype=" << dtype_name(dtype) << " index=" << index;
        }
        EXPECT_EQ(hip_cache.row_positions(),
                  (std::vector<std::int64_t>{1, 4}));
        EXPECT_EQ(hip_cache.row_positions(), cpu_cache.row_positions());
        EXPECT_EQ(hip_cache.layer(0).key.shape()[2], 4);
    }
}

}  // namespace microllm::inference
