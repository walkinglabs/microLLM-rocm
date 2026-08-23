#include <algorithm>
#include <cmath>
#include <cstdint>
#include <string>
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

void expect_finite_close(const std::vector<float>& actual,
                         const std::vector<float>& expected,
                         float maximum_tolerance, float rms_tolerance,
                         const std::string& boundary) {
    ASSERT_EQ(actual.size(), expected.size()) << boundary;
    float maximum = 0.0F;
    double squared = 0.0;
    for (std::size_t index = 0; index < actual.size(); ++index) {
        ASSERT_TRUE(std::isfinite(actual[index])) << boundary << " index=" << index;
        ASSERT_TRUE(std::isfinite(expected[index])) << boundary << " index=" << index;
        const auto difference = std::abs(actual[index] - expected[index]);
        maximum = std::max(maximum, difference);
        squared += static_cast<double>(difference) * difference;
    }
    const auto rms = std::sqrt(squared / static_cast<double>(actual.size()));
    EXPECT_LE(maximum, maximum_tolerance) << boundary << " rms=" << rms;
    EXPECT_LE(rms, rms_tolerance) << boundary << " maximum=" << maximum;
}

}  // namespace

TEST(HipInferenceShapeMatrixTest, CpuLogitsMatchAcrossBoundaryContextBatchAndCacheDtype) {
    if (runtime::hip_device_count() == 0) GTEST_SKIP() << "No visible HIP device";
    const auto config = hip_shape_matrix_config();
    for (const auto dtype : {DType::Float32, DType::BFloat16}) {
        model::TransformerModel cpu(config, 137);
        model::TransformerModel hip(config, 137);
        hip.to(Device::hip(0));
        for (const auto context : {1, 7, 16, 31, 32, 33, 63, 64, 65, 127, 128}) {
            for (const auto batch : {1, 2, 4, 8}) {
                const auto input = hip_prompts(context, batch);
                std::vector<std::int32_t> flat;
                for (const auto& row : input) {
                    flat.insert(flat.end(), row.begin(), row.end());
                }
                const auto tokens = Tensor::from_int32_vector(
                    flat, {batch, context});
                KVCache cpu_cache(
                    config.layers, config.max_sequence_length, batch, dtype);
                KVCache hip_cache(
                    config.layers, config.max_sequence_length, batch, dtype);
                const auto maximum_tolerance =
                    dtype == DType::Float32 ? 2.0e-3F : 1.0e-1F;
                const auto rms_tolerance =
                    dtype == DType::Float32 ? 2.0e-4F : 2.0e-2F;
                const auto boundary =
                    "context=" + std::to_string(context) +
                    " batch=" + std::to_string(batch) +
                    " dtype=" + std::string(dtype_name(dtype));
                const auto expected_uncached =
                    cpu.forward_inference_last_logits(tokens).to_vector();
                const auto actual_uncached = hip.forward_inference_last_logits(
                    tokens.to(Device::hip(0))).to_vector();
                expect_finite_close(
                    actual_uncached, expected_uncached,
                    maximum_tolerance, rms_tolerance, boundary + " uncached");
                const auto expected_prefill =
                    cpu.forward_prefill_cached(tokens, cpu_cache).to_vector();
                const auto actual_prefill = hip.forward_prefill_cached(
                    tokens.to(Device::hip(0)), hip_cache).to_vector();
                expect_finite_close(
                    actual_prefill, expected_prefill,
                    maximum_tolerance, rms_tolerance, boundary + " prefill");

                for (std::int64_t step = 0; step < 2; ++step) {
                    std::vector<std::int32_t> next(
                        static_cast<std::size_t>(batch));
                    for (std::int64_t row = 0; row < batch; ++row) {
                        next[static_cast<std::size_t>(row)] =
                            static_cast<std::int32_t>(
                                (row * 7 + context + step * 11) %
                                config.vocabulary_size);
                    }
                    const auto next_tokens = Tensor::from_int32_vector(
                        next, {batch, 1});
                    const auto expected =
                        cpu.forward_cached(next_tokens, cpu_cache).to_vector();
                    const auto actual = hip.forward_cached(
                        next_tokens.to(Device::hip(0)), hip_cache).to_vector();
                    expect_finite_close(
                        actual, expected, maximum_tolerance, rms_tolerance,
                        boundary + " decode_step=" + std::to_string(step));
                }
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

TEST(HipInferenceShapeMatrixTest, RowPrefillPreservesOtherSlotAndMatchesCpu) {
    if (runtime::hip_device_count() == 0) GTEST_SKIP() << "No visible HIP device";
    auto config = hip_shape_matrix_config();
    config.max_sequence_length = 8;
    for (const auto dtype : {DType::Float32, DType::BFloat16}) {
        model::TransformerModel cpu(config, 179);
        model::TransformerModel hip(config, 179);
        hip.to(Device::hip(0));
        KVCache cpu_cache(config.layers, config.max_sequence_length, 2, dtype);
        KVCache hip_cache(config.layers, config.max_sequence_length, 2, dtype);
        const auto prefix = Tensor::from_int32_vector(
            {1, 2, 3, 4, 3, 2}, {2, 3});
        (void)cpu.forward_prefill_cached(prefix, cpu_cache);
        (void)hip.forward_prefill_cached(prefix.to(Device::hip(0)), hip_cache);
        const auto preserved_key = hip_cache.layer(0).key.slice(0, 1, 2).to_vector();
        cpu_cache.reset_row(0);
        hip_cache.reset_row(0);
        const auto prompt = Tensor::from_int32_vector({7, 8}, {1, 2});
        const auto device_prompt = prompt.to(Device::hip(0));
        const auto expected = cpu.forward_prefill_cached_row(prompt, cpu_cache, 0).to_vector();
        runtime::reset_transfer_stats();
        const auto device_logits = hip.forward_prefill_cached_row(
            device_prompt, hip_cache, 0);
        runtime::synchronize(Device::hip(0));
        EXPECT_EQ(runtime::transfer_stats().device_to_host_calls, 0U);
        const auto actual = device_logits.to_vector();
        const auto tolerance = dtype == DType::Float32 ? 2.0e-4F : 5.0e-2F;
        ASSERT_EQ(actual.size(), expected.size());
        for (std::size_t index = 0; index < actual.size(); ++index) {
            EXPECT_NEAR(actual[index], expected[index], tolerance)
                << "dtype=" << dtype_name(dtype) << " index=" << index;
        }
        EXPECT_EQ(hip_cache.row_positions(),
                  (std::vector<std::int64_t>{2, 3}));
        EXPECT_EQ(hip_cache.row_positions(), cpu_cache.row_positions());
        EXPECT_EQ(hip_cache.layer(0).key.slice(0, 1, 2).to_vector(),
                  preserved_key);
    }
}

TEST(HipInferenceShapeMatrixTest, BatchedRowPrefillMapsEqualPromptsAndMatchesCpu) {
    if (runtime::hip_device_count() == 0) GTEST_SKIP() << "No visible HIP device";
    auto config = hip_shape_matrix_config();
    config.max_sequence_length = 8;
    for (const auto dtype : {DType::Float32, DType::BFloat16}) {
        model::TransformerModel cpu(config, 193);
        model::TransformerModel hip(config, 193);
        hip.to(Device::hip(0));
        KVCache cpu_cache(config.layers, config.max_sequence_length, 4, dtype);
        KVCache hip_cache(config.layers, config.max_sequence_length, 4, dtype);
        const auto existing = Tensor::from_int32_vector({1, 2, 3}, {1, 3});
        (void)cpu.forward_prefill_cached_row(existing, cpu_cache, 0);
        (void)hip.forward_prefill_cached_row(
            existing.to(Device::hip(0)), hip_cache, 0);
        const auto preserved =
            hip_cache.layer(0).key.slice(0, 0, 1).to_vector();
        const auto prompts = Tensor::from_int32_vector(
            {4, 5, 6, 7}, {2, 2});
        const auto expected = cpu.forward_prefill_cached_rows(
            prompts, cpu_cache, {1, 3}).to_vector();
        const auto device_prompts = prompts.to(Device::hip(0));
        runtime::reset_transfer_stats();
        const auto device_logits = hip.forward_prefill_cached_rows(
            device_prompts, hip_cache, {1, 3});
        runtime::synchronize(Device::hip(0));
        EXPECT_EQ(runtime::transfer_stats().device_to_host_calls, 0U);
        const auto actual = device_logits.to_vector();
        const auto tolerance = dtype == DType::Float32 ? 2.0e-4F : 5.0e-2F;
        ASSERT_EQ(actual.size(), expected.size());
        for (std::size_t index = 0; index < actual.size(); ++index) {
            EXPECT_NEAR(actual[index], expected[index], tolerance)
                << "dtype=" << dtype_name(dtype) << " index=" << index;
        }
        EXPECT_EQ(hip_cache.row_positions(),
                  (std::vector<std::int64_t>{3, 2, 0, 2}));
        EXPECT_EQ(hip_cache.row_positions(), cpu_cache.row_positions());
        EXPECT_EQ(hip_cache.layer(0).key.slice(0, 0, 1).to_vector(),
                  preserved);
    }
}

TEST(HipInferenceShapeMatrixTest, ActiveRowsSkipInactiveSlotAndMatchCpu) {
    if (runtime::hip_device_count() == 0) GTEST_SKIP() << "No visible HIP device";
    auto config = hip_shape_matrix_config();
    config.max_sequence_length = 8;
    config.attention_bias = true;
    for (const auto dtype : {DType::Float32, DType::BFloat16}) {
        model::TransformerModel cpu(config, 181);
        model::TransformerModel hip(config, 181);
        hip.to(Device::hip(0));
        KVCache cpu_cache(config.layers, config.max_sequence_length, 3, dtype);
        KVCache hip_cache(config.layers, config.max_sequence_length, 3, dtype);
        const auto first = Tensor::from_int32_vector({1, 2}, {1, 2});
        const auto second = Tensor::from_int32_vector({3, 4, 5}, {1, 3});
        (void)cpu.forward_prefill_cached_row(first, cpu_cache, 0);
        (void)cpu.forward_prefill_cached_row(second, cpu_cache, 1);
        (void)hip.forward_prefill_cached_row(
            first.to(Device::hip(0)), hip_cache, 0);
        (void)hip.forward_prefill_cached_row(
            second.to(Device::hip(0)), hip_cache, 1);
        cpu_cache.reset_row(2);
        hip_cache.reset_row(2);
        const auto full_row = [&hip_cache](const Tensor& tensor,
                                           std::int64_t row) {
            return Tensor::from_storage(
                       tensor.storage(),
                       {1, tensor.shape()[1], hip_cache.max_sequence_length(),
                        tensor.shape()[3]},
                       tensor.strides(),
                       tensor.storage_offset() + row * tensor.stride(0),
                       tensor.dtype())
                .to_vector();
        };
        const auto inactive_key = full_row(hip_cache.layer(0).key, 2);
        const auto inactive_value = full_row(hip_cache.layer(0).value, 2);
        const auto tokens = Tensor::from_int32_vector({9, 10}, {2, 1});
        const auto device_tokens = tokens.to(Device::hip(0));
        const auto expected = cpu.forward_cached_active_rows(
            tokens, cpu_cache, {0, 1}).to_vector();
        runtime::reset_transfer_stats();
        const auto device_logits = hip.forward_cached_active_rows(
            device_tokens, hip_cache, {0, 1});
        runtime::synchronize(Device::hip(0));
        EXPECT_EQ(runtime::transfer_stats().device_to_host_calls, 0U);
        const auto actual = device_logits.to_vector();
        const auto tolerance = dtype == DType::Float32 ? 2.0e-4F : 5.0e-2F;
        ASSERT_EQ(actual.size(), expected.size());
        for (std::size_t index = 0; index < actual.size(); ++index) {
            EXPECT_NEAR(actual[index], expected[index], tolerance)
                << "dtype=" << dtype_name(dtype) << " index=" << index;
        }
        EXPECT_EQ(hip_cache.row_positions(),
                  (std::vector<std::int64_t>{3, 4, 0}));
        EXPECT_EQ(hip_cache.row_positions(), cpu_cache.row_positions());
        EXPECT_EQ(full_row(hip_cache.layer(0).key, 2), inactive_key);
        EXPECT_EQ(full_row(hip_cache.layer(0).value, 2), inactive_value);
    }
}

}  // namespace microllm::inference
