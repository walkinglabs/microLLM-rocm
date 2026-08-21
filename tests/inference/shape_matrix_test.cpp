#include <cstddef>
#include <cstdint>
#include <vector>

#include <gtest/gtest.h>
#include <microllm/inference/generator.h>
#include <microllm/inference/kv_cache.h>
#include <microllm/model/model.h>

namespace microllm::inference {
namespace {

model::ModelConfig shape_matrix_config() {
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

std::vector<std::vector<std::int32_t>> prompts(std::int64_t context,
                                                std::int64_t batch) {
    std::vector<std::vector<std::int32_t>> result(
        static_cast<std::size_t>(batch),
        std::vector<std::int32_t>(static_cast<std::size_t>(context)));
    for (std::int64_t row = 0; row < batch; ++row) {
        for (std::int64_t token = 0; token < context; ++token) {
            result[static_cast<std::size_t>(row)][static_cast<std::size_t>(token)] =
                static_cast<std::int32_t>((row * 11 + token * 7 + 1) % 32);
        }
    }
    return result;
}

std::size_t allocated_cache_bytes(const KVCache& cache) {
    std::size_t result = 0;
    for (std::size_t layer = 0; layer < cache.layer_count(); ++layer) {
        result += cache.layer(layer).key.storage().num_bytes();
        result += cache.layer(layer).value.storage().num_bytes();
    }
    return result;
}

std::size_t active_cache_bytes(const KVCache& cache) {
    std::size_t result = 0;
    for (std::size_t layer = 0; layer < cache.layer_count(); ++layer) {
        result += static_cast<std::size_t>(cache.layer(layer).key.numel()) *
                  dtype_size(cache.layer(layer).key.dtype());
        result += static_cast<std::size_t>(cache.layer(layer).value.numel()) *
                  dtype_size(cache.layer(layer).value.dtype());
    }
    return result;
}

}  // namespace

TEST(InferenceShapeMatrixTest, BoundaryShortLongBatchAndCacheDtypesMatchB1) {
    const auto config = shape_matrix_config();
    constexpr std::int64_t generated_tokens = 3;
    for (const auto dtype : {DType::Float32, DType::BFloat16}) {
        for (const auto context : {1, 7, 16, 31, 32, 33, 63, 64, 65, 127, 128}) {
            for (const auto batch : {1, 2, 4}) {
                const auto input = prompts(context, batch);
                const GenerationConfig generation{
                    .max_new_tokens = generated_tokens,
                    .temperature = 0.0F,
                    .top_k = 1,
                    .seed = 29,
                    .kv_cache_dtype = dtype,
                    .kv_cache_layer_dtypes = {},
                    .stop_tokens = {}};
                model::TransformerModel batched_model(config, 131);
                const auto actual = generate_batch(batched_model, input, generation);
                ASSERT_EQ(actual.size(), input.size());
                for (std::size_t row = 0; row < input.size(); ++row) {
                    model::TransformerModel independent(config, 131);
                    EXPECT_EQ(actual[row], generate(independent, input[row], generation))
                        << "context=" << context << " batch=" << batch
                        << " row=" << row << " dtype=" << dtype_name(dtype);
                }

                KVCache cache(config.layers, context + generated_tokens, batch, dtype);
                std::vector<std::int32_t> flat;
                for (const auto& row : input) flat.insert(flat.end(), row.begin(), row.end());
                model::TransformerModel cache_model(config, 131);
                (void)cache_model.forward_prefill_cached(
                    Tensor::from_int32_vector(flat, {batch, context}), cache);
                const auto expected = static_cast<std::size_t>(
                    2 * config.layers * config.kv_heads * config.head_dimension() *
                    batch * (context + generated_tokens)) * dtype_size(dtype);
                const auto expected_active = static_cast<std::size_t>(
                    2 * config.layers * config.kv_heads * config.head_dimension() *
                    batch * context) * dtype_size(dtype);
                EXPECT_EQ(cache.position(), context);
                EXPECT_EQ(allocated_cache_bytes(cache), expected);
                EXPECT_EQ(active_cache_bytes(cache), expected_active);

                const auto storage_before = cache.layer(0).key.storage().data();
                std::vector<std::int32_t> next(static_cast<std::size_t>(batch), 3);
                (void)cache_model.forward_cached(
                    Tensor::from_int32_vector(next, {batch, 1}), cache);
                EXPECT_EQ(cache.position(), context + 1);
                EXPECT_EQ(allocated_cache_bytes(cache), expected);
                EXPECT_EQ(cache.layer(0).key.storage().data(), storage_before);
                EXPECT_EQ(active_cache_bytes(cache),
                          expected_active + static_cast<std::size_t>(
                              2 * config.layers * config.kv_heads *
                              config.head_dimension() * batch) * dtype_size(dtype));
            }
        }
    }
}

}  // namespace microllm::inference
