#include <cmath>
#include <set>
#include <string>

#include <gtest/gtest.h>
#include <microllm/model/model.h>

namespace microllm::model {
namespace {

ModelConfig tiny_config(bool gqa = true) {
    return {.vocabulary_size = 16,
            .dimension = 8,
            .layers = 1,
            .heads = 2,
            .kv_heads = gqa ? 1 : 2,
            .ffn_dimension = 16,
            .max_sequence_length = 8,
            .rope_base = 10000.0F,
            .tie_embeddings = false};
}

void expect_near(const std::vector<float>& actual, const std::vector<float>& expected,
                 float tolerance) {
    ASSERT_EQ(actual.size(), expected.size());
    for (std::size_t index = 0; index < actual.size(); ++index) {
        EXPECT_NEAR(actual[index], expected[index], tolerance) << "index=" << index;
    }
}

}  // namespace

TEST(TransformerModelTest, ConstructedParametersMatchConfigAndHaveUniqueNames) {
    TransformerModel model(tiny_config(), 7);
    EXPECT_EQ(model.parameter_count(), model.config().parameter_count());
    const auto named = model.named_parameters();
    std::set<std::string> names;
    for (const auto& [name, parameter] : named) {
        EXPECT_TRUE(names.insert(name).second);
        EXPECT_TRUE(parameter->requires_grad());
    }
}

TEST(TransformerModelTest, AttentionBiasAddsNamedTrainableParameters) {
    auto config = tiny_config();
    const auto without_bias = config.parameter_count();
    config.attention_bias = true;
    TransformerModel model(config, 7);
    EXPECT_EQ(config.parameter_count(), without_bias + 16U);
    const auto named = model.named_parameters();
    std::set<std::string> names;
    for (const auto& [name, parameter] : named) {
        names.insert(name);
        (void)parameter;
    }
    EXPECT_TRUE(names.contains("blocks.0.attention.q_proj.bias"));
    EXPECT_TRUE(names.contains("blocks.0.attention.k_proj.bias"));
    EXPECT_TRUE(names.contains("blocks.0.attention.v_proj.bias"));
}

TEST(TransformerModelTest, ForwardAndBackwardCoverEveryParameter) {
    TransformerModel model(tiny_config(), 11);
    const auto tokens = Tensor::from_int32_vector({1, 2, 3, 4, 4, 3, 2, 1}, {2, 4});
    const auto targets = Tensor::from_int32_vector({2, 3, 4, 5, 3, 2, 1, 0}, {2, 4});
    const auto logits = model.forward(tokens);
    EXPECT_EQ(logits.data().shape(), (Shape{2, 4, 16}));
    for (const auto value : logits.data().to_vector()) EXPECT_TRUE(std::isfinite(value));

    const auto loss = model.loss(tokens, targets);
    EXPECT_TRUE(std::isfinite(loss.data().to_vector()[0]));
    loss.backward();
    for (const auto& [name, parameter] : model.named_parameters()) {
        ASSERT_TRUE(parameter->has_grad()) << name;
        EXPECT_EQ(parameter->grad().shape(), parameter->data().shape()) << name;
        for (const auto value : parameter->grad().to_vector()) {
            EXPECT_TRUE(std::isfinite(value)) << name;
        }
    }
}

TEST(TransformerModelTest, GraphFreeInferenceMatchesAutogradForMhaAndGqa) {
    const auto tokens = Tensor::from_int32_vector(
        {1, 2, 3, 4, 4, 3, 2, 1}, {2, 4});
    for (const auto gqa : {false, true}) {
        TransformerModel model(tiny_config(gqa), 13);
        const auto graph = model.forward(tokens).data().to_vector();
        const auto inference = model.forward_inference(tokens);
        EXPECT_EQ(inference.dtype(), DType::Float32);
        EXPECT_EQ(inference.shape(), (Shape{2, 4, 16}));
        expect_near(inference.to_vector(), graph, 2.0e-5F);
    }
}

TEST(TransformerModelTest, Bf16FfnPreparationIsSingleRepresentationAndInferenceOnly) {
    TransformerModel model(tiny_config(), 17);
    const auto input = Tensor::from_int32_vector({1, 2, 3, 4}, {1, 4});
    const auto before = model.forward_inference(input).to_vector();
    const auto report = model.prepare_bf16_ffn_inference();
    EXPECT_TRUE(model.bf16_ffn_inference_prepared());
    EXPECT_EQ(report.converted_tensors, 3U);
    EXPECT_EQ(report.fp32_bytes_released, 3U * 8U * 16U * sizeof(float));
    EXPECT_EQ(report.bf16_bytes_retained, 3U * 8U * 16U * sizeof(std::uint16_t));

    std::size_t bf16_weights = 0;
    for (const auto& [name, parameter] : model.named_parameters()) {
        const auto is_ffn = name.find("feed_forward") != std::string::npos;
        EXPECT_EQ(parameter->data().dtype(),
                  is_ffn ? DType::BFloat16 : DType::Float32) << name;
        if (is_ffn) {
            ++bf16_weights;
            EXPECT_FALSE(parameter->requires_grad()) << name;
        }
    }
    EXPECT_EQ(bf16_weights, 3U);
    const auto after = model.forward_inference(input).to_vector();
    expect_near(after, before, 2.0e-2F);
    EXPECT_THROW((void)model.forward(input), std::logic_error);
    EXPECT_THROW((void)model.prepare_bf16_ffn_inference(), std::logic_error);
}

TEST(TransformerModelTest, Bf16AttentionPreparationConvertsOnlyProjectionWeights) {
    auto config = tiny_config();
    config.attention_bias = true;
    TransformerModel model(config, 18);
    const auto input = Tensor::from_int32_vector({1, 2, 3, 4}, {1, 4});
    const auto before = model.forward_inference(input).to_vector();
    (void)model.prepare_bf16_ffn_inference();
    const auto report = model.prepare_bf16_attention_inference();
    EXPECT_TRUE(model.bf16_attention_inference_prepared());
    EXPECT_EQ(report.converted_tensors, 4U);
    EXPECT_EQ(report.fp32_bytes_released, 192U * sizeof(float));
    EXPECT_EQ(report.bf16_bytes_retained, 192U * sizeof(std::uint16_t));
    std::size_t attention_weights = 0;
    for (const auto& [name, parameter] : model.named_parameters()) {
        const auto selected = name.find(".attention.") != std::string::npos &&
                              name.ends_with(".weight");
        if (selected) {
            ++attention_weights;
            EXPECT_EQ(parameter->data().dtype(), DType::BFloat16) << name;
            EXPECT_FALSE(parameter->requires_grad()) << name;
        } else if (name.ends_with(".bias")) {
            EXPECT_EQ(parameter->data().dtype(), DType::Float32) << name;
        }
    }
    EXPECT_EQ(attention_weights, 4U);
    expect_near(model.forward_inference(input).to_vector(), before, 5.0e-2F);
    EXPECT_THROW((void)model.forward(input), std::logic_error);
    EXPECT_THROW((void)model.prepare_bf16_attention_inference(), std::logic_error);
}

TEST(TransformerModelTest, Fp8LinearPolicyRunsFullForwardLossAndBackward) {
    auto config = tiny_config();
    config.linear_precision = LinearPrecision::Float8E4M3FNUZ;
    config.fp8_activation_scale = 0.025F;
    config.fp8_weight_scale = 0.005F;
    TransformerModel model(config, 11);
    const auto tokens = Tensor::from_int32_vector({1, 2, 3, 4}, {1, 4});
    const auto targets = Tensor::from_int32_vector({2, 3, 4, 5}, {1, 4});
    const auto logits = model.forward(tokens);
    EXPECT_EQ(logits.data().dtype(), DType::Float32);
    EXPECT_EQ(logits.data().shape(), (Shape{1, 4, 16}));
    const auto loss = model.loss(tokens, targets);
    EXPECT_TRUE(std::isfinite(loss.data().to_vector()[0]));
    loss.backward();
    for (const auto& [name, parameter] : model.named_parameters()) {
        ASSERT_TRUE(parameter->has_grad()) << name;
        EXPECT_EQ(parameter->data().dtype(), DType::Float32) << name;
        EXPECT_EQ(parameter->grad().dtype(), DType::Float32) << name;
    }
}

TEST(TransformerModelTest, Bf16LinearPolicyRunsFullForwardLossAndBackward) {
    auto config = tiny_config();
    config.linear_precision = LinearPrecision::BFloat16;
    TransformerModel model(config, 12);
    const auto tokens = Tensor::from_int32_vector({1, 2, 3, 4}, {1, 4});
    const auto targets = Tensor::from_int32_vector({2, 3, 4, 5}, {1, 4});
    const auto logits = model.forward(tokens);
    EXPECT_EQ(logits.data().dtype(), DType::Float32);
    EXPECT_EQ(logits.data().shape(), (Shape{1, 4, 16}));
    const auto loss = model.loss(tokens, targets);
    EXPECT_TRUE(std::isfinite(loss.data().to_vector()[0]));
    loss.backward();
    for (const auto& [name, parameter] : model.named_parameters()) {
        ASSERT_TRUE(parameter->has_grad()) << name;
        EXPECT_EQ(parameter->data().dtype(), DType::Float32) << name;
        EXPECT_EQ(parameter->grad().dtype(), DType::Float32) << name;
    }
    EXPECT_THROW((void)model.prepare_bf16_ffn_inference(), std::logic_error);
}

TEST(TransformerModelTest, Bf16TrainingMirrorsCoverEveryLinearAndPreserveForward) {
    auto config = tiny_config();
    config.linear_precision = LinearPrecision::BFloat16;
    TransformerModel model(config, 14);
    const auto tokens = Tensor::from_int32_vector({1, 2, 3, 4}, {1, 4});
    const auto before = model.forward(tokens).data().to_vector();
    const auto mirrors = model.prepare_bf16_training_mirrors();
    EXPECT_TRUE(model.bf16_training_mirrors_prepared());
    EXPECT_EQ(mirrors.size(), 8U);  // 7 block Linears + untied output head.
    for (const auto& [master, mirror] : mirrors) {
        ASSERT_NE(master, nullptr);
        ASSERT_NE(mirror, nullptr);
        EXPECT_EQ(master->data().dtype(), DType::Float32);
        EXPECT_EQ(mirror->dtype(), DType::BFloat16);
        EXPECT_EQ(mirror->shape(), master->data().shape());
    }
    EXPECT_EQ(model.forward(tokens).data().to_vector(), before);
    EXPECT_THROW((void)model.prepare_bf16_training_mirrors(), std::logic_error);
    EXPECT_THROW((void)model.load_state_dict(model.state_dict()), std::logic_error);
}

TEST(TransformerModelTest, CausalPrefixLogitsIgnoreFutureTokens) {
    TransformerModel model(tiny_config(false), 19);
    const auto first = Tensor::from_int32_vector({1, 2, 3, 4}, {1, 4});
    const auto second = Tensor::from_int32_vector({1, 2, 9, 10}, {1, 4});
    const auto first_logits = model.forward(first).data().to_vector();
    const auto second_logits = model.forward(second).data().to_vector();
    for (std::size_t position = 0; position < 2; ++position) {
        for (std::size_t token = 0; token < 16; ++token) {
            EXPECT_EQ(first_logits[position * 16 + token], second_logits[position * 16 + token]);
        }
    }
}

TEST(TransformerModelTest, RejectsBadTokenShapeAndLongSequence) {
    TransformerModel model(tiny_config(), 23);
    EXPECT_THROW((void)model.forward(Tensor::from_int32_vector({1, 2}, {2})),
                 std::invalid_argument);
    EXPECT_THROW((void)model.forward(Tensor::from_int32_vector(
                                      {1, 2, 3, 4, 5, 6, 7, 8, 9}, {1, 9})),
                 std::invalid_argument);
}

TEST(TransformerModelTest, CachedLogitsMatchFullPrefixForMhaAndGqa) {
    for (const auto gqa : {false, true}) {
        auto config = tiny_config(gqa);
        config.max_sequence_length = 4;
        TransformerModel model(config, 41);
        inference::KVCache cache(model.config().layers, model.config().max_sequence_length);
        const std::vector<std::int32_t> tokens{1, 2, 3, 4};
        const void* key_address = nullptr;
        const void* value_address = nullptr;
        for (std::size_t position = 0; position < tokens.size(); ++position) {
            const auto cached =
                model.forward_cached(Tensor::from_int32_vector({tokens[position]}, {1, 1}), cache)
                    .to_vector();
            const std::vector<std::int32_t> prefix(tokens.begin(), tokens.begin() +
                                                                  static_cast<std::ptrdiff_t>(position + 1));
            const auto full = model.forward(Tensor::from_int32_vector(
                                                prefix, {1, static_cast<std::int64_t>(prefix.size())}))
                                  .data()
                                  .to_vector();
            const auto offset = position * static_cast<std::size_t>(model.config().vocabulary_size);
            for (std::size_t token = 0;
                 token < static_cast<std::size_t>(model.config().vocabulary_size); ++token) {
                EXPECT_NEAR(cached[token], full[offset + token], 2.0e-5F)
                    << "gqa=" << gqa << " position=" << position << " token=" << token;
            }
            if (position == 0) {
                key_address = cache.layer(0).key.storage().data();
                value_address = cache.layer(0).value.storage().data();
            } else {
                EXPECT_EQ(cache.layer(0).key.storage().data(), key_address);
                EXPECT_EQ(cache.layer(0).value.storage().data(), value_address);
            }
            const auto expected_cache_bytes = static_cast<std::size_t>(
                config.kv_heads * config.max_sequence_length *
                config.head_dimension()) * sizeof(float);
            EXPECT_EQ(cache.layer(0).key.storage().num_bytes(), expected_cache_bytes);
            EXPECT_EQ(cache.layer(0).value.storage().num_bytes(), expected_cache_bytes);
            EXPECT_EQ(cache.layer(0).key.numel(),
                      config.kv_heads * static_cast<std::int64_t>(position + 1) *
                          config.head_dimension());
        }
        EXPECT_EQ(cache.position(), 4);
        EXPECT_EQ(cache.layer(0).key.shape()[2], 4);
        EXPECT_THROW((void)model.forward_cached(
                         Tensor::from_int32_vector({5}, {1, 1}), cache),
                     std::out_of_range);
        cache.reset();
        EXPECT_EQ(cache.position(), 0);
        EXPECT_FALSE(cache.layer(0).key.defined());
        const auto reused = model.forward_cached(
            Tensor::from_int32_vector({tokens.front()}, {1, 1}), cache).to_vector();
        TransformerModel fresh(config, 41);
        inference::KVCache fresh_cache(config.layers, config.max_sequence_length);
        EXPECT_EQ(reused, fresh.forward_cached(
                              Tensor::from_int32_vector({tokens.front()}, {1, 1}),
                              fresh_cache).to_vector());

        cache.reset();
        const auto prefix = Tensor::from_int32_vector({1, 2, 3}, {1, 3});
        const auto prefilled = model.forward_prefill_cached(prefix, cache).to_vector();
        const auto full_prefix = model.forward_inference(prefix).to_vector();
        const auto prefix_offset = 2 * model.config().vocabulary_size;
        expect_near(prefilled,
                    std::vector<float>(full_prefix.begin() + prefix_offset,
                                       full_prefix.end()),
                    2.0e-5F);
        EXPECT_EQ(cache.position(), 3);
        EXPECT_EQ(cache.layer(0).key.shape()[2], 3);
        const auto continued = model.forward_cached(
            Tensor::from_int32_vector({4}, {1, 1}), cache).to_vector();
        const auto full_four = model.forward_inference(
            Tensor::from_int32_vector({1, 2, 3, 4}, {1, 4})).to_vector();
        const auto last_offset = 3 * model.config().vocabulary_size;
        expect_near(continued,
                    std::vector<float>(full_four.begin() + last_offset, full_four.end()),
                    2.0e-5F);

        inference::KVCache invalid_cache(config.layers, config.max_sequence_length);
        EXPECT_THROW((void)model.forward_prefill_cached(
                         Tensor::from_int32_vector({1, 2}, {2, 1}), invalid_cache),
                     std::invalid_argument);
        EXPECT_EQ(invalid_cache.position(), 0);
        EXPECT_FALSE(invalid_cache.layer(0).key.defined());
    }
}

}  // namespace microllm::model
