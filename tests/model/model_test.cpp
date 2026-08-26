#include <algorithm>
#include <cmath>
#include <limits>
#include <set>
#include <string>

#include <gtest/gtest.h>
#include <microllm/autograd/diagnostics.h>
#include <microllm/model/model.h>
#include <microllm/ops/ops.h>
#include <microllm/profiling/trace.h>

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

TEST(TransformerModelTest, SplitHalfAttentionLayoutFusionMatchesMaterializedGraph) {
    auto config = tiny_config();
    config.attention_bias = true;
    config.rope_layout = RopeLayout::SplitHalf;
    const auto tokens = Tensor::from_int32_vector({1, 2, 3, 4}, {1, 4});
    const auto targets = Tensor::from_int32_vector({2, 3, 4, 5}, {1, 4});

    autograd::enable_attention_rope_layout_fusion(true);
    TransformerModel fused_model(config, 19);
    const auto fused_loss = fused_model.loss(tokens, targets);
    const auto fused_graph = autograd::inspect_graph(fused_loss);
    fused_loss.backward();

    autograd::enable_attention_rope_layout_fusion(false);
    TransformerModel materialized_model(config, 19);
    const auto materialized_loss = materialized_model.loss(tokens, targets);
    const auto materialized_graph = autograd::inspect_graph(materialized_loss);
    materialized_loss.backward();
    autograd::enable_attention_rope_layout_fusion(true);

    EXPECT_LT(fused_graph.nodes.size(), materialized_graph.nodes.size());
    EXPECT_EQ(std::count_if(
                  fused_graph.nodes.begin(), fused_graph.nodes.end(),
                  [](const auto& node) {
                      return node.operation == "rope_split_half_bias_bthd";
                  }),
              2);
    EXPECT_EQ(std::count_if(
                  fused_graph.nodes.begin(), fused_graph.nodes.end(),
                  [](const auto& node) {
                      return node.operation == "rope_split_half_bias";
                  }),
              0);
    expect_near(fused_loss.data().to_vector(), materialized_loss.data().to_vector(),
                3.0e-5F);
    const auto fused_parameters = fused_model.named_parameters();
    const auto materialized_parameters = materialized_model.named_parameters();
    ASSERT_EQ(fused_parameters.size(), materialized_parameters.size());
    for (std::size_t index = 0; index < fused_parameters.size(); ++index) {
        EXPECT_EQ(fused_parameters[index].first, materialized_parameters[index].first);
        expect_near(fused_parameters[index].second->grad().to_vector(),
                    materialized_parameters[index].second->grad().to_vector(),
                    3.0e-5F);
    }
    EXPECT_TRUE(autograd::attention_rope_layout_fusion_enabled());
}

TEST(TransformerModelTest, AttentionContextLayoutFusionMatchesMaterializedGraph) {
    auto config = tiny_config();
    config.attention_bias = true;
    config.rope_layout = RopeLayout::SplitHalf;
    const auto tokens = Tensor::from_int32_vector({1, 2, 3, 4}, {1, 4});
    const auto targets = Tensor::from_int32_vector({2, 3, 4, 5}, {1, 4});

    autograd::enable_attention_rope_layout_fusion(true);
    autograd::enable_attention_context_layout_fusion(true);
    TransformerModel fused_model(config, 23);
    const auto fused_loss = fused_model.loss(tokens, targets);
    const auto fused_graph = autograd::inspect_graph(fused_loss);
    fused_loss.backward();

    autograd::enable_attention_context_layout_fusion(false);
    TransformerModel materialized_model(config, 23);
    const auto materialized_loss = materialized_model.loss(tokens, targets);
    const auto materialized_graph = autograd::inspect_graph(materialized_loss);
    materialized_loss.backward();
    autograd::enable_attention_context_layout_fusion(true);

    EXPECT_LT(fused_graph.nodes.size(), materialized_graph.nodes.size());
    EXPECT_EQ(std::count_if(
                  fused_graph.nodes.begin(), fused_graph.nodes.end(),
                  [](const auto& node) {
                      return node.operation == "causal_gqa_attention_bthd";
                  }),
              1);
    expect_near(fused_loss.data().to_vector(), materialized_loss.data().to_vector(),
                3.0e-5F);
    const auto fused_parameters = fused_model.named_parameters();
    const auto materialized_parameters = materialized_model.named_parameters();
    ASSERT_EQ(fused_parameters.size(), materialized_parameters.size());
    for (std::size_t index = 0; index < fused_parameters.size(); ++index) {
        EXPECT_EQ(fused_parameters[index].first, materialized_parameters[index].first);
        expect_near(fused_parameters[index].second->grad().to_vector(),
                    materialized_parameters[index].second->grad().to_vector(),
                    3.0e-5F);
    }
    EXPECT_TRUE(autograd::attention_context_layout_fusion_enabled());
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
        profiling::TraceOptions trace_options;
        trace_options.record_operators = false;
        trace_options.max_captured_elements = 256;
        profiling::TraceSession trace("microllm", "inference-unit",
                                      trace_options);
        Tensor inference;
        {
            profiling::ScopedTraceSession active(trace);
            inference = model.forward_inference(tokens);
        }
        ASSERT_EQ(trace.records().size(), 18U);
        EXPECT_EQ(trace.records()[1].name, "inference.embedding");
        EXPECT_EQ(trace.records()[2].name,
                  "inference.blocks.0.attention_norm");
        EXPECT_EQ(trace.records()[3].name,
                  "inference.blocks.0.attention.q_projection");
        EXPECT_EQ(trace.records()[14].name, "inference.blocks.0");
        EXPECT_EQ(trace.records()[15].name, "inference.final_norm");
        EXPECT_EQ(trace.records()[16].name, "inference.logits");
        EXPECT_EQ(trace.records()[16].values.size(), 128U);
        EXPECT_EQ(inference.dtype(), DType::Float32);
        EXPECT_EQ(inference.shape(), (Shape{2, 4, 16}));
        expect_near(inference.to_vector(), graph, 2.0e-5F);
        const auto last = model.forward_inference_last_logits(tokens);
        EXPECT_EQ(last.dtype(), DType::Float32);
        EXPECT_EQ(last.shape(), (Shape{2, 1, 16}));
        const auto full = inference.to_vector();
        std::vector<float> expected_last;
        for (std::int64_t batch = 0; batch < 2; ++batch) {
            const auto offset = static_cast<std::size_t>((batch * 4 + 3) * 16);
            expected_last.insert(expected_last.end(), full.begin() + offset,
                                 full.begin() + offset + 16U);
        }
        expect_near(last.to_vector(), expected_last, 2.0e-5F);
    }
}

TEST(TransformerModelTest, TraceAllLayerDetailsExposesEveryLinearInputBoundary) {
    auto config = tiny_config();
    config.layers = 2;
    TransformerModel model(config, 31);
    profiling::TraceOptions options;
    options.record_operators = false;
    options.record_all_layer_details = true;
    options.capture_values = false;
    profiling::TraceSession trace("microllm", "all-layer-details", options);
    {
        profiling::ScopedTraceSession active(trace);
        (void)model.forward_inference(
            Tensor::from_int32_vector({1, 2}, {1, 2}));
    }
    const auto has = [&](const std::string& name) {
        return std::any_of(trace.records().begin(), trace.records().end(),
                           [&](const auto& record) { return record.name == name; });
    };
    for (const auto layer : {0, 1}) {
        const auto prefix = "inference.blocks." + std::to_string(layer);
        EXPECT_TRUE(has(prefix + ".attention_norm"));
        EXPECT_TRUE(has(prefix + ".attention.context"));
        EXPECT_TRUE(has(prefix + ".ffn_norm"));
        EXPECT_TRUE(has(prefix + ".ffn.activated"));
    }

    inference::KVCache cache(config.layers, config.max_sequence_length, 1,
                             DType::BFloat16);
    (void)model.forward_prefill_cached(
        Tensor::from_int32_vector({1, 2}, {1, 2}), cache);
    profiling::TraceSession cached_trace(
        "microllm", "cached-all-layer-details", options);
    {
        profiling::ScopedTraceSession active(cached_trace);
        (void)model.forward_cached(
            Tensor::from_int32_vector({3}, {1, 1}), cache);
    }
    const auto cached_has = [&](const std::string& name) {
        return std::any_of(cached_trace.records().begin(),
                           cached_trace.records().end(),
                           [&](const auto& record) {
                               return record.name == name;
                           });
    };
    for (const auto layer : {0, 1}) {
        const auto prefix =
            "inference.cached.blocks." + std::to_string(layer);
        EXPECT_TRUE(cached_has(prefix + ".attention.q_projection"));
        EXPECT_TRUE(cached_has(prefix + ".attention.k_projection"));
        EXPECT_TRUE(cached_has(prefix + ".attention.v_projection"));
        EXPECT_TRUE(cached_has(prefix + ".attention.q_rope"));
        EXPECT_TRUE(cached_has(prefix + ".attention.k_rope"));
        EXPECT_TRUE(cached_has(prefix + ".attention.value"));
        EXPECT_TRUE(cached_has(prefix + ".attention.context"));
        EXPECT_TRUE(cached_has(prefix + ".attention.output"));
        EXPECT_TRUE(cached_has(prefix + ".attention_residual"));
        EXPECT_TRUE(cached_has(prefix + ".ffn_norm"));
        EXPECT_TRUE(cached_has(prefix + ".ffn.activated"));
    }

    inference::KVCache prefill_cache(
        config.layers, config.max_sequence_length, 1, DType::BFloat16);
    profiling::TraceSession cached_prefill_trace(
        "microllm", "cached-prefill-details", options);
    {
        profiling::ScopedTraceSession active(cached_prefill_trace);
        (void)model.forward_prefill_cached(
            Tensor::from_int32_vector({1, 2}, {1, 2}), prefill_cache);
    }
    const auto prefill_has = [&](const std::string& name) {
        return std::any_of(cached_prefill_trace.records().begin(),
                           cached_prefill_trace.records().end(),
                           [&](const auto& record) {
                               return record.name == name;
                           });
    };
    for (const auto layer : {0, 1}) {
        const auto prefix =
            "inference.cached_prefill.blocks." + std::to_string(layer);
        EXPECT_TRUE(prefill_has(prefix + ".attention_norm"));
        EXPECT_TRUE(prefill_has(prefix + ".attention.q_projection"));
        EXPECT_TRUE(prefill_has(prefix + ".attention.k_projection"));
        EXPECT_TRUE(prefill_has(prefix + ".attention.v_projection"));
        EXPECT_TRUE(prefill_has(prefix + ".attention.q_rope"));
        EXPECT_TRUE(prefill_has(prefix + ".attention.k_rope"));
        EXPECT_TRUE(prefill_has(prefix + ".attention.cache_key"));
        EXPECT_TRUE(prefill_has(prefix + ".attention.cache_value"));
        EXPECT_FALSE(prefill_has(prefix + ".attention.scaled_query"));
        EXPECT_FALSE(prefill_has(prefix + ".attention.scores"));
        EXPECT_FALSE(prefill_has(prefix + ".attention.probabilities"));
        EXPECT_FALSE(prefill_has(prefix + ".attention.pv_output"));
        EXPECT_TRUE(prefill_has(prefix + ".attention.context"));
        EXPECT_TRUE(prefill_has(prefix + ".ffn_output"));
        EXPECT_TRUE(prefill_has(prefix + ".output"));
    }
}

TEST(TransformerModelTest,
     CachedPrefillTraceCapturesRequestedAttentionCoreWithoutChangingOutput) {
    const auto config = tiny_config();
    TransformerModel model(config, 37);
    const auto tokens = Tensor::from_int32_vector({1, 2, 3}, {1, 3});
    inference::KVCache baseline_cache(
        config.layers, config.max_sequence_length, 1, DType::BFloat16);
    const auto baseline =
        model.forward_prefill_cached(tokens, baseline_cache).to_vector();

    profiling::TraceOptions options;
    options.record_operators = false;
    options.record_all_layer_details = true;
    options.capture_values = true;
    options.value_name_filters = {
        "inference.cached_prefill.blocks.0.attention.scores"};
    options.max_captured_elements = 64;
    profiling::TraceSession trace(
        "microllm", "cached-prefill-attention-core", options);
    inference::KVCache diagnostic_cache(
        config.layers, config.max_sequence_length, 1, DType::BFloat16);
    std::vector<float> diagnostic;
    {
        profiling::ScopedTraceSession active(trace);
        diagnostic =
            model.forward_prefill_cached(tokens, diagnostic_cache).to_vector();
    }
    expect_near(diagnostic, baseline, 1.0e-6F);

    const auto find = [&](const std::string& name) -> const profiling::TraceRecord* {
        const auto found = std::find_if(
            trace.records().begin(), trace.records().end(),
            [&](const auto& record) { return record.name == name; });
        return found == trace.records().end() ? nullptr : &*found;
    };
    const std::string prefix =
        "inference.cached_prefill.blocks.0.attention.";
    const auto* scaled_query = find(prefix + "scaled_query");
    const auto* scores = find(prefix + "scores");
    const auto* probabilities = find(prefix + "probabilities");
    const auto* pv_output = find(prefix + "pv_output");
    ASSERT_NE(scaled_query, nullptr);
    ASSERT_NE(scores, nullptr);
    ASSERT_NE(probabilities, nullptr);
    ASSERT_NE(pv_output, nullptr);
    EXPECT_EQ(scaled_query->shape, (Shape{1, 2, 3, 4}));
    EXPECT_EQ(scores->shape, (Shape{1, 2, 3, 3}));
    EXPECT_EQ(probabilities->shape, scores->shape);
    EXPECT_EQ(pv_output->shape, scaled_query->shape);
    EXPECT_TRUE(scaled_query->values.empty());
    EXPECT_EQ(scores->values.size(), 18U);
    EXPECT_TRUE(probabilities->values.empty());
    EXPECT_TRUE(pv_output->values.empty());
    EXPECT_EQ(baseline_cache.layer(0).key.to_vector(),
              diagnostic_cache.layer(0).key.to_vector());
    EXPECT_EQ(baseline_cache.layer(0).value.to_vector(),
              diagnostic_cache.layer(0).value.to_vector());

    options.capture_values = false;
    profiling::TraceSession metadata_only_trace(
        "microllm", "cached-prefill-attention-core-metadata", options);
    inference::KVCache metadata_only_cache(
        config.layers, config.max_sequence_length, 1, DType::BFloat16);
    std::vector<float> metadata_only;
    {
        profiling::ScopedTraceSession active(metadata_only_trace);
        metadata_only =
            model.forward_prefill_cached(tokens, metadata_only_cache).to_vector();
    }
    expect_near(metadata_only, baseline, 1.0e-6F);
    EXPECT_FALSE(std::any_of(
        metadata_only_trace.records().begin(),
        metadata_only_trace.records().end(),
        [&](const auto& record) {
            return record.name == prefix + "scaled_query" ||
                   record.name == prefix + "scores" ||
                   record.name == prefix + "probabilities" ||
                   record.name == prefix + "pv_output";
        }));
}

TEST(TransformerModelTest, Bf16FfnPreparationIsSingleRepresentationAndInferenceOnly) {
    TransformerModel model(tiny_config(), 17);
    const auto input = Tensor::from_int32_vector({1, 2, 3, 4}, {1, 4});
    EXPECT_THROW(model.set_bf16_ffn_arena_enabled(true), std::logic_error);
    EXPECT_THROW(model.set_bf16_ffn_norm_fusion_enabled(true), std::logic_error);
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
    profiling::TraceOptions trace_options;
    trace_options.record_operators = false;
    trace_options.max_captured_elements = 256;
    profiling::TraceSession trace("microllm", "bf16-ffn-inference",
                                  trace_options);
    std::vector<float> after;
    {
        profiling::ScopedTraceSession active(trace);
        after = model.forward_inference(input).to_vector();
    }
    ASSERT_EQ(trace.records().size(), 23U);
    EXPECT_EQ(trace.records()[13].name,
              "inference.blocks.0.ffn.input_bf16");
    EXPECT_EQ(trace.records()[14].name, "inference.blocks.0.ffn.gate");
    EXPECT_EQ(trace.records()[15].name, "inference.blocks.0.ffn.up");
    EXPECT_EQ(trace.records()[16].name,
              "inference.blocks.0.ffn.activated");
    EXPECT_EQ(trace.records()[17].name, "inference.blocks.0.ffn.down");
    expect_near(after, before, 2.0e-2F);
    model.set_bf16_ffn_arena_enabled(true);
    EXPECT_TRUE(model.bf16_ffn_arena_enabled());
    EXPECT_TRUE(model.bf16_ffn_norm_fusion_enabled());
    const auto arena_first = model.forward_inference(input).to_vector();
    const auto arena_second = model.forward_inference(input).to_vector();
    EXPECT_EQ(arena_first, after);
    EXPECT_EQ(arena_second, after);
    const auto arena_stats = model.bf16_ffn_arena_stats();
    EXPECT_EQ(arena_stats.entries, 1U);
    EXPECT_EQ(arena_stats.misses, 1U);
    EXPECT_EQ(arena_stats.hits,
              static_cast<std::size_t>(model.config().layers * 2 - 1));
    EXPECT_EQ(arena_stats.eligible_calls,
              static_cast<std::size_t>(model.config().layers * 2));
    EXPECT_EQ(arena_stats.bypassed_calls, 0U);
    EXPECT_EQ(arena_stats.minimum_rows, 1);
    EXPECT_GT(arena_stats.capacity_bytes, 0U);
    model.set_bf16_ffn_norm_fusion_enabled(false);
    EXPECT_FALSE(model.bf16_ffn_norm_fusion_enabled());
    EXPECT_EQ(model.forward_inference(input).to_vector(), after);
    model.set_bf16_ffn_norm_fusion_enabled(true);
    EXPECT_TRUE(model.bf16_ffn_norm_fusion_enabled());
    EXPECT_EQ(model.forward_inference(input).to_vector(), after);
    const auto short_input = Tensor::from_int32_vector({1, 2}, {1, 2});
    const auto short_expected = model.forward_inference(short_input).to_vector();
    EXPECT_EQ(model.bf16_ffn_arena_stats().entries, 2U);
    EXPECT_EQ(model.bf16_ffn_arena_stats().misses, 2U);
    model.to(Device::cpu());
    EXPECT_TRUE(model.bf16_ffn_arena_enabled());
    EXPECT_EQ(model.bf16_ffn_arena_stats().entries, 0U);
    EXPECT_THROW(model.set_bf16_ffn_arena_enabled(true, 0),
                 std::invalid_argument);
    model.set_bf16_ffn_arena_enabled(true, 3);
    EXPECT_EQ(model.forward_inference(short_input).to_vector(), short_expected);
    auto selective_stats = model.bf16_ffn_arena_stats();
    EXPECT_EQ(selective_stats.entries, 0U);
    EXPECT_EQ(selective_stats.eligible_calls, 0U);
    EXPECT_EQ(selective_stats.bypassed_calls,
              static_cast<std::size_t>(model.config().layers));
    EXPECT_EQ(selective_stats.minimum_rows, 3);
    EXPECT_EQ(model.forward_inference(input).to_vector(), after);
    selective_stats = model.bf16_ffn_arena_stats();
    EXPECT_EQ(selective_stats.entries, 1U);
    EXPECT_EQ(selective_stats.misses, 1U);
    EXPECT_EQ(selective_stats.eligible_calls,
              static_cast<std::size_t>(model.config().layers));
    model.set_bf16_ffn_arena_enabled(false);
    EXPECT_FALSE(model.bf16_ffn_arena_enabled());
    EXPECT_FALSE(model.bf16_ffn_norm_fusion_enabled());
    EXPECT_EQ(model.bf16_ffn_arena_stats().capacity_bytes, 0U);
    EXPECT_EQ(model.bf16_ffn_arena_stats().minimum_rows, 1);
    EXPECT_THROW((void)model.forward(input), std::logic_error);
    EXPECT_THROW((void)model.prepare_bf16_ffn_inference(), std::logic_error);
}

TEST(TransformerModelTest, Bf16FfnPreparationCanKeepExplicitBlocksFp32) {
    auto config = tiny_config();
    config.layers = 2;
    TransformerModel model(config, 19);
    const auto input = Tensor::from_int32_vector({1, 2, 3, 4}, {1, 4});
    const auto before = model.forward_inference(input).to_vector();
    const auto report = model.prepare_bf16_ffn_inference({0});
    EXPECT_EQ(report.converted_tensors, 3U);
    std::size_t block0_fp32 = 0;
    std::size_t block1_bf16 = 0;
    for (const auto& [name, parameter] : model.named_parameters()) {
        if (name.starts_with("blocks.0.feed_forward.") &&
            name.ends_with(".weight")) {
            EXPECT_EQ(parameter->data().dtype(), DType::Float32) << name;
            ++block0_fp32;
        }
        if (name.starts_with("blocks.1.feed_forward.") &&
            name.ends_with(".weight")) {
            EXPECT_EQ(parameter->data().dtype(), DType::BFloat16) << name;
            ++block1_bf16;
        }
    }
    EXPECT_EQ(block0_fp32, 3U);
    EXPECT_EQ(block1_bf16, 3U);
    expect_near(model.forward_inference(input).to_vector(), before, 5.0e-2F);

    TransformerModel duplicate(config, 20);
    EXPECT_THROW((void)duplicate.prepare_bf16_ffn_inference({0, 0}),
                 std::invalid_argument);
    EXPECT_FALSE(duplicate.bf16_ffn_inference_prepared());
    TransformerModel outside(config, 21);
    EXPECT_THROW((void)outside.prepare_bf16_ffn_inference({2}),
                 std::invalid_argument);
    EXPECT_FALSE(outside.bf16_ffn_inference_prepared());
}

TEST(TransformerModelTest, Bf16AttentionPreparationConvertsOnlyProjectionWeights) {
    auto config = tiny_config();
    config.attention_bias = true;
    TransformerModel model(config, 18);
    const auto input = Tensor::from_int32_vector({1, 2, 3, 4}, {1, 4});
    EXPECT_THROW(model.set_bf16_qkv_arena_enabled(true), std::logic_error);
    EXPECT_THROW(model.set_bf16_attention_norm_fusion_enabled(true),
                 std::logic_error);
    EXPECT_THROW((void)model.prewarm_bf16_grouped_qkv(4), std::logic_error);
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
    const auto prepared = model.forward_inference(input).to_vector();
    expect_near(prepared, before, 5.0e-2F);
    model.set_bf16_qkv_arena_enabled(true, 5);
    EXPECT_TRUE(model.bf16_attention_norm_fusion_enabled());
    EXPECT_THROW((void)model.prewarm_bf16_grouped_qkv(4), std::logic_error);
    EXPECT_EQ(model.forward_inference(input).to_vector(), prepared);
    auto arena_stats = model.bf16_qkv_arena_stats();
    EXPECT_EQ(arena_stats.entries, 0U);
    EXPECT_EQ(arena_stats.bypassed_calls,
              static_cast<std::size_t>(model.config().layers));
    EXPECT_EQ(arena_stats.minimum_rows, 5);
    model.set_bf16_qkv_arena_enabled(true, 1);
    EXPECT_EQ(model.forward_inference(input).to_vector(), prepared);
    EXPECT_EQ(model.forward_inference(input).to_vector(), prepared);
    arena_stats = model.bf16_qkv_arena_stats();
    EXPECT_EQ(arena_stats.entries, 1U);
    EXPECT_EQ(arena_stats.misses, 1U);
    EXPECT_EQ(arena_stats.hits,
              static_cast<std::size_t>(model.config().layers * 2 - 1));
    EXPECT_EQ(arena_stats.eligible_calls,
              static_cast<std::size_t>(model.config().layers * 2));
    EXPECT_GT(arena_stats.capacity_bytes, 0U);
    model.set_bf16_attention_norm_fusion_enabled(false);
    EXPECT_FALSE(model.bf16_attention_norm_fusion_enabled());
    EXPECT_EQ(model.forward_inference(input).to_vector(), prepared);
    model.set_bf16_attention_norm_fusion_enabled(true);
    EXPECT_TRUE(model.bf16_attention_norm_fusion_enabled());
    EXPECT_EQ(model.forward_inference(input).to_vector(), prepared);
    model.set_bf16_qkv_arena_enabled(false);
    EXPECT_FALSE(model.bf16_qkv_arena_enabled());
    EXPECT_FALSE(model.bf16_attention_norm_fusion_enabled());
    EXPECT_EQ(model.bf16_qkv_arena_stats().entries, 0U);
    EXPECT_THROW(model.set_attention_core_arena_enabled(true, 0),
                 std::invalid_argument);
    model.set_attention_core_arena_enabled(true, 5);
    EXPECT_EQ(model.forward_inference(input).to_vector(), prepared);
    auto core_stats = model.attention_core_arena_stats();
    EXPECT_EQ(core_stats.entries, 0U);
    EXPECT_EQ(core_stats.bypassed_calls,
              static_cast<std::size_t>(model.config().layers));
    EXPECT_EQ(core_stats.minimum_sequence, 5);
    model.set_attention_core_arena_enabled(true, 1);
    EXPECT_EQ(model.forward_inference(input).to_vector(), prepared);
    EXPECT_EQ(model.forward_inference(input).to_vector(), prepared);
    core_stats = model.attention_core_arena_stats();
    EXPECT_EQ(core_stats.entries, 1U);
    EXPECT_EQ(core_stats.misses, 1U);
    EXPECT_EQ(core_stats.hits,
              static_cast<std::size_t>(model.config().layers * 2 - 1));
    EXPECT_GT(core_stats.capacity_bytes, 0U);
    model.set_attention_core_arena_enabled(false);
    EXPECT_FALSE(model.attention_core_arena_enabled());
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

TEST(TransformerModelTest, Fp8InferencePreparationCachesOneByteLinearWeights) {
    auto config = tiny_config();
    config.linear_precision = LinearPrecision::Float8E4M3FNUZ;
    config.fp8_activation_scale = 0.025F;
    config.fp8_weight_scale = 0.005F;
    TransformerModel model(config, 19);
    const auto tokens = Tensor::from_int32_vector({1, 2, 3, 4}, {1, 4});
    const auto before = model.forward_inference(tokens).to_vector();
    const auto state = model.state_dict();
    const auto report = model.prepare_fp8_inference_weights();
    EXPECT_EQ(report.converted_tensors, 8U);
    EXPECT_EQ(report.fp32_bytes_released, report.fp8_bytes_retained * 4U);
    EXPECT_EQ(report.scale_bytes_retained,
              report.converted_tensors * 2U * sizeof(float));
    EXPECT_EQ(report.weight_bytes_scanned, 0U);
    EXPECT_FLOAT_EQ(report.minimum_weight_scale, config.fp8_weight_scale);
    EXPECT_FLOAT_EQ(report.maximum_weight_scale, config.fp8_weight_scale);
    EXPECT_TRUE(model.fp8_inference_weights_prepared());
    std::size_t fp8_weights = 0;
    for (const auto& [name, parameter] : model.named_parameters()) {
        if (name.ends_with(".weight") &&
            name.find("norm") == std::string::npos &&
            name != "token_embedding.weight") {
            EXPECT_EQ(parameter->data().dtype(), DType::Float8E4M3FNUZ)
                << name;
            EXPECT_FALSE(parameter->requires_grad()) << name;
            ++fp8_weights;
        }
    }
    EXPECT_EQ(fp8_weights, 8U);
    EXPECT_EQ(model.forward_inference(tokens).to_vector(), before);
    EXPECT_THROW((void)model.forward(tokens), std::logic_error);
    EXPECT_THROW((void)model.prepare_fp8_inference_weights(), std::logic_error);
    EXPECT_THROW((void)model.load_state_dict(state), std::logic_error);
    TransformerModel unloaded(
        config, 19, ParameterInitialization::Uninitialized);
    EXPECT_THROW((void)unloaded.prepare_fp8_inference_weights(),
                 std::logic_error);
}

TEST(TransformerModelTest, Fp8TensorAmaxPreparationReportsIndependentWeightScales) {
    auto config = tiny_config();
    config.linear_precision = LinearPrecision::Float8E4M3FNUZ;
    config.fp8_activation_scale = 0.2F;
    config.fp8_weight_scale = 0.005F;
    config.fp8_weight_scale_mode = Fp8WeightScaleMode::TensorAmax;
    TransformerModel model(config, 23);
    auto state = model.state_dict();
    float multiplier = 1.0F;
    for (auto& [name, tensor] : state) {
        if (name.ends_with(".weight") && name.find("norm") == std::string::npos &&
            name != "token_embedding.weight") {
            auto values = tensor.to_vector();
            for (auto& value : values) value *= multiplier;
            tensor = Tensor::from_vector(values, tensor.shape());
            multiplier *= 1.5F;
        }
    }
    ASSERT_TRUE(model.load_state_dict(state).complete());
    const auto report = model.prepare_fp8_inference_weights();
    EXPECT_EQ(report.converted_tensors, 8U);
    EXPECT_EQ(report.weight_bytes_scanned, report.fp32_bytes_released);
    EXPECT_GT(report.minimum_weight_scale, 0.0F);
    EXPECT_GT(report.maximum_weight_scale, report.minimum_weight_scale);
    EXPECT_TRUE(model.fp8_inference_weights_prepared());
    const auto tokens = Tensor::from_int32_vector({1, 2}, {1, 2});
    EXPECT_THROW((void)model.forward(tokens), std::logic_error);
}

TEST(TransformerModelTest, Fp8OutputChannelPreparationRetainsEveryColumnScale) {
    auto config = tiny_config();
    config.linear_precision = LinearPrecision::Float8E4M3FNUZ;
    config.fp8_weight_scale = 1.0e-4F;
    config.fp8_weight_scale_mode = Fp8WeightScaleMode::OutputChannelAmax;
    config.fp8_activation_scale_mode = Fp8ActivationScaleMode::TensorAmax;
    TransformerModel model(config, 27);
    const auto tokens = Tensor::from_int32_vector({1, 2, 3, 4}, {1, 4});
    const auto before = model.forward_inference(tokens).to_vector();
    ops::clear_fp8_dynamic_quant_stats();
    const auto report = model.prepare_fp8_inference_weights();
    EXPECT_EQ(report.linears_covered, 8U);
    EXPECT_EQ(report.converted_tensors, 8U);
    EXPECT_EQ(report.scale_bytes_retained, 80U * sizeof(float));
    EXPECT_EQ(report.weight_bytes_scanned, report.fp32_bytes_released);
    EXPECT_EQ(report.device_amax_tensors, 0U);
    EXPECT_FALSE(report.host_scale_summary_available);
    EXPECT_EQ(ops::fp8_dynamic_quant_stats().column_calls, 8U);
    ops::clear_fp8_dynamic_quant_stats();
    EXPECT_EQ(model.forward_inference(tokens).to_vector(), before);
    EXPECT_EQ(ops::fp8_dynamic_quant_stats().column_calls, 0U);
    EXPECT_EQ(ops::fp8_dynamic_quant_stats().tensor_calls, 5U);
}

TEST(TransformerModelTest, Fp8AttentionOutputScopeChangesOnlyOProjection) {
    const auto tokens = Tensor::from_int32_vector({1, 2, 3, 4}, {1, 4});
    for (const auto tied : {false, true}) {
        auto config = tiny_config();
        config.tie_embeddings = tied;
        config.linear_precision = LinearPrecision::Float8E4M3FNUZ;
        config.fp8_weight_scale = 1.0e-4F;
        config.fp8_weight_scale_mode = Fp8WeightScaleMode::OutputChannelAmax;
        config.fp8_weight_scale_scope =
            Fp8WeightScaleScope::AttentionOutputOnly;
        config.fp8_activation_scale_mode = Fp8ActivationScaleMode::TensorAmax;
        TransformerModel model(config, 33);
        const auto before = model.forward_inference(tokens).to_vector();
        ops::clear_fp8_dynamic_quant_stats();
        const auto report = model.prepare_fp8_inference_weights();
        EXPECT_EQ(report.converted_tensors, tied ? 7U : 8U);
        EXPECT_EQ(report.scale_bytes_retained,
                  (tied ? 8U + 6U : 8U + 7U) * sizeof(float));
        EXPECT_EQ(ops::fp8_dynamic_quant_stats().column_calls, 1U);
        EXPECT_EQ(ops::fp8_dynamic_quant_stats().tensor_calls, tied ? 6U : 7U);
        ops::clear_fp8_dynamic_quant_stats();
        EXPECT_EQ(model.forward_inference(tokens).to_vector(), before);
        EXPECT_EQ(ops::fp8_dynamic_quant_stats().column_calls, 0U);
        EXPECT_EQ(ops::fp8_dynamic_quant_stats().tensor_calls,
                  tied ? 4U : 5U);
    }
}

TEST(TransformerModelTest, Fp8TensorAmaxPreparationRejectsNonfiniteWeightTransactionally) {
    auto config = tiny_config();
    config.linear_precision = LinearPrecision::Float8E4M3FNUZ;
    config.fp8_weight_scale_mode = Fp8WeightScaleMode::TensorAmax;
    TransformerModel model(config, 29);
    auto state = model.state_dict();
    auto selected = std::find_if(
        state.begin(), state.end(), [](const auto& entry) {
            return entry.first.ends_with(".weight") &&
                   entry.first.find("norm") == std::string::npos &&
                   entry.first != "token_embedding.weight";
        });
    ASSERT_NE(selected, state.end());
    auto values = selected->second.to_vector();
    values[0] = std::numeric_limits<float>::infinity();
    selected->second = Tensor::from_vector(values, selected->second.shape());
    ASSERT_TRUE(model.load_state_dict(state).complete());
    EXPECT_THROW((void)model.prepare_fp8_inference_weights(), std::invalid_argument);
    EXPECT_FALSE(model.fp8_inference_weights_prepared());
    for (const auto& [name, parameter] : model.named_parameters()) {
        if (name.ends_with(".weight")) {
            EXPECT_EQ(parameter->data().dtype(), DType::Float32) << name;
        }
    }
}

TEST(TransformerModelTest, Fp8DynamicActivationScaleNeedsNoPersistentScaleTensor) {
    auto config = tiny_config();
    config.linear_precision = LinearPrecision::Float8E4M3FNUZ;
    config.fp8_activation_scale = 1.0e-4F;
    config.fp8_activation_scale_mode = Fp8ActivationScaleMode::TensorAmax;
    TransformerModel model(config, 37);
    const auto tokens = Tensor::from_int32_vector({1, 2, 3, 4}, {1, 4});
    const auto before = model.forward_inference(tokens).to_vector();
    const auto report = model.prepare_fp8_inference_weights();
    EXPECT_EQ(report.scale_bytes_retained,
              report.converted_tensors * sizeof(float));
    EXPECT_EQ(model.forward_inference(tokens).to_vector(), before);
    EXPECT_THROW((void)model.forward(tokens), std::logic_error);
}

TEST(TransformerModelTest, Fp8FfnOuterRowPreparesScalesOnlyForNonFfnLinears) {
    auto config = tiny_config();
    config.linear_precision = LinearPrecision::Float8E4M3FNUZ;
    config.fp8_activation_scale = 0.2F;
    config.fp8_activation_minimum_scale = 1.0e-4F;
    config.fp8_activation_scale_mode = Fp8ActivationScaleMode::FfnOuterRow;
    TransformerModel model(config, 41);
    const auto tokens = Tensor::from_int32_vector({1, 2, 3, 4}, {1, 4});
    const auto before = model.forward_inference(tokens).to_vector();
    const auto report = model.prepare_fp8_inference_weights();
    EXPECT_EQ(report.converted_tensors, 8U);
    EXPECT_EQ(report.scale_bytes_retained, 13U * sizeof(float));
    EXPECT_EQ(model.forward_inference(tokens).to_vector(), before);
    EXPECT_THROW((void)model.forward(tokens), std::logic_error);
}

TEST(TransformerModelTest, Fp8SelectedBlockRemainsFp32AcrossPreparation) {
    auto config = tiny_config();
    config.layers = 2;
    config.linear_precision = LinearPrecision::Float8E4M3FNUZ;
    config.fp8_activation_scale_mode = Fp8ActivationScaleMode::TensorAmax;
    config.fp8_fp32_layers = {1};
    TransformerModel model(config, 43);
    const auto tokens = Tensor::from_int32_vector({1, 2, 3, 4}, {1, 4});
    const auto before = model.forward_inference(tokens).to_vector();
    const auto report = model.prepare_fp8_inference_weights();
    EXPECT_EQ(report.converted_tensors, 8U);  // block0 seven + output head one.
    std::size_t selected_fp32 = 0;
    for (const auto& [name, parameter] : model.named_parameters()) {
        if (name.starts_with("blocks.1.") && name.ends_with(".weight") &&
            name.find("norm") == std::string::npos) {
            EXPECT_EQ(parameter->data().dtype(), DType::Float32) << name;
            ++selected_fp32;
        }
    }
    EXPECT_EQ(selected_fp32, 7U);
    EXPECT_EQ(model.forward_inference(tokens).to_vector(), before);
}

TEST(TransformerModelTest, Fp8DiagnosticModesIsolateWeightAndActivationRounding) {
    const auto tokens = Tensor::from_int32_vector({1, 2, 3, 4}, {1, 4});
    for (const auto mode : {Fp8DiagnosticMode::WeightOnly,
                            Fp8DiagnosticMode::ActivationOnly,
                            Fp8DiagnosticMode::BothRoundtrip}) {
        const auto rounds_weight =
            mode != Fp8DiagnosticMode::ActivationOnly;
        const auto rounds_activation =
            mode != Fp8DiagnosticMode::WeightOnly;
        auto config = tiny_config();
        config.linear_precision = LinearPrecision::Float8E4M3FNUZ;
        config.fp8_weight_scale = 1.0e-4F;
        config.fp8_activation_minimum_scale = 1.0e-4F;
        config.fp8_weight_scale_mode = Fp8WeightScaleMode::TensorAmax;
        config.fp8_activation_scale_mode = Fp8ActivationScaleMode::TensorAmax;
        config.fp8_diagnostic_mode = mode;
        TransformerModel model(config, 47);
        const auto before = model.forward_inference(tokens).to_vector();
        const auto report = model.prepare_fp8_inference_weights();
        EXPECT_EQ(report.linears_covered, 8U);
        EXPECT_EQ(report.converted_tensors, rounds_weight ? 8U : 0U);
        ops::clear_fp8_dynamic_quant_stats();
        const auto after = model.forward_inference(tokens).to_vector();
        EXPECT_EQ(after, before);
        EXPECT_EQ(ops::fp8_dynamic_quant_stats().tensor_calls,
                  rounds_activation ? 5U : 0U);
        std::size_t linear_weights = 0;
        for (const auto& [name, parameter] : model.named_parameters()) {
            if (name.ends_with(".weight") &&
                name.find("norm") == std::string::npos &&
                name != "token_embedding.weight") {
                EXPECT_EQ(parameter->data().dtype(),
                          rounds_weight ? DType::Float8E4M3FNUZ
                                        : DType::Float32)
                    << name;
                ++linear_weights;
            }
        }
        EXPECT_EQ(linear_weights, 8U);
        for (const auto value : after) EXPECT_TRUE(std::isfinite(value));
        EXPECT_THROW((void)model.forward(tokens), std::logic_error);
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
        EXPECT_EQ(model.cached_attention_split_sequence_splits(), 0);
        EXPECT_EQ(model.cached_attention_split_minimum_sequence(), 512);
        EXPECT_FALSE(model.cached_attention_materialized_scores_enabled());
        EXPECT_EQ(
            model.cached_attention_materialized_minimum_sequence(), 512);
        EXPECT_EQ(model.cached_attention_split_pv_splits(), 0);
        EXPECT_EQ(model.cached_attention_split_pv_minimum_sequence(), 512);
        EXPECT_THROW(model.set_cached_attention_split_sequence(-1, 1),
                     std::invalid_argument);
        EXPECT_THROW(model.set_cached_attention_split_sequence(33, 1),
                     std::invalid_argument);
        EXPECT_THROW(model.set_cached_attention_split_sequence(2, 0),
                     std::invalid_argument);
        EXPECT_THROW(model.set_cached_attention_materialized_scores(true, 0),
                     std::invalid_argument);
        EXPECT_THROW(model.set_cached_attention_split_pv(-1, 1),
                     std::invalid_argument);
        EXPECT_THROW(model.set_cached_attention_split_pv(33, 1),
                     std::invalid_argument);
        EXPECT_THROW(model.set_cached_attention_split_pv(2, 0),
                     std::invalid_argument);
        model.set_cached_attention_materialized_scores(true, 1);
        EXPECT_TRUE(model.cached_attention_materialized_scores_enabled());
        EXPECT_THROW(model.set_cached_attention_split_sequence(2, 1),
                     std::logic_error);
        EXPECT_THROW(model.set_cached_attention_split_pv(2, 1),
                     std::logic_error);
        model.set_cached_attention_materialized_scores(false, 1);
        model.set_cached_attention_split_sequence(2, 1);
        EXPECT_EQ(model.cached_attention_split_sequence_splits(), 2);
        EXPECT_EQ(model.cached_attention_split_minimum_sequence(), 1);
        EXPECT_THROW(model.set_cached_attention_split_pv(2, 1),
                     std::logic_error);
        model.set_cached_attention_split_sequence(0, 1);
        model.set_cached_attention_split_pv(2, 1);
        EXPECT_EQ(model.cached_attention_split_pv_splits(), 2);
        EXPECT_EQ(model.cached_attention_split_pv_minimum_sequence(), 1);
        EXPECT_THROW(model.set_cached_attention_materialized_scores(true, 1),
                     std::logic_error);
        EXPECT_THROW(model.set_cached_attention_split_sequence(2, 1),
                     std::logic_error);
        inference::KVCache split_pv_cache(
            model.config().layers, model.config().max_sequence_length);
        const auto split_pv_logits = model.forward_cached(
            Tensor::from_int32_vector({1}, {1, 1}), split_pv_cache).to_vector();
        model.set_cached_attention_split_pv(0, 1);
        model.set_cached_attention_materialized_scores(true, 1);
        inference::KVCache materialized_cache(
            model.config().layers, model.config().max_sequence_length);
        const auto materialized_logits = model.forward_cached(
            Tensor::from_int32_vector({1}, {1, 1}), materialized_cache).to_vector();
        EXPECT_EQ(split_pv_logits, materialized_logits);
        model.set_cached_attention_materialized_scores(false, 1);
        model.set_cached_attention_split_sequence(2, 1);
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

TEST(TransformerModelTest, BatchedPrefillAndDecodeMatchIndependentFullSequences) {
    auto config = tiny_config(true);
    config.max_sequence_length = 4;
    TransformerModel model(config, 43);
    inference::KVCache cache(config.layers, config.max_sequence_length, 2);
    const auto prefix = Tensor::from_int32_vector(
        {1, 2, 3, 4, 3, 2}, {2, 3});
    const auto prefilled = model.forward_prefill_cached(prefix, cache).to_vector();
    const auto full_prefix = model.forward_inference(prefix).to_vector();
    std::vector<float> expected_prefix;
    for (std::int64_t batch = 0; batch < 2; ++batch) {
        const auto offset = (batch * 3 + 2) * config.vocabulary_size;
        expected_prefix.insert(expected_prefix.end(), full_prefix.begin() + offset,
                               full_prefix.begin() + offset + config.vocabulary_size);
    }
    expect_near(prefilled, expected_prefix, 2.0e-5F);
    EXPECT_EQ(cache.position(), 3);
    EXPECT_EQ(cache.batch_size(), 2);
    EXPECT_EQ(cache.layer(0).key.shape(), (Shape{2, config.kv_heads, 3,
                                                 config.head_dimension()}));

    const auto continued = model.forward_cached(
        Tensor::from_int32_vector({4, 1}, {2, 1}), cache).to_vector();
    const auto full = model.forward_inference(Tensor::from_int32_vector(
        {1, 2, 3, 4, 4, 3, 2, 1}, {2, 4})).to_vector();
    std::vector<float> expected_continued;
    for (std::int64_t batch = 0; batch < 2; ++batch) {
        const auto offset = (batch * 4 + 3) * config.vocabulary_size;
        expected_continued.insert(expected_continued.end(), full.begin() + offset,
                                  full.begin() + offset + config.vocabulary_size);
    }
    expect_near(continued, expected_continued, 2.0e-5F);
}

TEST(TransformerModelTest, ClearCacheRowRemovesOldPrefixAndPreservesOtherRows) {
    auto config = tiny_config(true);
    config.max_sequence_length = 8;
    for (const auto dtype : {DType::Float32, DType::BFloat16}) {
        TransformerModel model(config, 45);
        inference::KVCache cache(config.layers, config.max_sequence_length, 2, dtype);
        const auto prefix = Tensor::from_int32_vector(
            {1, 2, 3, 4, 5, 6}, {2, 3});
        (void)model.forward_prefill_cached(prefix, cache);
        std::vector<std::vector<float>> preserved;
        for (std::size_t layer = 0; layer < cache.layer_count(); ++layer) {
            preserved.push_back(cache.layer(layer).key.slice(0, 1, 2).to_vector());
            preserved.push_back(cache.layer(layer).value.slice(0, 1, 2).to_vector());
        }
        EXPECT_THROW(cache.clear_row(-1), std::out_of_range);
        EXPECT_THROW(cache.clear_row(2), std::out_of_range);
        cache.clear_row(0);
        EXPECT_EQ(cache.position(), 3);
        for (std::size_t layer = 0; layer < cache.layer_count(); ++layer) {
            for (const auto* tensor : {&cache.layer(layer).key,
                                       &cache.layer(layer).value}) {
                const auto cleared = tensor->slice(0, 0, 1).to_vector();
                EXPECT_TRUE(std::all_of(cleared.begin(), cleared.end(),
                                        [](float value) { return value == 0.0F; }));
            }
            EXPECT_EQ(cache.layer(layer).key.slice(0, 1, 2).to_vector(),
                      preserved[layer * 2]);
            EXPECT_EQ(cache.layer(layer).value.slice(0, 1, 2).to_vector(),
                      preserved[layer * 2 + 1]);
        }

        (void)model.forward_cached(Tensor::from_int32_vector({7, 8}, {2, 1}), cache);
        EXPECT_EQ(cache.position(), 4);
        for (std::size_t layer = 0; layer < cache.layer_count(); ++layer) {
            for (const auto* tensor : {&cache.layer(layer).key,
                                       &cache.layer(layer).value}) {
                const auto cleared = tensor->slice(0, 0, 1).to_vector();
                bool wrote_new_position = false;
                for (std::int64_t head = 0; head < config.kv_heads; ++head) {
                    for (std::int64_t position = 0; position < 4; ++position) {
                        for (std::int64_t column = 0;
                             column < config.head_dimension(); ++column) {
                            const auto index = static_cast<std::size_t>(
                                (head * 4 + position) * config.head_dimension() + column);
                            if (position < 3) EXPECT_EQ(cleared[index], 0.0F);
                            else wrote_new_position |= cleared[index] != 0.0F;
                        }
                    }
                }
                EXPECT_TRUE(wrote_new_position);
            }
        }
        cache.reset();
        EXPECT_NO_THROW(cache.clear_row(0));
    }
}

TEST(TransformerModelTest, DivergentCachedRowsMatchIndependentB1References) {
    auto config = tiny_config(true);
    config.max_sequence_length = 8;
    for (const auto dtype : {DType::Float32, DType::BFloat16}) {
        TransformerModel batched(config, 157);
        inference::KVCache cache(config.layers, config.max_sequence_length, 2, dtype);
        const auto prefix = Tensor::from_int32_vector(
            {1, 2, 3, 4, 3, 2}, {2, 3});
        (void)batched.forward_prefill_cached(prefix, cache);
        const auto key_address = cache.layer(0).key.storage().data();
        cache.reset_row(0);
        EXPECT_EQ(cache.row_positions(), (std::vector<std::int64_t>{0, 3}));

        TransformerModel first(config, 157);
        inference::KVCache first_cache(config.layers, config.max_sequence_length, 1, dtype);
        auto first_expected = first.forward_cached(
            Tensor::from_int32_vector({7}, {1, 1}), first_cache);
        TransformerModel second(config, 157);
        inference::KVCache second_cache(config.layers, config.max_sequence_length, 1, dtype);
        (void)second.forward_prefill_cached(
            Tensor::from_int32_vector({4, 3, 2}, {1, 3}), second_cache);
        auto second_expected = second.forward_cached(
            Tensor::from_int32_vector({8}, {1, 1}), second_cache);

        const auto actual = batched.forward_cached_rows(
            Tensor::from_int32_vector({7, 8}, {2, 1}), cache);
        const auto tolerance = dtype == DType::Float32 ? 2.0e-5F : 5.0e-2F;
        expect_near(actual.slice(0, 0, 1).to_vector(),
                    first_expected.to_vector(), tolerance);
        expect_near(actual.slice(0, 1, 2).to_vector(),
                    second_expected.to_vector(), tolerance);
        EXPECT_EQ(cache.row_positions(), (std::vector<std::int64_t>{1, 4}));
        EXPECT_EQ(cache.layer(0).key.shape()[2], 4);
        EXPECT_EQ(cache.layer(0).key.storage().data(), key_address);

        first_expected = first.forward_cached(
            Tensor::from_int32_vector({9}, {1, 1}), first_cache);
        second_expected = second.forward_cached(
            Tensor::from_int32_vector({10}, {1, 1}), second_cache);
        const auto next = batched.forward_cached_rows(
            Tensor::from_int32_vector({9, 10}, {2, 1}), cache);
        expect_near(next.slice(0, 0, 1).to_vector(),
                    first_expected.to_vector(), tolerance);
        expect_near(next.slice(0, 1, 2).to_vector(),
                    second_expected.to_vector(), tolerance);
        EXPECT_EQ(cache.row_positions(), (std::vector<std::int64_t>{2, 5}));
        EXPECT_EQ(cache.layer(0).key.shape()[2], 5);
        cache.reset_row(1);
        EXPECT_EQ(cache.row_positions(), (std::vector<std::int64_t>{2, 0}));
        EXPECT_EQ(cache.layer(0).key.shape()[2], 2);
        EXPECT_EQ(cache.layer(0).key.storage().data(), key_address);
    }
}

TEST(TransformerModelTest, CachedRowsKeepUniformFastPathAndRejectMissingStorage) {
    auto config = tiny_config(true);
    config.max_sequence_length = 6;
    TransformerModel baseline(config, 163);
    TransformerModel rows(config, 163);
    inference::KVCache baseline_cache(config.layers, config.max_sequence_length, 2);
    inference::KVCache rows_cache(config.layers, config.max_sequence_length, 2);
    const auto prefix = Tensor::from_int32_vector({1, 2, 3, 4}, {2, 2});
    (void)baseline.forward_prefill_cached(prefix, baseline_cache);
    (void)rows.forward_prefill_cached(prefix, rows_cache);
    const auto tokens = Tensor::from_int32_vector({5, 6}, {2, 1});
    expect_near(rows.forward_cached_rows(tokens, rows_cache).to_vector(),
                baseline.forward_cached(tokens, baseline_cache).to_vector(), 2.0e-5F);
    EXPECT_TRUE(rows_cache.positions_uniform());

    inference::KVCache invalid(config.layers, config.max_sequence_length, 2);
    invalid.advance_row(0, 1);
    EXPECT_THROW((void)rows.forward_cached_rows(tokens, invalid),
                 std::invalid_argument);
}

TEST(TransformerModelTest, RowPrefillReplacesOnlyAnEmptySharedCacheSlot) {
    auto config = tiny_config(true);
    config.max_sequence_length = 8;
    for (const auto dtype : {DType::Float32, DType::BFloat16}) {
        TransformerModel batched(config, 167);
        inference::KVCache cache(config.layers, config.max_sequence_length, 2, dtype);
        (void)batched.forward_prefill_cached(
            Tensor::from_int32_vector({1, 2, 3, 4, 3, 2}, {2, 3}), cache);
        std::vector<std::vector<float>> preserved;
        for (std::size_t layer = 0; layer < cache.layer_count(); ++layer) {
            preserved.push_back(cache.layer(layer).key.slice(0, 1, 2).to_vector());
            preserved.push_back(cache.layer(layer).value.slice(0, 1, 2).to_vector());
        }
        const auto key_address = cache.layer(0).key.storage().data();
        cache.reset_row(0);

        TransformerModel first(config, 167);
        inference::KVCache first_cache(config.layers, config.max_sequence_length, 1, dtype);
        const auto expected = first.forward_prefill_cached(
            Tensor::from_int32_vector({7, 8}, {1, 2}), first_cache);
        const auto actual = batched.forward_prefill_cached_row(
            Tensor::from_int32_vector({7, 8}, {1, 2}), cache, 0);
        const auto tolerance = dtype == DType::Float32 ? 2.0e-5F : 5.0e-2F;
        expect_near(actual.to_vector(), expected.to_vector(), tolerance);
        EXPECT_EQ(cache.row_positions(), (std::vector<std::int64_t>{2, 3}));
        EXPECT_EQ(cache.layer(0).key.storage().data(), key_address);
        for (std::size_t layer = 0; layer < cache.layer_count(); ++layer) {
            EXPECT_EQ(cache.layer(layer).key.slice(0, 1, 2).to_vector(),
                      preserved[layer * 2]);
            EXPECT_EQ(cache.layer(layer).value.slice(0, 1, 2).to_vector(),
                      preserved[layer * 2 + 1]);
        }

        TransformerModel second(config, 167);
        inference::KVCache second_cache(config.layers, config.max_sequence_length, 1, dtype);
        (void)second.forward_prefill_cached(
            Tensor::from_int32_vector({4, 3, 2}, {1, 3}), second_cache);
        const auto first_next = first.forward_cached(
            Tensor::from_int32_vector({9}, {1, 1}), first_cache);
        const auto second_next = second.forward_cached(
            Tensor::from_int32_vector({10}, {1, 1}), second_cache);
        const auto next = batched.forward_cached_rows(
            Tensor::from_int32_vector({9, 10}, {2, 1}), cache);
        expect_near(next.slice(0, 0, 1).to_vector(), first_next.to_vector(), tolerance);
        expect_near(next.slice(0, 1, 2).to_vector(), second_next.to_vector(), tolerance);
        EXPECT_EQ(cache.row_positions(), (std::vector<std::int64_t>{3, 4}));
        EXPECT_THROW((void)batched.forward_prefill_cached_row(
                         Tensor::from_int32_vector({1}, {1, 1}), cache, 0),
                     std::invalid_argument);
        EXPECT_THROW((void)batched.forward_prefill_cached_row(
                         Tensor::from_int32_vector({1}, {1, 1}), cache, 2),
                     std::out_of_range);
    }

    TransformerModel fresh(config, 167);
    inference::KVCache fresh_cache(config.layers, config.max_sequence_length, 2);
    EXPECT_NO_THROW((void)fresh.forward_prefill_cached_row(
        Tensor::from_int32_vector({5, 6}, {1, 2}), fresh_cache, 1));
    EXPECT_EQ(fresh_cache.row_positions(), (std::vector<std::int64_t>{0, 2}));
}

TEST(TransformerModelTest, BatchedRowPrefillMapsEqualPromptsIntoEmptySlots) {
    auto config = tiny_config(true);
    config.max_sequence_length = 8;
    for (const auto dtype : {DType::Float32, DType::BFloat16}) {
        TransformerModel shared_model(config, 191);
        inference::KVCache shared(config.layers, config.max_sequence_length, 4,
                                  dtype);
        (void)shared_model.forward_prefill_cached_row(
            Tensor::from_int32_vector({1, 2, 3}, {1, 3}), shared, 0);
        std::vector<std::vector<float>> preserved;
        for (std::size_t layer = 0; layer < shared.layer_count(); ++layer) {
            preserved.push_back(shared.layer(layer).key.slice(0, 0, 1).to_vector());
            preserved.push_back(shared.layer(layer).value.slice(0, 0, 1).to_vector());
        }
        TransformerModel oracle(config, 191);
        inference::KVCache oracle_cache(config.layers,
                                         config.max_sequence_length, 2, dtype);
        const auto prompts = Tensor::from_int32_vector(
            {4, 5, 6, 7}, {2, 2});
        const auto expected = oracle.forward_prefill_cached(prompts, oracle_cache);
        const auto actual = shared_model.forward_prefill_cached_rows(
            prompts, shared, {1, 3});
        expect_near(actual.to_vector(), expected.to_vector(),
                    dtype == DType::Float32 ? 2.0e-5F : 5.0e-2F);
        EXPECT_EQ(shared.row_positions(),
                  (std::vector<std::int64_t>{3, 2, 0, 2}));
        for (std::size_t layer = 0; layer < shared.layer_count(); ++layer) {
            EXPECT_EQ(shared.layer(layer).key.slice(0, 0, 1).slice(2, 0, 3).to_vector(),
                      preserved[layer * 2]);
            EXPECT_EQ(shared.layer(layer).value.slice(0, 0, 1).slice(2, 0, 3).to_vector(),
                      preserved[layer * 2 + 1]);
        }
        EXPECT_THROW((void)shared_model.forward_prefill_cached_rows(
                         prompts, shared, {2, 2}),
                     std::invalid_argument);
        EXPECT_THROW((void)shared_model.forward_prefill_cached_rows(
                         prompts, shared, {3, 1}),
                     std::invalid_argument);
    }
}

TEST(TransformerModelTest, ActiveCachedRowsSkipInactiveStorageAndMatchB1) {
    auto config = tiny_config(true);
    config.max_sequence_length = 8;
    for (const auto dtype : {DType::Float32, DType::BFloat16}) {
        TransformerModel batched(config, 173);
        inference::KVCache cache(config.layers, config.max_sequence_length, 3, dtype);
        (void)batched.forward_prefill_cached_row(
            Tensor::from_int32_vector({1, 2}, {1, 2}), cache, 0);
        (void)batched.forward_prefill_cached_row(
            Tensor::from_int32_vector({3, 4, 5}, {1, 3}), cache, 1);
        cache.reset_row(2);
        const auto full_row = [&cache](const Tensor& tensor,
                                       std::int64_t row) {
            return Tensor::from_storage(
                       tensor.storage(),
                       {1, tensor.shape()[1], cache.max_sequence_length(),
                        tensor.shape()[3]},
                       tensor.strides(),
                       tensor.storage_offset() + row * tensor.stride(0),
                       tensor.dtype())
                .to_vector();
        };
        std::vector<std::vector<float>> inactive;
        for (std::size_t layer = 0; layer < cache.layer_count(); ++layer) {
            inactive.push_back(full_row(cache.layer(layer).key, 2));
            inactive.push_back(full_row(cache.layer(layer).value, 2));
        }
        const auto address = cache.layer(0).key.storage().data();

        TransformerModel first(config, 173);
        TransformerModel second(config, 173);
        inference::KVCache first_cache(config.layers, config.max_sequence_length, 1, dtype);
        inference::KVCache second_cache(config.layers, config.max_sequence_length, 1, dtype);
        (void)first.forward_prefill_cached(
            Tensor::from_int32_vector({1, 2}, {1, 2}), first_cache);
        (void)second.forward_prefill_cached(
            Tensor::from_int32_vector({3, 4, 5}, {1, 3}), second_cache);
        const auto first_logits = first.forward_cached(
            Tensor::from_int32_vector({9}, {1, 1}), first_cache);
        const auto second_logits = second.forward_cached(
            Tensor::from_int32_vector({10}, {1, 1}), second_cache);
        const auto actual = batched.forward_cached_active_rows(
            Tensor::from_int32_vector({9, 10}, {2, 1}), cache, {0, 1});
        const auto tolerance = dtype == DType::Float32 ? 2.0e-5F : 5.0e-2F;
        expect_near(actual.slice(0, 0, 1).to_vector(),
                    first_logits.to_vector(), tolerance);
        expect_near(actual.slice(0, 1, 2).to_vector(),
                    second_logits.to_vector(), tolerance);
        EXPECT_EQ(cache.row_positions(),
                  (std::vector<std::int64_t>{3, 4, 0}));
        EXPECT_EQ(cache.layer(0).key.storage().data(), address);
        for (std::size_t layer = 0; layer < cache.layer_count(); ++layer) {
            EXPECT_EQ(full_row(cache.layer(layer).key, 2),
                      inactive[layer * 2]);
            EXPECT_EQ(full_row(cache.layer(layer).value, 2),
                      inactive[layer * 2 + 1]);
        }
        EXPECT_THROW((void)batched.forward_cached_active_rows(
                         Tensor::from_int32_vector({}, {0, 1}), cache, {}),
                     std::invalid_argument);
        EXPECT_THROW((void)batched.forward_cached_active_rows(
                         Tensor::from_int32_vector({1, 2}, {2, 1}), cache,
                         {0, 0}),
                     std::invalid_argument);
        EXPECT_THROW((void)batched.forward_cached_active_rows(
                         Tensor::from_int32_vector({1, 2}, {2, 1}), cache,
                         {1, 0}),
                     std::invalid_argument);
        EXPECT_THROW((void)batched.forward_cached_active_rows(
                         Tensor::from_int32_vector({1}, {1, 1}), cache, {3}),
                     std::invalid_argument);
    }

    TransformerModel active_model(config, 179);
    TransformerModel uniform_model(config, 179);
    inference::KVCache active_cache(config.layers, config.max_sequence_length, 2);
    inference::KVCache uniform_cache(config.layers, config.max_sequence_length, 2);
    const auto prompts = Tensor::from_int32_vector({1, 2, 3, 4}, {2, 2});
    (void)active_model.forward_prefill_cached(prompts, active_cache);
    (void)uniform_model.forward_prefill_cached(prompts, uniform_cache);
    const auto tokens = Tensor::from_int32_vector({5, 6}, {2, 1});
    expect_near(active_model.forward_cached_active_rows(
                    tokens, active_cache, {0, 1}).to_vector(),
                uniform_model.forward_cached(tokens, uniform_cache).to_vector(),
                2.0e-5F);
}

TEST(KVCacheTest, PerRowPositionsRejectAmbiguousUniformReads) {
    inference::KVCache cache(2, 6, 3, DType::Float32);
    EXPECT_TRUE(cache.positions_uniform());
    EXPECT_EQ(cache.position(), 0);
    EXPECT_EQ(cache.row_positions(), (std::vector<std::int64_t>{0, 0, 0}));
    cache.advance_row(0, 2);
    EXPECT_FALSE(cache.positions_uniform());
    EXPECT_EQ(cache.row_position(0), 2);
    EXPECT_EQ(cache.row_position(1), 0);
    EXPECT_THROW((void)cache.position(), std::logic_error);
    EXPECT_THROW(cache.advance_row(-1), std::out_of_range);
    EXPECT_THROW(cache.advance_row(3), std::out_of_range);
    EXPECT_THROW(cache.advance_row(0, 5), std::out_of_range);
    EXPECT_THROW(cache.advance_row(1, 0), std::out_of_range);
    cache.reset_row(0);
    EXPECT_TRUE(cache.positions_uniform());
    EXPECT_EQ(cache.position(), 0);
    cache.advance(3);
    EXPECT_EQ(cache.row_positions(), (std::vector<std::int64_t>{3, 3, 3}));
    cache.reset_row(1);
    EXPECT_EQ(cache.row_positions(), (std::vector<std::int64_t>{3, 0, 3}));
    EXPECT_THROW((void)cache.position(), std::logic_error);
    cache.advance_row(1, 3);
    EXPECT_TRUE(cache.positions_uniform());
    EXPECT_EQ(cache.position(), 3);
    cache.reset();
    EXPECT_EQ(cache.row_positions(), (std::vector<std::int64_t>{0, 0, 0}));
}

TEST(TransformerModelTest, Bf16KvCacheHalvesStorageAndTracksFp32Decode) {
    auto config = tiny_config(true);
    config.max_sequence_length = 8;
    TransformerModel fp32_model(config, 47);
    TransformerModel bf16_model(config, 47);
    inference::KVCache fp32_cache(config.layers, config.max_sequence_length, 2,
                                  DType::Float32);
    inference::KVCache bf16_cache(config.layers, config.max_sequence_length, 2,
                                  DType::BFloat16);
    EXPECT_THROW((inference::KVCache(config.layers, config.max_sequence_length, 2,
                                     DType::Float16)),
                 std::invalid_argument);
    const auto prefix = Tensor::from_int32_vector(
        {1, 2, 3, 4, 3, 2, 1, 2}, {2, 4});
    const auto fp32_prefill = fp32_model.forward_prefill_cached(prefix, fp32_cache);
    const auto bf16_prefill = bf16_model.forward_prefill_cached(prefix, bf16_cache);
    EXPECT_EQ(bf16_cache.dtype(), DType::BFloat16);
    EXPECT_EQ(bf16_cache.layer(0).key.dtype(), DType::BFloat16);
    EXPECT_EQ(bf16_cache.layer(0).key.storage().num_bytes() * 2U,
              fp32_cache.layer(0).key.storage().num_bytes());
    expect_near(bf16_prefill.to_vector(), fp32_prefill.to_vector(), 2.0e-2F);

    const auto next = Tensor::from_int32_vector({4, 1}, {2, 1});
    const auto fp32_decode = fp32_model.forward_cached(next, fp32_cache);
    const auto bf16_decode = bf16_model.forward_cached(next, bf16_cache);
    expect_near(bf16_decode.to_vector(), fp32_decode.to_vector(), 2.0e-2F);
    EXPECT_EQ(ops::argmax_last_dim(bf16_decode).to_int32_vector(),
              ops::argmax_last_dim(fp32_decode).to_int32_vector());
}

TEST(TransformerModelTest, MixedLayerKvCacheUsesEachLayerPolicy) {
    auto config = tiny_config(true);
    config.layers = 2;
    config.max_sequence_length = 6;
    TransformerModel reference_model(config, 53);
    TransformerModel mixed_model(config, 53);
    inference::KVCache reference_cache(config.layers, config.max_sequence_length);
    inference::KVCache mixed_cache(
        {DType::BFloat16, DType::Float32}, config.max_sequence_length);
    EXPECT_TRUE(mixed_cache.has_mixed_dtypes());
    EXPECT_THROW((void)mixed_cache.dtype(), std::logic_error);
    EXPECT_EQ(mixed_cache.layer_dtype(0), DType::BFloat16);
    EXPECT_EQ(mixed_cache.layer_dtype(1), DType::Float32);

    const auto prefix = Tensor::from_int32_vector({1, 2, 3}, {1, 3});
    const auto reference = reference_model.forward_prefill_cached(
        prefix, reference_cache);
    const auto mixed = mixed_model.forward_prefill_cached(prefix, mixed_cache);
    EXPECT_EQ(mixed_cache.layer(0).key.dtype(), DType::BFloat16);
    EXPECT_EQ(mixed_cache.layer(1).key.dtype(), DType::Float32);
    EXPECT_EQ(mixed_cache.layer(0).key.storage().num_bytes() * 2U,
              mixed_cache.layer(1).key.storage().num_bytes());
    expect_near(mixed.to_vector(), reference.to_vector(), 2.0e-2F);

    const auto next = Tensor::from_int32_vector({4}, {1, 1});
    expect_near(mixed_model.forward_cached(next, mixed_cache).to_vector(),
                reference_model.forward_cached(next, reference_cache).to_vector(),
                2.0e-2F);
    EXPECT_THROW((inference::KVCache(std::vector<DType>{}, 4)),
                 std::invalid_argument);
    EXPECT_THROW((inference::KVCache(-1, 4)), std::invalid_argument);
    EXPECT_THROW((inference::KVCache({DType::Float16}, 4)),
                 std::invalid_argument);
}

}  // namespace microllm::model
