#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <map>
#include <string>
#include <vector>

#include <gtest/gtest.h>
#include <microllm/model/model.h>

namespace microllm::model {
namespace {

ModelConfig weight_config(bool tied = false) {
    return {.vocabulary_size = 8,
            .dimension = 8,
            .layers = 1,
            .heads = 2,
            .kv_heads = 1,
            .ffn_dimension = 16,
            .max_sequence_length = 4,
            .rope_base = 10000.0F,
            .tie_embeddings = tied};
}

class TemporaryDirectory {
public:
    TemporaryDirectory() {
        path_ = std::filesystem::temp_directory_path() /
                ("microllm-model-weights-" + std::to_string(
                    std::chrono::steady_clock::now().time_since_epoch().count()));
        std::filesystem::create_directories(path_);
    }
    ~TemporaryDirectory() {
        std::error_code ignored;
        std::filesystem::remove_all(path_, ignored);
    }
    [[nodiscard]] const std::filesystem::path& path() const noexcept { return path_; }

private:
    std::filesystem::path path_;
};

void expect_state_equal(const io::StateDict& actual, const io::StateDict& expected,
                        float tolerance = 0.0F) {
    ASSERT_EQ(actual.size(), expected.size());
    for (const auto& [name, tensor] : expected) {
        ASSERT_TRUE(actual.contains(name)) << name;
        EXPECT_EQ(actual.at(name).shape(), tensor.shape()) << name;
        const auto left = actual.at(name).to_vector();
        const auto right = tensor.to_vector();
        ASSERT_EQ(left.size(), right.size()) << name;
        for (std::size_t index = 0; index < left.size(); ++index) {
            EXPECT_NEAR(left[index], right[index], tolerance) << name << " index=" << index;
        }
    }
}

Tensor tokens() { return Tensor::from_int32_vector({0, 1, 2, 3}, {1, 4}); }

ModelConfig moe_weight_config(std::int64_t layers, std::int64_t num_experts) {
    return {.vocabulary_size = 8,
            .dimension = 8,
            .layers = layers,
            .heads = 2,
            .kv_heads = 1,
            .ffn_dimension = 16,
            .max_sequence_length = 4,
            .rope_base = 10000.0F,
            .tie_embeddings = true,
            .moe_num_experts = num_experts,
            .moe_num_experts_per_tok = 1,
            .moe_intermediate_size = 4};
}

}  // namespace

TEST(ModelWeightsTest, StateDictIsAnIndependentNamedSnapshot) {
    TransformerModel model(weight_config(), 301);
    const auto before = model.state_dict();
    auto snapshot = model.state_dict();
    ASSERT_EQ(snapshot.size(), model.named_parameters().size());
    snapshot.begin()->second.fill(99.0F);
    const auto after = model.state_dict();
    expect_state_equal(after, before);
}

TEST(ModelWeightsTest, StrictLoadIsAtomicAndNonStrictLoadReportsEveryProblem) {
    TransformerModel source(weight_config(), 307);
    TransformerModel target(weight_config(), 311);
    auto broken = source.state_dict();
    broken.erase("token_embedding.weight");
    broken["final_norm.weight"] = Tensor::from_vector({1}, {1});
    broken.emplace("unused.weight", Tensor::from_vector({5}, {1}));
    const auto before = target.state_dict();

    EXPECT_THROW((void)target.load_state_dict(broken), std::invalid_argument);
    expect_state_equal(target.state_dict(), before);

    const auto report = target.load_state_dict(
        broken, {.strict = false, .mapping = {}, .aliases = {}});
    EXPECT_FALSE(report.complete());
    EXPECT_EQ(report.missing.size(), 1U);
    EXPECT_EQ(report.unexpected.size(), 1U);
    EXPECT_EQ(report.incompatible.size(), 1U);
    EXPECT_EQ(report.loaded.size(), target.named_parameters().size() - 2U);
}

TEST(ModelWeightsTest, StrictLoadClearsGradientsAndReproducesForward) {
    TransformerModel source(weight_config(), 313);
    TransformerModel target(weight_config(), 317);
    const auto targets = Tensor::from_int32_vector({1, 2, 3, 0}, {1, 4});
    target.loss(tokens(), targets).backward();
    ASSERT_TRUE(target.parameters()[0]->has_grad());
    auto source_state = source.state_dict();
    const auto report = target.load_state_dict(source_state);
    EXPECT_TRUE(report.complete());
    EXPECT_EQ(report.loaded.size(), source.named_parameters().size());
    for (const auto* parameter : target.parameters()) EXPECT_FALSE(parameter->has_grad());
    EXPECT_EQ(target.forward(tokens()).data().to_vector(),
              source.forward(tokens()).data().to_vector());
    source_state.begin()->second.fill(99.0F);
    expect_state_equal(target.state_dict(), source.state_dict());
}

TEST(ModelWeightsTest, PreparedBf16FfnExportsFp32SnapshotAndRejectsReload) {
    TransformerModel model(weight_config(), 319);
    const auto before = model.state_dict();
    const auto report = model.prepare_bf16_ffn_inference();
    EXPECT_EQ(report.converted_tensors, 3U);
    const auto snapshot = model.state_dict();
    ASSERT_EQ(snapshot.size(), before.size());
    for (const auto& [name, tensor] : snapshot) {
        EXPECT_EQ(tensor.dtype(), DType::Float32) << name;
        const auto expected = name.find("feed_forward") == std::string::npos
                                  ? before.at(name)
                                  : before.at(name).cast(DType::BFloat16)
                                        .cast(DType::Float32);
        EXPECT_EQ(tensor.to_vector(), expected.to_vector()) << name;
    }
    model.to(Device::cpu());
    EXPECT_TRUE(model.bf16_ffn_inference_prepared());
    for (const auto& [name, parameter] : model.named_parameters()) {
        if (name.find("feed_forward") != std::string::npos) {
            EXPECT_EQ(parameter->data().dtype(), DType::BFloat16) << name;
        }
    }
    EXPECT_THROW((void)model.load_state_dict(before), std::logic_error);
}

TEST(ModelWeightsTest, QwenStyleMappingTransposesLinearWeights) {
    const auto config = weight_config();
    TransformerModel source(config, 331);
    TransformerModel target(config, 337, ParameterInitialization::Uninitialized);
    EXPECT_THROW((void)target.forward(tokens()), std::logic_error);
    const auto native = source.state_dict();
    const auto mapping = qwen_style_weight_mapping(config);
    io::StateDict external;
    for (const auto& [target_name, source_spec] : mapping) {
        auto tensor = native.at(target_name);
        if (source_spec.transform == WeightTransform::Transpose2D) {
            tensor = tensor.transpose(0, 1).contiguous();
        }
        external.emplace(source_spec.name,
                         Tensor::from_vector(tensor.to_vector(), tensor.shape()));
    }
    const auto report = target.load_state_dict(
        external, {.strict = true, .mapping = mapping, .aliases = {}});
    EXPECT_TRUE(report.complete());
    expect_state_equal(target.state_dict(), native);
}

TEST(ModelWeightsTest, QwenStyleMappingIncludesAttentionBiasWhenConfigured) {
    auto config = weight_config(false);
    config.attention_bias = true;
    const auto mapping = qwen_style_weight_mapping(config);
    EXPECT_EQ(mapping.at("blocks.0.attention.q_proj.bias").name,
              "model.layers.0.self_attn.q_proj.bias");
    EXPECT_EQ(mapping.at("blocks.0.attention.k_proj.bias").name,
              "model.layers.0.self_attn.k_proj.bias");
    EXPECT_EQ(mapping.at("blocks.0.attention.v_proj.bias").name,
              "model.layers.0.self_attn.v_proj.bias");
    EXPECT_EQ(mapping.at("blocks.0.attention.q_proj.bias").transform,
              WeightTransform::Identity);
}

TEST(ModelWeightsTest, QwenStyleMappingIncludesQkNormWhenConfigured) {
    auto config = weight_config(false);
    config.attention_head_dimension = 6;
    config.qk_norm = true;
    const auto mapping = qwen_style_weight_mapping(config);
    EXPECT_EQ(mapping.at("blocks.0.attention.q_norm.weight").name,
              "model.layers.0.self_attn.q_norm.weight");
    EXPECT_EQ(mapping.at("blocks.0.attention.k_norm.weight").name,
              "model.layers.0.self_attn.k_norm.weight");
    EXPECT_EQ(mapping.at("blocks.0.attention.q_norm.weight").transform,
              WeightTransform::Identity);
}

TEST(ModelWeightsTest, StrictTiedAliasMustExactlyMatchPrimarySource) {
    auto config = weight_config(true);
    config.attention_head_dimension = 6;
    config.qk_norm = true;
    TransformerModel source(config, 709);
    const auto native = source.state_dict();
    const auto mapping = qwen_style_weight_mapping(config);
    io::StateDict external;
    for (const auto& [target, spec] : mapping) {
        auto tensor = native.at(target);
        if (spec.transform == WeightTransform::Transpose2D) {
            tensor = tensor.transpose(0, 1).contiguous();
        }
        external.emplace(spec.name, Tensor::from_vector(
            tensor.to_vector(), tensor.shape()));
    }
    external.emplace("lm_head.weight", Tensor::from_vector(
        external.at("model.embed_tokens.weight").to_vector(),
        external.at("model.embed_tokens.weight").shape()));
    const LoadWeightsOptions options{
        .strict = true, .mapping = mapping,
        .aliases = qwen3_tied_weight_aliases(config)};
    TransformerModel target(
        config, 711, ParameterInitialization::Uninitialized);
    EXPECT_TRUE(target.load_state_dict(external, options).complete());
    expect_state_equal(target.state_dict(), native);

    auto changed = external.at("lm_head.weight").to_vector();
    changed[0] += 1.0F;
    external.at("lm_head.weight") = Tensor::from_vector(
        changed, external.at("lm_head.weight").shape());
    TransformerModel rejected(
        config, 713, ParameterInitialization::Uninitialized);
    EXPECT_THROW((void)rejected.load_state_dict(external, options),
                 std::invalid_argument);
    EXPECT_THROW((void)rejected.forward(tokens()), std::logic_error);
}

TEST(ModelWeightsTest, LoadsSingleAndIndexedSafetensorsFiles) {
    TemporaryDirectory directory;
    TransformerModel source(weight_config(), 347);
    const auto state = source.state_dict();
    const auto single_path = directory.path() / "model.safetensors";
    source.save_safetensors(single_path);
    TransformerModel single_target(weight_config(), 349);
    EXPECT_TRUE(single_target.load_safetensors(single_path).complete());
    expect_state_equal(single_target.state_dict(), state);

    TransformerModel files_target(weight_config(), 351);
    // The multi-file model API is exercised below after the two shards are written.

    io::StateDict first;
    io::StateDict second;
    bool alternate = false;
    for (const auto& item : state) {
        (alternate ? first : second).insert(item);
        alternate = !alternate;
    }
    io::save_safetensors(directory.path() / "model-00001.safetensors", first);
    io::save_safetensors(directory.path() / "model-00002.safetensors", second);
    EXPECT_TRUE(files_target.load_safetensors_files(
        {directory.path() / "model-00001.safetensors",
         directory.path() / "model-00002.safetensors"}).complete());
    expect_state_equal(files_target.state_dict(), state);
    const auto index_path = directory.path() / "model.safetensors.index.json";
    std::ofstream index(index_path);
    index << "{\"weight_map\":{";
    bool first_name = true;
    for (const auto& [name, tensor] : state) {
        (void)tensor;
        if (!first_name) index << ',';
        first_name = false;
        index << '"' << name << "\":\""
              << (first.contains(name) ? "model-00001.safetensors"
                                       : "model-00002.safetensors")
              << '"';
    }
    index << "}}";
    index.close();
    TransformerModel indexed_target(weight_config(), 353);
    EXPECT_TRUE(indexed_target.load_safetensors_index(index_path).complete());
    expect_state_equal(indexed_target.state_dict(), state);
}

TEST(ModelWeightsTest, QwenMappingOmitsOutputHeadForTiedEmbeddings) {
    const auto mapping = qwen_style_weight_mapping(weight_config(true));
    EXPECT_TRUE(mapping.contains("token_embedding.weight"));
    EXPECT_FALSE(mapping.contains("output_head.weight"));
}

TEST(ModelWeightsTest, RejectsUnknownMappingTargetsAndNonFloatSources) {
    TransformerModel model(weight_config(), 359);
    auto state = model.state_dict();
    state["token_embedding.weight"] =
        Tensor::from_int32_vector(std::vector<std::int32_t>(64, 1), {8, 8});
    LoadWeightsOptions options;
    options.strict = false;
    options.mapping.emplace("not_a_parameter", WeightSource{"anything", WeightTransform::Identity});
    const auto report = model.load_state_dict(state, options);
    EXPECT_EQ(report.incompatible.size(), 2U);
    EXPECT_EQ(report.loaded.size(), model.named_parameters().size() - 1U);

    options.strict = true;
    EXPECT_THROW((void)model.load_state_dict(state, options), std::invalid_argument);
}

TEST(ModelWeightsTest, MoeStateDictHasExactPerExpertTensorCountAndMatchesMapping) {
    // M7 corrected internal MoE storage back to per-expert separate tensors
    // after downloading and inspecting an actual Qwen3-MoE checkpoint (M6 had
    // briefly assumed a fused gate_up_proj layout after reading only
    // transformers' current in-memory module source, which turned out not to
    // match what real checkpoints -- both a downloaded tiny one and the
    // official Qwen/Qwen3-30B-A3B's safetensors index -- actually serialize).
    // See docs/development/2026-09-05-m7-qwen3-moe-real-checkpoint.md.
    const auto config = moe_weight_config(/*layers=*/2, /*num_experts=*/3);
    TransformerModel model(config, 401);
    // Per layer: attention_norm(1) + attention q/k/v/o(4, no bias/qk_norm) +
    // ffn_norm(1) + moe.router(1) + 3 experts * 3 projections(9) = 16.
    // Tied embeddings: + token_embedding(1) + final_norm(1), no output_head.
    // A silently-dropped expert would show up here as an off-by-3 mismatch,
    // not a vague "fewer tensors than expected."
    constexpr std::size_t per_layer = 1 + 4 + 1 + 1 + 3 * 3;
    const std::size_t expected = per_layer * 2 + 2;
    EXPECT_EQ(model.named_parameters().size(), expected);
    EXPECT_EQ(model.state_dict().size(), expected);

    const auto mapping = qwen_style_weight_mapping(config);
    EXPECT_EQ(mapping.size(), expected);
    EXPECT_TRUE(mapping.contains("blocks.0.moe.router.weight"));
    EXPECT_TRUE(mapping.contains("blocks.1.moe.experts.2.down_proj.weight"));
    EXPECT_FALSE(mapping.contains("blocks.0.moe.experts.3.gate_proj.weight"));
    EXPECT_FALSE(mapping.contains("blocks.0.moe.gate_up_proj"));
    EXPECT_FALSE(mapping.contains("blocks.0.feed_forward.gate_proj.weight"));
}

TEST(ModelWeightsTest, MoeStrictLoadRejectsMissingUnexpectedAndIncompatibleExpertTensors) {
    const auto config = moe_weight_config(/*layers=*/1, /*num_experts=*/3);
    TransformerModel source(config, 403);
    TransformerModel target(config, 409);
    auto broken = source.state_dict();
    // Missing: a silently-dropped expert tensor is exactly the failure mode
    // this milestone's exact-count assertion exists to catch.
    broken.erase("blocks.0.moe.experts.1.down_proj.weight");
    // Incompatible: right name, wrong shape.
    broken["blocks.0.moe.experts.2.up_proj.weight"] = Tensor::from_vector({1}, {1});
    // Unexpected: a fourth expert nobody asked for.
    broken.emplace("blocks.0.moe.experts.3.gate_proj.weight", Tensor::from_vector({1}, {1}));
    const auto before = target.state_dict();

    EXPECT_THROW((void)target.load_state_dict(broken), std::invalid_argument);
    expect_state_equal(target.state_dict(), before);

    const auto report = target.load_state_dict(
        broken, {.strict = false, .mapping = {}, .aliases = {}});
    EXPECT_FALSE(report.complete());
    EXPECT_EQ(report.missing.size(), 1U);
    EXPECT_EQ(report.unexpected.size(), 1U);
    EXPECT_EQ(report.incompatible.size(), 1U);
    EXPECT_EQ(report.loaded.size(), target.named_parameters().size() - 2U);
}

TEST(ModelWeightsTest, MoeForwardProducesFiniteLogitsAndTrainsEveryParameter) {
    // M6: forward is real now. Checks the whole graph runs end to end
    // (embedding -> attention -> MoE router/expert_ffn/combine -> logits ->
    // loss -> backward) and every MoE parameter -- router and every expert's
    // gate/up/down -- receives a gradient.
    const auto config = moe_weight_config(/*layers=*/1, /*num_experts=*/2);
    TransformerModel model(config, 419);
    const auto logits_value = model.forward(tokens());
    EXPECT_EQ(logits_value.data().shape(), (Shape{1, 4, config.vocabulary_size}));
    for (const auto value : logits_value.data().to_vector()) {
        EXPECT_TRUE(std::isfinite(value));
    }
    const auto inference_logits = model.forward_inference(tokens());
    EXPECT_EQ(inference_logits.shape(), logits_value.data().shape());

    const auto targets = Tensor::from_int32_vector({1, 2, 3, 0}, {1, 4});
    model.loss(tokens(), targets).backward();
    for (const auto& [name, parameter] : model.named_parameters()) {
        EXPECT_TRUE(parameter->has_grad()) << name;
    }
}

TEST(ModelWeightsTest, MoeAdvancedInferencePreparationIsStillExplicitlyUnimplemented) {
    // Forward is implemented (M6), but BF16/FP8/INT8 MoE support is not --
    // each must still fail loudly rather than silently computing a wrong or
    // partial result.
    TransformerModel model(moe_weight_config(/*layers=*/1, /*num_experts=*/2), 423);
    EXPECT_THROW((void)model.prepare_bf16_ffn_inference(), std::logic_error);

    auto fp8_config = moe_weight_config(/*layers=*/1, /*num_experts=*/2);
    fp8_config.linear_precision = LinearPrecision::Float8E4M3FNUZ;
    TransformerModel fp8_model(fp8_config, 429);
    EXPECT_THROW((void)fp8_model.prepare_fp8_inference_weights(), std::logic_error);
}

}  // namespace microllm::model
