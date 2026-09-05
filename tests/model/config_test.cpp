#include <filesystem>
#include <fstream>

#include <gtest/gtest.h>
#include <microllm/model/config.h>
#include <microllm/model/huggingface.h>
#include <microllm/model/model.h>

namespace microllm::model {

TEST(ModelConfigTest, ModelSHasExecutableParameterBudget) {
    const auto config = ModelConfig::model_s();
    EXPECT_EQ(config.head_dimension(), 64);
    EXPECT_EQ(config.parameter_count(), 15'586'176U);
    EXPECT_EQ(config.weight_bytes(4), 62'344'704U);
}

TEST(ModelConfigTest, ModelMTargetsTheSecondWeightTier) {
    const auto config = ModelConfig::model_m();
    EXPECT_EQ(config.parameter_count(), 31'334'912U);
    EXPECT_EQ(config.weight_bytes(4), 125'339'648U);
}

TEST(ModelConfigTest, GqaBudgetUsesReducedKeyValueProjection) {
    auto config = ModelConfig::model_s();
    const auto mha = config.parameter_count();
    config.kv_heads = 2;
    EXPECT_LT(config.parameter_count(), mha);
    EXPECT_EQ(config.kv_dimension(), 128);
}

TEST(ModelConfigTest, Bf16LinearPolicyIsVisibleInSummary) {
    auto config = ModelConfig::model_s();
    config.linear_precision = LinearPrecision::BFloat16;
    config.validate();
    EXPECT_NE(config.summary().find("linear_precision=bf16_fp32_master"),
              std::string::npos);
}

TEST(ModelConfigTest, Fp8TensorAmaxPolicyIsVisibleInSummary) {
    auto config = ModelConfig::model_s();
    config.linear_precision = LinearPrecision::Float8E4M3FNUZ;
    config.fp8_weight_scale_mode = Fp8WeightScaleMode::TensorAmax;
    config.fp8_activation_scale_mode = Fp8ActivationScaleMode::TensorAmax;
    config.validate();
    EXPECT_NE(config.summary().find("fp8_weight_scale_mode=tensor_amax"),
              std::string::npos);
    EXPECT_NE(config.summary().find("fp8_activation_scale_mode=tensor_amax"),
              std::string::npos);
}

TEST(ModelConfigTest, Fp8FfnOuterRowPolicyIsVisibleInSummary) {
    auto config = ModelConfig::model_s();
    config.linear_precision = LinearPrecision::Float8E4M3FNUZ;
    config.fp8_activation_scale_mode = Fp8ActivationScaleMode::FfnOuterRow;
    config.validate();
    EXPECT_NE(config.summary().find("fp8_activation_scale_mode=ffn_outer_row"),
              std::string::npos);
}

TEST(ModelConfigTest, Fp8DeviceWeightAmaxPolicyIsVisibleInSummary) {
    auto config = ModelConfig::model_s();
    config.linear_precision = LinearPrecision::Float8E4M3FNUZ;
    config.fp8_weight_scale_mode = Fp8WeightScaleMode::DeviceTensorAmax;
    config.validate();
    EXPECT_NE(config.summary().find("fp8_weight_scale_mode=device_tensor_amax"),
              std::string::npos);
}

TEST(ModelConfigTest, Fp8OutputChannelWeightPolicyIsVisibleInSummary) {
    auto config = ModelConfig::model_s();
    config.linear_precision = LinearPrecision::Float8E4M3FNUZ;
    config.fp8_weight_scale_mode = Fp8WeightScaleMode::OutputChannelAmax;
    config.validate();
    EXPECT_NE(config.summary().find("fp8_weight_scale_mode=output_channel_amax"),
              std::string::npos);
    config.fp8_weight_scale_scope = Fp8WeightScaleScope::AttentionOutputOnly;
    config.validate();
    EXPECT_NE(config.summary().find("fp8_weight_scale_scope=attention_output_only"),
              std::string::npos);
    config.fp8_weight_scale_mode = Fp8WeightScaleMode::DeviceTensorAmax;
    EXPECT_THROW(config.validate(), std::invalid_argument);
}

TEST(ModelConfigTest, Fp8DiagnosticModesAreExplicitAndRequireFp8Linear) {
    auto config = ModelConfig::model_s();
    config.linear_precision = LinearPrecision::Float8E4M3FNUZ;
    config.fp8_diagnostic_mode = Fp8DiagnosticMode::WeightOnly;
    config.validate();
    EXPECT_NE(config.summary().find("fp8_diagnostic_mode=weight_only"),
              std::string::npos);
    config.fp8_diagnostic_mode = Fp8DiagnosticMode::ActivationOnly;
    EXPECT_NE(config.summary().find("fp8_diagnostic_mode=activation_only"),
              std::string::npos);
    config.fp8_diagnostic_mode = Fp8DiagnosticMode::BothRoundtrip;
    EXPECT_NE(config.summary().find("fp8_diagnostic_mode=both_roundtrip"),
              std::string::npos);
    config.linear_precision = LinearPrecision::Float32;
    EXPECT_THROW(config.validate(), std::invalid_argument);
}

TEST(ModelConfigTest, Fp8Fp32LayerOverridesAreStrictlyIncreasingAndInRange) {
    auto config = ModelConfig::model_s();
    config.linear_precision = LinearPrecision::Float8E4M3FNUZ;
    config.fp8_fp32_layers = {1, 4};
    config.validate();
    EXPECT_NE(config.summary().find("fp8_fp32_layers=1:4"), std::string::npos);
    config.fp8_fp32_layers = {1, 1};
    EXPECT_THROW(config.validate(), std::invalid_argument);
    config.fp8_fp32_layers = {config.layers};
    EXPECT_THROW(config.validate(), std::invalid_argument);
}

TEST(ModelConfigTest, MoeFieldsAreExplicitAndConsistent) {
    auto config = ModelConfig::model_s();
    config.validate();
    EXPECT_NE(config.summary().find("moe_num_experts=0"), std::string::npos);
    EXPECT_NE(config.summary().find("moe_num_experts_per_tok=0"), std::string::npos);
    EXPECT_NE(config.summary().find("moe_intermediate_size=0"), std::string::npos);
    EXPECT_NE(config.summary().find("moe_norm_topk_prob=false"), std::string::npos);

    config.moe_num_experts = 4;
    config.moe_num_experts_per_tok = 2;
    config.moe_intermediate_size = 32;
    config.moe_norm_topk_prob = true;
    config.validate();
    EXPECT_NE(config.summary().find("moe_num_experts=4"), std::string::npos);
    EXPECT_NE(config.summary().find("moe_norm_topk_prob=true"), std::string::npos);
    // parameter_count()/weight_bytes() intentionally do not support MoE yet:
    // the exact per-expert tensor layout is a weight-loading decision, not a
    // parsing one, and reporting a wrong dense-only count would be worse than
    // refusing to answer.
    EXPECT_THROW((void)config.parameter_count(), std::invalid_argument);

    auto missing_per_tok = config;
    missing_per_tok.moe_num_experts_per_tok = 0;
    EXPECT_THROW(missing_per_tok.validate(), std::invalid_argument);

    auto too_many_per_tok = config;
    too_many_per_tok.moe_num_experts_per_tok = 5;
    EXPECT_THROW(too_many_per_tok.validate(), std::invalid_argument);

    auto missing_intermediate = config;
    missing_intermediate.moe_intermediate_size = 0;
    EXPECT_THROW(missing_intermediate.validate(), std::invalid_argument);

    auto dense_with_leftover_field = ModelConfig::model_s();
    dense_with_leftover_field.moe_num_experts_per_tok = 2;
    EXPECT_THROW(dense_with_leftover_field.validate(), std::invalid_argument);
}

TEST(ModelConfigTest, RejectsInvalidHeadAndRopeConfigurations) {
    auto config = ModelConfig::model_s();
    config.dimension = 383;
    EXPECT_THROW(config.validate(), std::invalid_argument);
    config = ModelConfig::model_s();
    config.kv_heads = 4;
    EXPECT_THROW(config.validate(), std::invalid_argument);
    config = ModelConfig::model_s();
    config.linear_precision = LinearPrecision::Float8E4M3FNUZ;
    config.fp8_activation_scale = 0.0F;
    EXPECT_THROW(config.validate(), std::invalid_argument);
    config.fp8_activation_scale = 0.025F;
    config.fp8_activation_minimum_scale = 0.0F;
    EXPECT_THROW(config.validate(), std::invalid_argument);
}

TEST(ModelConfigTest, ExplicitHeadDimensionAndQkNormChangeProjectionBudget) {
    ModelConfig config{.vocabulary_size = 16,
                       .dimension = 8,
                       .layers = 1,
                       .heads = 2,
                       .kv_heads = 1,
                       .attention_head_dimension = 6,
                       .ffn_dimension = 16,
                       .max_sequence_length = 8,
                       .qk_norm = true};
    config.validate();
    EXPECT_EQ(config.head_dimension(), 6);
    EXPECT_EQ(config.query_dimension(), 12);
    EXPECT_EQ(config.kv_dimension(), 6);
    EXPECT_EQ(config.parameter_count(), 964U);
    EXPECT_NE(config.summary().find("head_dim=6"), std::string::npos);
    EXPECT_NE(config.summary().find("qk_norm=true"), std::string::npos);
    config.attention_head_dimension = 5;
    EXPECT_THROW(config.validate(), std::invalid_argument);
}

TEST(HuggingFaceConfigTest, ParsesPinnedQwen25AndMatchesCheckpointParameterCount) {
    const auto path = std::filesystem::temp_directory_path() / "microllm-qwen25-config.json";
    std::ofstream(path) << R"({
      "architectures":["Qwen2ForCausalLM"],
      "bos_token_id":151643,"eos_token_id":151643,
      "hidden_act":"silu","hidden_size":896,"intermediate_size":4864,
      "max_position_embeddings":32768,"model_type":"qwen2",
      "num_attention_heads":14,"num_hidden_layers":24,"num_key_value_heads":2,
      "rms_norm_eps":1e-6,"rope_theta":1000000.0,
      "tie_word_embeddings":true,"torch_dtype":"bfloat16",
      "use_mrope":false,"use_sliding_window":false,"vocab_size":151936
    })";
    const auto parsed = load_huggingface_config(path);
    EXPECT_EQ(parsed.model_type, "qwen2");
    EXPECT_EQ(parsed.torch_dtype, "bfloat16");
    EXPECT_EQ(parsed.bos_token_id, 151643);
    EXPECT_EQ(parsed.model.dimension, 896);
    EXPECT_EQ(parsed.model.layers, 24);
    EXPECT_EQ(parsed.model.kv_heads, 2);
    EXPECT_TRUE(parsed.model.attention_bias);
    EXPECT_EQ(parsed.model.rope_layout, RopeLayout::SplitHalf);
    EXPECT_TRUE(parsed.model.tie_embeddings);
    EXPECT_FLOAT_EQ(parsed.model.rms_norm_epsilon, 1.0e-6F);
    EXPECT_EQ(parsed.model.parameter_count(), 494'032'768U);
    std::error_code ignored;
    std::filesystem::remove(path, ignored);
}

TEST(HuggingFaceConfigTest, RejectsUnsupportedFamilyAndAttentionVariants) {
    const auto path = std::filesystem::temp_directory_path() / "microllm-bad-hf-config.json";
    const auto write = [&](const char* model_type, bool sliding) {
        std::ofstream(path) << "{\"bos_token_id\":1,\"eos_token_id\":2,"
            "\"hidden_act\":\"silu\",\"hidden_size\":8,\"intermediate_size\":16,"
            "\"max_position_embeddings\":8,\"model_type\":\"" << model_type << "\","
            "\"num_attention_heads\":2,\"num_hidden_layers\":1,\"num_key_value_heads\":1,"
            "\"rms_norm_eps\":1e-6,\"rope_theta\":10000,\"tie_word_embeddings\":true,"
            "\"torch_dtype\":\"float32\",\"use_mrope\":false,"
            "\"use_sliding_window\":" << (sliding ? "true" : "false") << ","
            "\"vocab_size\":16}";
    };
    write("llama", false);
    EXPECT_THROW((void)load_huggingface_config(path), std::invalid_argument);
    write("qwen2", true);
    EXPECT_THROW((void)load_huggingface_config(path), std::invalid_argument);
    std::error_code ignored;
    std::filesystem::remove(path, ignored);
}

TEST(HuggingFaceConfigTest, ParsesPinnedDeepSeekDistillQwenParameterContract) {
    const auto path = std::filesystem::path(MICROLLM_SOURCE_DIR) /
                      "tests/fixtures/deepseek-r1-distill-qwen-1.5b-config.json";
    const auto parsed = load_huggingface_config(path);
    EXPECT_EQ(parsed.model.dimension, 1536);
    EXPECT_EQ(parsed.model.layers, 28);
    EXPECT_EQ(parsed.model.heads, 12);
    EXPECT_EQ(parsed.model.kv_heads, 2);
    EXPECT_EQ(parsed.model.ffn_dimension, 8960);
    EXPECT_FALSE(parsed.model.tie_embeddings);
    EXPECT_EQ(parsed.model.parameter_count(), 1'777'088'000U);
}

TEST(HuggingFaceConfigTest, ParsesPinnedQwen3ExplicitHeadAndQkNormContract) {
    const auto path = std::filesystem::path(MICROLLM_SOURCE_DIR) /
                      "tests/fixtures/qwen3-0.6b-config.json";
    const auto parsed = load_huggingface_config(path);
    EXPECT_EQ(parsed.model_type, "qwen3");
    EXPECT_EQ(parsed.model.dimension, 1024);
    EXPECT_EQ(parsed.model.heads, 16);
    EXPECT_EQ(parsed.model.kv_heads, 8);
    EXPECT_EQ(parsed.model.head_dimension(), 128);
    EXPECT_EQ(parsed.model.query_dimension(), 2048);
    EXPECT_EQ(parsed.model.kv_dimension(), 1024);
    EXPECT_TRUE(parsed.model.qk_norm);
    EXPECT_FALSE(parsed.model.attention_bias);
    EXPECT_TRUE(parsed.model.tie_embeddings);
    EXPECT_EQ(parsed.model.parameter_count(), 596'049'920U);
    const auto mapping = qwen_style_weight_mapping(parsed.model);
    EXPECT_EQ(mapping.size(), 11U * 28U + 2U);
    EXPECT_TRUE(mapping.contains("blocks.0.attention.q_norm.weight"));
    EXPECT_FALSE(mapping.contains("blocks.0.attention.q_proj.bias"));
}

TEST(HuggingFaceConfigTest, ParsesQwen3MoeConfigAndRejectsParameterCounting) {
    const auto path = std::filesystem::temp_directory_path() / "microllm-qwen3-moe-config.json";
    std::ofstream(path) << R"({
      "bos_token_id":151643,"eos_token_id":151645,
      "hidden_act":"silu","hidden_size":8,"intermediate_size":16,
      "head_dim":4,"max_position_embeddings":32,"model_type":"qwen3_moe",
      "num_attention_heads":2,"num_hidden_layers":2,"num_key_value_heads":1,
      "rms_norm_eps":1e-6,"rope_theta":1000000.0,
      "tie_word_embeddings":false,"torch_dtype":"bfloat16",
      "use_mrope":false,"use_sliding_window":false,"vocab_size":32,
      "num_experts":4,"num_experts_per_tok":2,"moe_intermediate_size":8,
      "norm_topk_prob":true,"decoder_sparse_step":1,"mlp_only_layers":[]
    })";
    const auto parsed = load_huggingface_config(path);
    EXPECT_EQ(parsed.model_type, "qwen3_moe");
    EXPECT_TRUE(parsed.model.qk_norm);
    EXPECT_EQ(parsed.model.attention_head_dimension, 4);
    EXPECT_EQ(parsed.model.moe_num_experts, 4);
    EXPECT_EQ(parsed.model.moe_num_experts_per_tok, 2);
    EXPECT_EQ(parsed.model.moe_intermediate_size, 8);
    EXPECT_TRUE(parsed.model.moe_norm_topk_prob);
    EXPECT_THROW((void)parsed.model.parameter_count(), std::invalid_argument);
    EXPECT_NE(parsed.model.summary().find("moe_num_experts=4"), std::string::npos);
    std::error_code ignored;
    std::filesystem::remove(path, ignored);
}

TEST(HuggingFaceConfigTest, RejectsUnsupportedQwen3MoeFields) {
    const auto path =
        std::filesystem::temp_directory_path() / "microllm-bad-qwen3-moe-config.json";
    const auto write = [&](const char* extra_fields) {
        std::ofstream(path) << "{\"bos_token_id\":1,\"eos_token_id\":2,"
            "\"hidden_act\":\"silu\",\"hidden_size\":8,\"intermediate_size\":16,"
            "\"head_dim\":4,\"max_position_embeddings\":32,\"model_type\":\"qwen3_moe\","
            "\"num_attention_heads\":2,\"num_hidden_layers\":2,\"num_key_value_heads\":1,"
            "\"rms_norm_eps\":1e-6,\"rope_theta\":10000,\"tie_word_embeddings\":false,"
            "\"torch_dtype\":\"bfloat16\",\"use_mrope\":false,\"use_sliding_window\":false,"
            "\"vocab_size\":32,\"num_experts\":4,\"num_experts_per_tok\":2,"
            "\"moe_intermediate_size\":8,\"norm_topk_prob\":true,"
            << extra_fields << "}";
    };
    // decoder_sparse_step != 1 means some layers are dense-only: unsupported
    // per-layer mixing.
    write("\"decoder_sparse_step\":2,\"mlp_only_layers\":[]");
    EXPECT_THROW((void)load_huggingface_config(path), std::invalid_argument);
    // A non-empty mlp_only_layers is the same unsupported mixing from the other
    // field HF uses to express it.
    write("\"decoder_sparse_step\":1,\"mlp_only_layers\":[0]");
    EXPECT_THROW((void)load_huggingface_config(path), std::invalid_argument);
    // router_aux_loss_coef configures a training-time loss this repo does not
    // implement; its presence is rejected rather than silently dropped.
    write("\"decoder_sparse_step\":1,\"mlp_only_layers\":[],\"router_aux_loss_coef\":0.001");
    EXPECT_THROW((void)load_huggingface_config(path), std::invalid_argument);
    std::error_code ignored;
    std::filesystem::remove(path, ignored);
}

}  // namespace microllm::model
