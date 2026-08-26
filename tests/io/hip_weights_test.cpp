#include <chrono>
#include <filesystem>
#include <fstream>

#include <gtest/gtest.h>
#include <microllm/io/safetensors.h>
#include <microllm/model/model.h>
#include <microllm/ops/ops.h>
#include <microllm/runtime/runtime.h>

namespace microllm::io {
namespace {

void require_gpu() {
    if (runtime::hip_device_count() == 0) GTEST_SKIP() << "No visible HIP device";
}

std::filesystem::path temporary_path() {
    return std::filesystem::temp_directory_path() /
           ("microllm-hip-weights-" + std::to_string(
               std::chrono::steady_clock::now().time_since_epoch().count()) +
            ".safetensors");
}

model::ModelConfig config() {
    return {.vocabulary_size = 8,
            .dimension = 8,
            .layers = 1,
            .heads = 2,
            .kv_heads = 1,
            .ffn_dimension = 16,
            .max_sequence_length = 4,
            .rope_base = 10000.0F,
            .tie_embeddings = false};
}

}  // namespace

TEST(HipWeightsTest, LoadsSafetensorsDirectlyToGpuAndIntoGpuModel) {
    require_gpu();
    const auto path = temporary_path();
    model::TransformerModel source(config(), 401);
    const auto expected = source.state_dict();
    save_safetensors(path, expected,
                     {.dtype = WeightFileDType::BFloat16, .atomic_replace = true});

    const auto gpu_state = load_safetensors(path, Device::hip());
    ASSERT_EQ(gpu_state.size(), expected.size());
    for (const auto& [name, tensor] : expected) {
        EXPECT_EQ(gpu_state.at(name).device(), Device::hip());
        const auto actual = gpu_state.at(name).to_vector();
        const auto reference = tensor.to_vector();
        ASSERT_EQ(actual.size(), reference.size());
        for (std::size_t index = 0; index < actual.size(); ++index) {
            EXPECT_NEAR(actual[index], reference[index], 2.0e-2F) << name;
        }
    }

    model::TransformerModel target(
        config(), 409, model::ParameterInitialization::Uninitialized);
    target.to(Device::hip());
    runtime::reset_transfer_stats();
    const auto report = target.load_safetensors(
        path, {.strict = true, .mapping = {}, .aliases = {}});
    const auto transfers = runtime::transfer_stats();
    EXPECT_TRUE(report.complete());
    EXPECT_EQ(transfers.host_to_device_bytes,
              static_cast<std::size_t>(source.parameter_count()) * 2U);
    EXPECT_EQ(transfers.device_to_host_calls, 0U);
    EXPECT_EQ(transfers.device_to_device_calls, 0U);
    EXPECT_EQ(target.device(), Device::hip());
    const auto snapshot = target.state_dict(Device::hip());
    for (const auto& [name, tensor] : snapshot) {
        EXPECT_EQ(tensor.device(), Device::hip());
        EXPECT_EQ(tensor.to_vector(), gpu_state.at(name).to_vector());
    }

    auto incompatible_config = config();
    incompatible_config.vocabulary_size += 1;
    model::TransformerModel incompatible(
        incompatible_config, 419, model::ParameterInitialization::Uninitialized);
    incompatible.to(Device::hip());
    runtime::reset_transfer_stats();
    EXPECT_THROW((void)incompatible.load_safetensors(path), std::invalid_argument);
    EXPECT_EQ(runtime::transfer_stats().host_to_device_calls, 0U);
    EXPECT_THROW((void)incompatible.forward_inference(
                     Tensor::from_int32_vector({1}, {1, 1}).to(Device::hip())),
                 std::logic_error);
    std::error_code ignored;
    std::filesystem::remove(path, ignored);
}

TEST(HipWeightsTest, LoadsMixedInt8WeightAndScaleWithoutDequantizeD2H) {
    require_gpu();
    const auto path = temporary_path();
    save_safetensors(
        path,
        {{"linear.weight", Tensor::from_int8_vector(
                               {-127, -4, -1, 0, 3, 126}, {2, 3})},
         {"linear.weight.scale", Tensor::from_vector({0.125F}, {})}},
        {.dtype = WeightFileDType::Preserve, .atomic_replace = true});

    runtime::reset_transfer_stats();
    const auto state = load_safetensors(path, Device::hip(0));
    const auto loading = runtime::transfer_stats();
    ASSERT_EQ(state.size(), 2U);
    EXPECT_EQ(loading.host_to_device_calls, 2U);
    EXPECT_EQ(loading.host_to_device_bytes, 10U);
    EXPECT_EQ(loading.device_to_host_calls, 0U);
    EXPECT_EQ(state.at("linear.weight").dtype(), DType::Int8);
    EXPECT_EQ(state.at("linear.weight").device(), Device::hip(0));
    EXPECT_EQ(state.at("linear.weight.scale").dtype(), DType::Float32);

    const ops::Int8ScaledTensor weight{
        state.at("linear.weight"), state.at("linear.weight.scale")};
    runtime::reset_transfer_stats();
    const auto restored = ops::dequantize_int8(weight);
    runtime::synchronize(Device::hip(0));
    const auto hot = runtime::transfer_stats();
    EXPECT_EQ(hot.host_to_device_calls, 0U);
    EXPECT_EQ(hot.device_to_host_calls, 0U);
    EXPECT_EQ(restored.to_vector(),
              (std::vector<float>{-15.875F, -0.5F, -0.125F,
                                  0.0F, 0.375F, 15.75F}));

    std::error_code ignored;
    std::filesystem::remove(path, ignored);
}

TEST(HipWeightsTest, StrictStreamingVerifiesTiedAliasBeforeAnyH2D) {
    require_gpu();
    auto qwen3 = config();
    qwen3.attention_head_dimension = 6;
    qwen3.qk_norm = true;
    qwen3.tie_embeddings = true;
    model::TransformerModel source(qwen3, 461);
    const auto native = source.state_dict();
    const auto mapping = model::qwen_style_weight_mapping(qwen3);
    StateDict external;
    for (const auto& [target, spec] : mapping) {
        auto tensor = native.at(target);
        if (spec.transform == model::WeightTransform::Transpose2D) {
            tensor = tensor.transpose(0, 1).contiguous();
        }
        external.emplace(spec.name, Tensor::from_vector(
            tensor.to_vector(), tensor.shape()));
    }
    external.emplace("lm_head.weight", Tensor::from_vector(
        external.at("model.embed_tokens.weight").to_vector(),
        external.at("model.embed_tokens.weight").shape()));
    const model::LoadWeightsOptions options{
        .strict = true, .mapping = mapping,
        .aliases = model::qwen3_tied_weight_aliases(qwen3)};
    const auto good_path = temporary_path();
    save_safetensors(good_path, external,
                     {.dtype = WeightFileDType::BFloat16});
    model::TransformerModel loaded(
        qwen3, 463, model::ParameterInitialization::Uninitialized);
    loaded.to(Device::hip(0));
    EXPECT_TRUE(loaded.load_safetensors(good_path, options).complete());
    EXPECT_TRUE(loaded.forward_inference(
        Tensor::from_int32_vector({1}, {1, 1}).to(Device::hip(0))).defined());

    auto bad_values = external.at("lm_head.weight").to_vector();
    bad_values[0] += 1.0F;
    external.at("lm_head.weight") = Tensor::from_vector(
        bad_values, external.at("lm_head.weight").shape());
    const auto bad_path = temporary_path();
    save_safetensors(bad_path, external,
                     {.dtype = WeightFileDType::BFloat16});
    model::TransformerModel rejected(
        qwen3, 467, model::ParameterInitialization::Uninitialized);
    rejected.to(Device::hip(0));
    runtime::reset_transfer_stats();
    EXPECT_THROW((void)rejected.load_safetensors(bad_path, options),
                 std::invalid_argument);
    EXPECT_EQ(runtime::transfer_stats().host_to_device_calls, 0U);
    EXPECT_THROW((void)rejected.forward_inference(
                     Tensor::from_int32_vector({1}, {1, 1}).to(Device::hip(0))),
                 std::logic_error);
    std::error_code ignored;
    std::filesystem::remove(good_path, ignored);
    std::filesystem::remove(bad_path, ignored);
}

TEST(HipWeightsTest, StreamsMultipleShardsAfterWholeSetPreflight) {
    require_gpu();
    const auto first_path = temporary_path();
    const auto second_path = temporary_path();
    model::TransformerModel source(config(), 431);
    const auto expected = source.state_dict();
    StateDict first;
    StateDict second;
    bool alternate = false;
    for (const auto& item : expected) {
        (alternate ? first : second).insert(item);
        alternate = !alternate;
    }
    save_safetensors(first_path, first,
                     {.dtype = WeightFileDType::BFloat16});
    save_safetensors(second_path, second,
                     {.dtype = WeightFileDType::BFloat16});

    model::TransformerModel target(
        config(), 433, model::ParameterInitialization::Uninitialized);
    target.to(Device::hip());
    runtime::reset_transfer_stats();
    const auto report = target.load_safetensors_files(
        {first_path, second_path},
        {.strict = true, .mapping = {}, .aliases = {}});
    const auto transfers = runtime::transfer_stats();
    EXPECT_TRUE(report.complete());
    EXPECT_EQ(transfers.host_to_device_bytes,
              static_cast<std::size_t>(source.parameter_count()) * 2U);
    EXPECT_EQ(transfers.device_to_host_calls, 0U);
    EXPECT_EQ(transfers.device_to_device_calls, 0U);
    const auto actual = target.state_dict();
    for (const auto& [name, tensor] : expected) {
        const auto left = actual.at(name).to_vector();
        const auto right = tensor.to_vector();
        ASSERT_EQ(left.size(), right.size());
        for (std::size_t index = 0; index < left.size(); ++index) {
            EXPECT_NEAR(left[index], right[index], 2.0e-2F) << name;
        }
    }

    model::TransformerModel missing(
        config(), 439, model::ParameterInitialization::Uninitialized);
    missing.to(Device::hip());
    runtime::reset_transfer_stats();
    EXPECT_THROW((void)missing.load_safetensors_files(
                     {first_path},
                     {.strict = true, .mapping = {}, .aliases = {}}),
                 std::invalid_argument);
    EXPECT_EQ(runtime::transfer_stats().host_to_device_calls, 0U);

    model::TransformerModel duplicate(
        config(), 443, model::ParameterInitialization::Uninitialized);
    duplicate.to(Device::hip());
    runtime::reset_transfer_stats();
    EXPECT_THROW((void)duplicate.load_safetensors_files(
                     {first_path, first_path},
                     {.strict = true, .mapping = {}, .aliases = {}}),
                 std::runtime_error);
    EXPECT_EQ(runtime::transfer_stats().host_to_device_calls, 0U);

    const auto index_path = first_path.parent_path() /
        (first_path.stem().string() + "-index.json");
    {
        std::ofstream index(index_path);
        index << "{\"weight_map\":{";
        bool first_name = true;
        for (const auto& [name, tensor] : expected) {
            (void)tensor;
            if (!first_name) index << ',';
            first_name = false;
            index << '"' << name << "\":\""
                  << (first.contains(name) ? first_path.filename().string()
                                           : second_path.filename().string())
                  << '"';
        }
        index << "}}";
    }
    model::TransformerModel indexed(
        config(), 449, model::ParameterInitialization::Uninitialized);
    indexed.to(Device::hip());
    runtime::reset_transfer_stats();
    const auto indexed_report = indexed.load_safetensors_index(
        index_path, {.strict = true, .mapping = {}, .aliases = {}});
    EXPECT_TRUE(indexed_report.complete());
    EXPECT_EQ(runtime::transfer_stats().host_to_device_bytes,
              static_cast<std::size_t>(source.parameter_count()) * 2U);
    EXPECT_EQ(runtime::transfer_stats().device_to_host_calls, 0U);

    std::error_code ignored;
    std::filesystem::remove(first_path, ignored);
    std::filesystem::remove(second_path, ignored);
    std::filesystem::remove(index_path, ignored);
}

}  // namespace microllm::io
