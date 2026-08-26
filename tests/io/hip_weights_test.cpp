#include <chrono>
#include <filesystem>

#include <gtest/gtest.h>
#include <microllm/io/safetensors.h>
#include <microllm/model/model.h>
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
        path, {.strict = true, .mapping = {}});
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
        {first_path, second_path}, {.strict = true, .mapping = {}});
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
                     {first_path}, {.strict = true, .mapping = {}}),
                 std::invalid_argument);
    EXPECT_EQ(runtime::transfer_stats().host_to_device_calls, 0U);

    model::TransformerModel duplicate(
        config(), 443, model::ParameterInitialization::Uninitialized);
    duplicate.to(Device::hip());
    runtime::reset_transfer_stats();
    EXPECT_THROW((void)duplicate.load_safetensors_files(
                     {first_path, first_path},
                     {.strict = true, .mapping = {}}),
                 std::runtime_error);
    EXPECT_EQ(runtime::transfer_stats().host_to_device_calls, 0U);

    std::error_code ignored;
    std::filesystem::remove(first_path, ignored);
    std::filesystem::remove(second_path, ignored);
}

}  // namespace microllm::io
