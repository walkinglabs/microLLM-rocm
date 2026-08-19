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

    model::TransformerModel target(config(), 409);
    target.to(Device::hip());
    const auto report = target.load_safetensors(
        path, {.strict = true, .mapping = {}});
    EXPECT_TRUE(report.complete());
    EXPECT_EQ(target.device(), Device::hip());
    const auto snapshot = target.state_dict(Device::hip());
    for (const auto& [name, tensor] : snapshot) {
        (void)name;
        EXPECT_EQ(tensor.device(), Device::hip());
    }
    std::error_code ignored;
    std::filesystem::remove(path, ignored);
}

}  // namespace microllm::io
