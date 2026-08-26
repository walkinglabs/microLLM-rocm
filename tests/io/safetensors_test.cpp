#include <chrono>
#include <filesystem>
#include <fstream>
#include <string>
#include <tuple>
#include <vector>

#include <gtest/gtest.h>
#include <microllm/io/safetensors.h>

namespace microllm::io {
namespace {

class TemporaryDirectory {
public:
    TemporaryDirectory() {
        path_ = std::filesystem::temp_directory_path() /
                ("microllm-weights-" + std::to_string(
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

void expect_near(const Tensor& actual, const Tensor& expected, float tolerance) {
    EXPECT_EQ(actual.shape(), expected.shape());
    const auto left = actual.to_vector();
    const auto right = expected.to_vector();
    ASSERT_EQ(left.size(), right.size());
    for (std::size_t index = 0; index < left.size(); ++index) {
        EXPECT_NEAR(left[index], right[index], tolerance) << "index=" << index;
    }
}

StateDict fixture() {
    StateDict state;
    state.emplace("layer.weight",
                  Tensor::from_vector({-3.25F, -1.0F, -0.125F, 0.0F, 1.5F, 7.75F}, {2, 3}));
    state.emplace("norm.weight", Tensor::from_vector({1.0F, 0.5F, 2.0F}, {3}));
    state.emplace("scalar", Tensor::from_vector({0.333251953125F}, {}));
    return state;
}

}  // namespace

TEST(SafetensorsTest, RoundTripsFloat32Bfloat16AndFloat16) {
    TemporaryDirectory directory;
    const auto expected = fixture();
    for (const auto& [dtype, tolerance, suffix] :
         {std::tuple{WeightFileDType::Float32, 0.0F, "f32"},
          std::tuple{WeightFileDType::BFloat16, 2.0e-2F, "bf16"},
          std::tuple{WeightFileDType::Float16, 2.0e-3F, "f16"}}) {
        const auto path = directory.path() / (std::string(suffix) + ".safetensors");
        save_safetensors(path, expected, {.dtype = dtype, .atomic_replace = true});
        EXPECT_TRUE(std::filesystem::is_regular_file(path));
        EXPECT_FALSE(std::filesystem::exists(path.string() + ".tmp"));
        const auto actual = load_safetensors(path);
        ASSERT_EQ(actual.size(), expected.size());
        for (const auto& [name, tensor] : expected) {
            ASSERT_TRUE(actual.contains(name));
            expect_near(actual.at(name), tensor, tolerance);
        }
    }
}

TEST(SafetensorsTest, InspectAndVisitExposeOrderedBoundedRawPayloads) {
    TemporaryDirectory directory;
    const auto path = directory.path() / "stream.safetensors";
    const auto expected = fixture();
    save_safetensors(path, expected,
                     {.dtype = WeightFileDType::BFloat16, .atomic_replace = true});
    const auto metadata = inspect_safetensors(path);
    ASSERT_EQ(metadata.size(), expected.size());
    std::uint64_t expected_bytes = 0;
    for (const auto& info : metadata) {
        EXPECT_TRUE(expected.contains(info.name));
        EXPECT_EQ(info.dtype, DType::BFloat16);
        EXPECT_EQ(info.shape, expected.at(info.name).shape());
        EXPECT_EQ(info.data_bytes,
                  static_cast<std::uint64_t>(expected.at(info.name).numel()) * 2U);
        expected_bytes += info.data_bytes;
    }
    std::vector<std::string> visited;
    std::uint64_t visited_bytes = 0;
    const auto visit_report = visit_safetensors(
        path, [&](const SafetensorsTensorInfo& info,
                  std::span<const std::byte> bytes) {
        visited.push_back(info.name);
        EXPECT_EQ(bytes.size(), info.data_bytes);
        visited_bytes += bytes.size();
    });
    ASSERT_EQ(visited.size(), metadata.size());
    EXPECT_EQ(visited_bytes, expected_bytes);
    EXPECT_EQ(visit_report.tensors, metadata.size());
    EXPECT_EQ(visit_report.payload_bytes, expected_bytes);
#if defined(__unix__) || defined(__APPLE__)
    EXPECT_TRUE(visit_report.memory_mapped);
#endif
    EXPECT_THROW(visit_safetensors(path, {}), std::invalid_argument);
}

TEST(SafetensorsTest, LoadsMultipleShardsAndIndexWeightMap) {
    TemporaryDirectory directory;
    const auto first_path = directory.path() / "model-00001.safetensors";
    const auto second_path = directory.path() / "model-00002.safetensors";
    save_safetensors(first_path,
                     {{"a.weight", Tensor::from_vector({1, 2, 3, 4}, {2, 2})}});
    save_safetensors(second_path,
                     {{"b.weight", Tensor::from_vector({5, 6, 7}, {3})}});

    const auto combined = load_safetensors_files({first_path, second_path});
    EXPECT_EQ(combined.size(), 2U);
    EXPECT_EQ(combined.at("a.weight").to_vector(), (std::vector<float>{1, 2, 3, 4}));
    EXPECT_EQ(combined.at("b.weight").to_vector(), (std::vector<float>{5, 6, 7}));

    const auto index_path = directory.path() / "model.safetensors.index.json";
    std::ofstream index(index_path);
    index << "{\"metadata\":{\"total_size\":28},\"weight_map\":{"
             "\"a.weight\":\"model-00001.safetensors\","
             "\"b.weight\":\"model-00002.safetensors\"}}";
    index.close();
    const auto inspected_index = inspect_safetensors_index(index_path);
    EXPECT_EQ(inspected_index.size(), 2U);
    EXPECT_EQ(inspected_index.at("a.weight"), first_path.lexically_normal());
    EXPECT_EQ(inspected_index.at("b.weight"), second_path.lexically_normal());
    const auto indexed = load_safetensors_index(index_path);
    EXPECT_EQ(indexed.size(), 2U);
    EXPECT_EQ(indexed.at("a.weight").to_vector(), combined.at("a.weight").to_vector());
    EXPECT_EQ(indexed.at("b.weight").to_vector(), combined.at("b.weight").to_vector());
}

TEST(SafetensorsTest, RejectsCorruptionUnsupportedDataAndDuplicateShards) {
    TemporaryDirectory directory;
    EXPECT_THROW(save_safetensors(directory.path() / "empty.safetensors", {}),
                 std::invalid_argument);
    EXPECT_THROW(save_safetensors(
                     directory.path() / "integer.safetensors",
                     {{"bad", Tensor::from_int32_vector({1, 2}, {2})}}),
                 std::invalid_argument);

    const auto short_path = directory.path() / "short.safetensors";
    std::ofstream(short_path, std::ios::binary) << "short";
    EXPECT_THROW((void)load_safetensors(short_path), std::runtime_error);

    const auto unsupported_path = directory.path() / "unsupported.safetensors";
    const std::string header =
        "{\"bad\":{\"dtype\":\"I8\",\"shape\":[1],\"data_offsets\":[0,1]}}";
    std::ofstream unsupported(unsupported_path, std::ios::binary);
    const auto length = static_cast<std::uint64_t>(header.size());
    for (unsigned index = 0; index < 8; ++index) {
        unsupported.put(static_cast<char>((length >> (index * 8U)) & 0xffU));
    }
    unsupported << header;
    unsupported.put('\0');
    unsupported.close();
    EXPECT_THROW((void)load_safetensors(unsupported_path), std::runtime_error);

    const auto shard_a = directory.path() / "a.safetensors";
    const auto shard_b = directory.path() / "b.safetensors";
    save_safetensors(shard_a, {{"duplicate", Tensor::from_vector({1}, {1})}});
    save_safetensors(shard_b, {{"duplicate", Tensor::from_vector({2}, {1})}});
    EXPECT_THROW((void)load_safetensors_files({shard_a, shard_b}), std::runtime_error);
    EXPECT_THROW((void)load_safetensors_files({}), std::invalid_argument);
}

TEST(SafetensorsTest, RejectsUnsafeOrMissingIndexEntries) {
    TemporaryDirectory directory;
    const auto unsafe_index = directory.path() / "unsafe.index.json";
    std::ofstream(unsafe_index)
        << "{\"weight_map\":{\"weight\":\"../outside.safetensors\"}}";
    EXPECT_THROW((void)load_safetensors_index(unsafe_index), std::runtime_error);

    const auto shard = directory.path() / "shard.safetensors";
    save_safetensors(shard, {{"other", Tensor::from_vector({1}, {1})}});
    const auto missing_index = directory.path() / "missing.index.json";
    std::ofstream(missing_index)
        << "{\"weight_map\":{\"wanted\":\"shard.safetensors\"}}";
    EXPECT_THROW((void)load_safetensors_index(missing_index), std::runtime_error);
}

}  // namespace microllm::io
