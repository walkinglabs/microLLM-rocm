#include <chrono>
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

    const auto report = target.load_state_dict(broken, {.strict = false, .mapping = {}});
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
    const auto report = target.load_state_dict(source.state_dict());
    EXPECT_TRUE(report.complete());
    EXPECT_EQ(report.loaded.size(), source.named_parameters().size());
    for (const auto* parameter : target.parameters()) EXPECT_FALSE(parameter->has_grad());
    EXPECT_EQ(target.forward(tokens()).data().to_vector(),
              source.forward(tokens()).data().to_vector());
}

TEST(ModelWeightsTest, QwenStyleMappingTransposesLinearWeights) {
    const auto config = weight_config();
    TransformerModel source(config, 331);
    TransformerModel target(config, 337);
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
        external, {.strict = true, .mapping = mapping});
    EXPECT_TRUE(report.complete());
    expect_state_equal(target.state_dict(), native);
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

}  // namespace microllm::model
