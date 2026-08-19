#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include <gtest/gtest.h>
#include <microllm/autograd/autograd.h>
#include <microllm/training/checkpoint.h>

namespace microllm::training {
using namespace microllm::autograd;
namespace {

std::filesystem::path checkpoint_path(const char* name) {
    return std::filesystem::temp_directory_path() / name;
}

void apply_gradient(Value& parameter, AdamW& optimizer, float gradient) {
    sum(scale(parameter, gradient)).backward();
    optimizer.step();
    optimizer.zero_grad();
}

}  // namespace

TEST(CheckpointTest, SavesAndRestoresCompleteTrainingState) {
    const auto path = checkpoint_path("microllm-complete-state.ckpt");
    Value parameter(Tensor::from_vector({1.0F, -2.0F}, {2}), true);
    const AdamWConfig config{.learning_rate = 0.01F, .weight_decay = 0.0F};
    AdamW optimizer({&parameter}, config);
    apply_gradient(parameter, optimizer, 3.0F);
    const ExperimentState expected{.global_step = 17,
                                   .data_cursor = 4096,
                                   .rng_state = "rng-state-17",
                                   .model_config = "dim=8,layers=1",
                                   .data_config = "generated-sequence-v1"};
    save_checkpoint(path, {{"weight", &parameter}}, optimizer, expected);
    save_checkpoint(path, {{"weight", &parameter}}, optimizer, expected);
    auto temporary = path;
    temporary += ".tmp";
    EXPECT_FALSE(std::filesystem::exists(temporary));

    Value restored(Tensor::from_vector({0, 0}, {2}), true);
    AdamW restored_optimizer({&restored}, config);
    ExperimentState actual;
    const auto loaded = load_checkpoint(path);
    restore_checkpoint(loaded, {{"weight", &restored}}, restored_optimizer, actual);

    EXPECT_EQ(restored.data().to_vector(), parameter.data().to_vector());
    EXPECT_EQ(restored_optimizer.state().step, optimizer.state().step);
    EXPECT_EQ(actual.global_step, expected.global_step);
    EXPECT_EQ(actual.data_cursor, expected.data_cursor);
    EXPECT_EQ(actual.rng_state, expected.rng_state);
    EXPECT_EQ(actual.model_config, expected.model_config);
    EXPECT_EQ(actual.data_config, expected.data_config);
    std::filesystem::remove(path);
}

TEST(CheckpointTest, RestoredRunMatchesUninterruptedSubsequentSteps) {
    const auto path = checkpoint_path("microllm-resume-trajectory.ckpt");
    const AdamWConfig config{.learning_rate = 0.005F, .weight_decay = 0.01F};
    Value uninterrupted(Tensor::from_vector({0.5F, -1.5F, 2.0F}, {3}), true);
    AdamW original({&uninterrupted}, config);
    apply_gradient(uninterrupted, original, 2.0F);
    save_checkpoint(path, {{"projection.weight", &uninterrupted}}, original,
                    {.global_step = 1,
                     .data_cursor = 3,
                     .rng_state = "state",
                     .model_config = "",
                     .data_config = ""});

    Value resumed(Tensor::from_vector({0, 0, 0}, {3}), true);
    AdamW resumed_optimizer({&resumed}, config);
    ExperimentState resumed_experiment;
    restore_checkpoint(load_checkpoint(path), {{"projection.weight", &resumed}},
                       resumed_optimizer, resumed_experiment);
    for (const auto gradient : {-1.0F, 0.25F, 4.0F}) {
        apply_gradient(uninterrupted, original, gradient);
        apply_gradient(resumed, resumed_optimizer, gradient);
        EXPECT_EQ(resumed.data().to_vector(), uninterrupted.data().to_vector());
    }
    EXPECT_EQ(resumed_optimizer.state().step, original.state().step);
    std::filesystem::remove(path);
}

TEST(CheckpointTest, RejectsCorruptionAndParameterContractMismatch) {
    const auto path = checkpoint_path("microllm-corrupt.ckpt");
    Value parameter(Tensor::from_vector({1}, {1}), true);
    AdamW optimizer({&parameter});
    save_checkpoint(path, {{"weight", &parameter}}, optimizer, {});
    {
        std::fstream file(path, std::ios::binary | std::ios::in | std::ios::out);
        ASSERT_TRUE(file);
        file.seekg(-1, std::ios::end);
        char value = 0;
        file.read(&value, 1);
        value = static_cast<char>(value ^ 0x01);
        file.seekp(-1, std::ios::end);
        file.write(&value, 1);
    }
    EXPECT_THROW((void)load_checkpoint(path), std::runtime_error);
    std::filesystem::remove(path);

    EXPECT_THROW(save_checkpoint(path, {{"same", &parameter}, {"same", &parameter}}, optimizer, {}),
                 std::invalid_argument);
}

}  // namespace microllm::training
