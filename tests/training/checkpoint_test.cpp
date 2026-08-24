#include <filesystem>
#include <fstream>
#include <cstdint>
#include <iterator>
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

std::uint64_t read_u64(const std::vector<unsigned char>& bytes,
                       std::size_t offset) {
    std::uint64_t value = 0;
    for (std::size_t byte = 0; byte < 8; ++byte) {
        value |= static_cast<std::uint64_t>(bytes.at(offset + byte))
                 << (byte * 8U);
    }
    return value;
}

void write_little_endian(std::vector<unsigned char>& bytes,
                         std::size_t offset, std::uint64_t value,
                         std::size_t width) {
    for (std::size_t byte = 0; byte < width; ++byte) {
        bytes.at(offset + byte) =
            static_cast<unsigned char>((value >> (byte * 8U)) & 0xffU);
    }
}

void convert_current_checkpoint_to_v1(const std::filesystem::path& path) {
    std::ifstream input(path, std::ios::binary);
    ASSERT_TRUE(input);
    std::vector<unsigned char> bytes{
        std::istreambuf_iterator<char>(input),
        std::istreambuf_iterator<char>()};
    ASSERT_GE(bytes.size(), 32U);
    std::size_t position = 32 + 16;
    const auto skip_string = [&] {
        const auto size = read_u64(bytes, position);
        position += 8 + static_cast<std::size_t>(size);
    };
    skip_string();
    skip_string();
    skip_string();
    const auto parameters = read_u64(bytes, position);
    position += 8;
    for (std::uint64_t parameter = 0; parameter < parameters; ++parameter) {
        skip_string();
        const auto rank = read_u64(bytes, position);
        position += 8 + static_cast<std::size_t>(rank) * 8;
        const auto elements = read_u64(bytes, position);
        position += 8 + static_cast<std::size_t>(elements) * sizeof(float);
    }
    position += 5 * sizeof(float);
    ASSERT_LE(position + sizeof(std::uint32_t), bytes.size());
    bytes.erase(bytes.begin() + static_cast<std::ptrdiff_t>(position),
                bytes.begin() + static_cast<std::ptrdiff_t>(
                                    position + sizeof(std::uint32_t)));
    write_little_endian(bytes, 8, 1, sizeof(std::uint32_t));
    write_little_endian(bytes, 16,
                        static_cast<std::uint64_t>(bytes.size() - 32),
                        sizeof(std::uint64_t));
    std::uint64_t checksum = 14695981039346656037ULL;
    for (std::size_t index = 32; index < bytes.size(); ++index) {
        checksum ^= bytes[index];
        checksum *= 1099511628211ULL;
    }
    write_little_endian(bytes, 24, checksum, sizeof(std::uint64_t));
    std::ofstream output(path, std::ios::binary | std::ios::trunc);
    ASSERT_TRUE(output);
    output.write(reinterpret_cast<const char*>(bytes.data()),
                 static_cast<std::streamsize>(bytes.size()));
    ASSERT_TRUE(output);
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

TEST(CheckpointTest, PersistsBf16MomentPolicyAndRestoresItsTrajectory) {
    const auto path = checkpoint_path("microllm-bf16-moment-state.ckpt");
    const AdamWConfig config{
        .learning_rate = 0.005F,
        .beta1 = 0.9F,
        .beta2 = 0.99F,
        .epsilon = 1.0e-8F,
        .weight_decay = 0.01F,
        .moment_precision = AdamWConfig::MomentPrecision::BFloat16};
    Value uninterrupted(Tensor::from_vector({0.5F, -1.5F, 2.0F}, {3}), true);
    AdamW original({&uninterrupted}, config);
    apply_gradient(uninterrupted, original, 2.0F);
    save_checkpoint(path, {{"projection.weight", &uninterrupted}}, original,
                    {.global_step = 1,
                     .data_cursor = 3,
                     .rng_state = "bf16-state",
                     .model_config = "tiny",
                     .data_config = "generated"});

    const auto loaded = load_checkpoint(path);
    EXPECT_EQ(loaded.format_version, kCheckpointFormatVersion);
    EXPECT_EQ(loaded.optimizer_config.moment_precision,
              AdamWConfig::MomentPrecision::BFloat16);
    ASSERT_EQ(loaded.optimizer_state.first_moments.size(), 1U);
    EXPECT_EQ(loaded.optimizer_state.first_moments[0].dtype(), DType::Float32);

    Value resumed(Tensor::from_vector({0.0F, 0.0F, 0.0F}, {3}), true);
    AdamW resumed_optimizer({&resumed}, config);
    ExperimentState experiment;
    restore_checkpoint(loaded, {{"projection.weight", &resumed}},
                       resumed_optimizer, experiment);
    EXPECT_EQ(resumed_optimizer.moment_state_bytes(), 12U);
    for (const auto gradient : {-1.0F, 0.25F, 4.0F}) {
        apply_gradient(uninterrupted, original, gradient);
        apply_gradient(resumed, resumed_optimizer, gradient);
        EXPECT_EQ(resumed.data().to_vector(), uninterrupted.data().to_vector());
    }

    Value wrong_policy_parameter(
        Tensor::from_vector({0.0F, 0.0F, 0.0F}, {3}), true);
    AdamW wrong_policy({&wrong_policy_parameter});
    EXPECT_THROW(
        restore_checkpoint(loaded,
                           {{"projection.weight", &wrong_policy_parameter}},
                           wrong_policy, experiment),
        std::invalid_argument);
    std::filesystem::remove(path);
}

TEST(CheckpointTest, LoadsVersionOneAsFp32MomentPolicy) {
    const auto path = checkpoint_path("microllm-version-one.ckpt");
    Value parameter(Tensor::from_vector({1.0F, -2.0F}, {2}), true);
    AdamW optimizer({&parameter});
    apply_gradient(parameter, optimizer, 0.5F);
    save_checkpoint(path, {{"weight", &parameter}}, optimizer,
                    {.global_step = 1,
                     .data_cursor = 2,
                     .rng_state = "legacy",
                     .model_config = "tiny",
                     .data_config = "generated"});
    convert_current_checkpoint_to_v1(path);
    const auto loaded = load_checkpoint(path);
    EXPECT_EQ(loaded.format_version, 1U);
    EXPECT_EQ(loaded.optimizer_config.moment_precision,
              AdamWConfig::MomentPrecision::Float32);
    Value restored(Tensor::from_vector({0.0F, 0.0F}, {2}), true);
    AdamW restored_optimizer({&restored});
    ExperimentState experiment;
    restore_checkpoint(loaded, {{"weight", &restored}}, restored_optimizer,
                       experiment);
    EXPECT_EQ(restored.data().to_vector(), parameter.data().to_vector());
    EXPECT_EQ(restored_optimizer.state().first_moments[0].to_vector(),
              optimizer.state().first_moments[0].to_vector());
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
