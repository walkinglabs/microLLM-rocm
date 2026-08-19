#pragma once

#include <cstdint>
#include <filesystem>
#include <string>
#include <utility>
#include <vector>

#include <microllm/training/optimizer.h>

namespace microllm::training {

inline constexpr std::uint32_t kCheckpointFormatVersion = 1;

using NamedParameters = std::vector<std::pair<std::string, autograd::Value*>>;

struct ExperimentState {
    std::uint64_t global_step = 0;
    std::uint64_t data_cursor = 0;
    std::string rng_state;
    std::string model_config;
    std::string data_config;
};

struct NamedTensor {
    std::string name;
    Tensor tensor;
};

struct LoadedCheckpoint {
    std::uint32_t format_version = 0;
    ExperimentState experiment;
    std::vector<NamedTensor> parameters;
    AdamWConfig optimizer_config;
    AdamWState optimizer_state;
};

void save_checkpoint(const std::filesystem::path& path, const NamedParameters& parameters,
                     const AdamW& optimizer, const ExperimentState& experiment);

[[nodiscard]] LoadedCheckpoint load_checkpoint(const std::filesystem::path& path);

void restore_checkpoint(const LoadedCheckpoint& checkpoint,
                        const NamedParameters& parameters, AdamW& optimizer,
                        ExperimentState& experiment);

}  // namespace microllm::training
