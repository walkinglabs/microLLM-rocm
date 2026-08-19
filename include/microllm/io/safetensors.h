#pragma once

#include <filesystem>
#include <map>
#include <string>
#include <vector>

#include <microllm/core/tensor.h>

namespace microllm::io {

using StateDict = std::map<std::string, Tensor>;

enum class WeightFileDType { Float32, BFloat16, Float16 };

struct SafetensorsSaveOptions {
    WeightFileDType dtype = WeightFileDType::Float32;
    bool atomic_replace = true;
};

[[nodiscard]] StateDict load_safetensors(
    const std::filesystem::path& path, Device target = Device::cpu());
[[nodiscard]] StateDict load_safetensors_files(
    const std::vector<std::filesystem::path>& paths,
    Device target = Device::cpu());
[[nodiscard]] StateDict load_safetensors_index(
    const std::filesystem::path& index_path,
    Device target = Device::cpu());
void save_safetensors(const std::filesystem::path& path,
                      const StateDict& state,
                      const SafetensorsSaveOptions& options = {});

}  // namespace microllm::io
