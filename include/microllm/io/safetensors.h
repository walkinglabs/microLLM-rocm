#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <map>
#include <span>
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

struct SafetensorsTensorInfo {
    std::string name;
    DType dtype = DType::Float32;
    Shape shape;
    std::uint64_t data_bytes = 0;
};

using SafetensorsTensorVisitor =
    std::function<void(const SafetensorsTensorInfo&, std::span<const std::byte>)>;

[[nodiscard]] std::vector<SafetensorsTensorInfo> inspect_safetensors(
    const std::filesystem::path& path);
// The byte span is valid only for the duration of the callback. Tensors are
// visited in payload-offset order so callers can stream large checkpoints.
void visit_safetensors(const std::filesystem::path& path,
                       const SafetensorsTensorVisitor& visitor);

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
