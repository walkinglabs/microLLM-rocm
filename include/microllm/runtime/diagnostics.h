#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

#include <microllm/base/device.h>

namespace microllm::runtime {

struct StridedCopyRecord {
    std::vector<std::int64_t> shape;
    std::vector<std::int64_t> strides;
    std::size_t element_bytes = 0;
    Device device = Device::cpu();
    std::uint64_t calls = 0;
    std::uint64_t elements = 0;
    std::uint64_t bytes = 0;
};

struct StridedCopyDiagnostics {
    std::vector<StridedCopyRecord> records;
    std::uint64_t calls = 0;
    std::uint64_t elements = 0;
    std::uint64_t bytes = 0;
};

void enable_strided_copy_diagnostics(bool enabled) noexcept;
void reset_strided_copy_diagnostics() noexcept;
[[nodiscard]] StridedCopyDiagnostics strided_copy_diagnostics();

}  // namespace microllm::runtime
