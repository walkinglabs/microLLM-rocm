#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

#include <microllm/base/device.h>

namespace microllm::runtime {

enum class AllocationSource : std::uint8_t {
    Unspecified,
    ModelEmbedding,
    AttentionNorm,
    AttentionProjection,
    AttentionLayout,
    AttentionCore,
    AttentionOutput,
    AttentionResidual,
    FfnNorm,
    Ffn,
    FfnResidual,
    ModelFinalNorm,
    ModelOutput
};

[[nodiscard]] const char* allocation_source_name(
    AllocationSource source) noexcept;

class ScopedAllocationSource {
public:
    explicit ScopedAllocationSource(AllocationSource source) noexcept;
    ~ScopedAllocationSource();
    ScopedAllocationSource(const ScopedAllocationSource&) = delete;
    ScopedAllocationSource& operator=(const ScopedAllocationSource&) = delete;
    ScopedAllocationSource(ScopedAllocationSource&&) = delete;
    ScopedAllocationSource& operator=(ScopedAllocationSource&&) = delete;

private:
    AllocationSource previous_ = AllocationSource::Unspecified;
    bool active_ = false;
};

struct AllocationSourceRecord {
    AllocationSource source = AllocationSource::Unspecified;
    Device device = Device::cpu();
    std::size_t allocation_bytes = 0;
    std::uint64_t calls = 0;
    std::uint64_t total_bytes = 0;
};

struct AllocationSourceDiagnostics {
    std::vector<AllocationSourceRecord> records;
    std::uint64_t calls = 0;
    std::uint64_t bytes = 0;
};

void enable_allocation_source_diagnostics(bool enabled) noexcept;
void reset_allocation_source_diagnostics() noexcept;
[[nodiscard]] AllocationSourceDiagnostics allocation_source_diagnostics();

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
