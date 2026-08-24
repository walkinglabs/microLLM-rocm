#pragma once

#include <cstddef>
#include <cstdint>
#include <span>

#include <microllm/base/device.h>

namespace microllm::runtime {

[[nodiscard]] void* allocate(std::size_t num_bytes, Device device);
void deallocate(void* pointer, Device device, std::size_t num_bytes) noexcept;

struct AllocationStats {
    std::size_t current_bytes = 0;
    std::size_t peak_bytes = 0;
    std::size_t total_allocated_bytes = 0;
    std::size_t allocation_calls = 0;
    std::size_t deallocation_calls = 0;
    std::size_t backend_allocation_calls = 0;
    std::size_t backend_deallocation_calls = 0;
    std::size_t cache_reuse_calls = 0;
    std::size_t cached_bytes = 0;
    std::size_t reserved_bytes = 0;
};

[[nodiscard]] AllocationStats allocation_stats(Device device) noexcept;
void reset_allocation_peak(Device device) noexcept;

void copy_bytes(void* destination, Device destination_device, const void* source,
                Device source_device, std::size_t num_bytes);
// Enqueues a copy on a caller-owned native HIP stream. A null stream selects
// the legacy default stream. Host memory must remain valid until the copy ends.
void copy_bytes_async_native(
    void* destination, Device destination_device, const void* source,
    Device source_device, std::size_t num_bytes, void* native_stream = nullptr);

void copy_strided(void* contiguous_destination, const void* strided_source,
                  std::size_t element_bytes, Device device,
                  std::span<const std::int64_t> shape,
                  std::span<const std::int64_t> strides);

}  // namespace microllm::runtime
