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
};

[[nodiscard]] AllocationStats allocation_stats(Device device) noexcept;
void reset_allocation_peak(Device device) noexcept;

void copy_bytes(void* destination, Device destination_device, const void* source,
                Device source_device, std::size_t num_bytes);

void copy_strided_float32(void* contiguous_destination, const void* strided_source,
                          Device device, std::span<const std::int64_t> shape,
                          std::span<const std::int64_t> strides);

}  // namespace microllm::runtime
