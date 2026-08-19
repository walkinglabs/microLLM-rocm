#pragma once

#include <cstddef>

#include <microllm/base/device.h>

namespace microllm::runtime {

[[nodiscard]] void* allocate(std::size_t num_bytes, Device device);
void deallocate(void* pointer, Device device) noexcept;

void copy_bytes(void* destination, Device destination_device, const void* source,
                Device source_device, std::size_t num_bytes);

}  // namespace microllm::runtime
