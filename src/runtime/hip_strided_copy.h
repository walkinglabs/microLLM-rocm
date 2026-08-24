#pragma once

#include <cstddef>
#include <cstdint>

namespace microllm::runtime::detail {

void launch_strided_copy(void* destination, const void* source,
                         std::size_t element_bytes, std::int64_t elements,
                         std::int64_t rank, const std::int64_t* shape,
                         const std::int64_t* strides, void* stream);

}  // namespace microllm::runtime::detail
