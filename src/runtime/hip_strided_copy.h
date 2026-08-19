#pragma once

#include <cstdint>

namespace microllm::runtime::detail {

void launch_strided_copy_float32(float* destination, const float* source,
                                 std::int64_t elements, std::int64_t rank,
                                 const std::int64_t* shape, const std::int64_t* strides);

}  // namespace microllm::runtime::detail
