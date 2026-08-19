#pragma once

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string_view>

namespace microllm {

enum class DType : std::uint8_t {
    Float32,
    Float16,
    BFloat16,
    Int32,
    Int64,
};

[[nodiscard]] constexpr std::size_t dtype_size(DType dtype) {
    switch (dtype) {
        case DType::Float32:
        case DType::Int32:
            return 4;
        case DType::Float16:
        case DType::BFloat16:
            return 2;
        case DType::Int64:
            return 8;
    }
    throw std::invalid_argument("unknown dtype");
}

[[nodiscard]] constexpr std::string_view dtype_name(DType dtype) {
    switch (dtype) {
        case DType::Float32: return "float32";
        case DType::Float16: return "float16";
        case DType::BFloat16: return "bfloat16";
        case DType::Int32: return "int32";
        case DType::Int64: return "int64";
    }
    return "unknown";
}

}  // namespace microllm
