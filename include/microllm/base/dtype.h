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
    Float8E4M3FNUZ,
    Float8E5M2FNUZ,
    Int8,
    Int32,
    Int64,
};

[[nodiscard]] constexpr bool is_floating_point(DType dtype) noexcept {
    return dtype == DType::Float32 || dtype == DType::Float16 ||
           dtype == DType::BFloat16 || dtype == DType::Float8E4M3FNUZ ||
           dtype == DType::Float8E5M2FNUZ;
}

[[nodiscard]] constexpr bool is_fp8_fnuz(DType dtype) noexcept {
    return dtype == DType::Float8E4M3FNUZ || dtype == DType::Float8E5M2FNUZ;
}

[[nodiscard]] constexpr std::size_t dtype_size(DType dtype) {
    switch (dtype) {
        case DType::Float32:
        case DType::Int32:
            return 4;
        case DType::Float16:
        case DType::BFloat16:
            return 2;
        case DType::Float8E4M3FNUZ:
        case DType::Float8E5M2FNUZ:
        case DType::Int8:
            return 1;
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
        case DType::Float8E4M3FNUZ: return "float8_e4m3_fnuz";
        case DType::Float8E5M2FNUZ: return "float8_e5m2_fnuz";
        case DType::Int8: return "int8";
        case DType::Int32: return "int32";
        case DType::Int64: return "int64";
    }
    return "unknown";
}

}  // namespace microllm
