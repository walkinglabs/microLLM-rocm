#pragma once

#include <bit>
#include <cstdint>

namespace microllm {

[[nodiscard]] inline std::uint16_t float_to_float16_bits(float value) noexcept {
    const auto bits = std::bit_cast<std::uint32_t>(value);
    const auto sign = static_cast<std::uint16_t>((bits >> 16U) & 0x8000U);
    const auto exponent = static_cast<int>((bits >> 23U) & 0xffU) - 127 + 15;
    auto mantissa = bits & 0x007fffffU;
    if (exponent <= 0) {
        if (exponent < -10) return sign;
        mantissa |= 0x00800000U;
        const auto shift = static_cast<unsigned>(14 - exponent);
        auto rounded = mantissa >> shift;
        const auto remainder = mantissa & ((1U << shift) - 1U);
        const auto halfway = 1U << (shift - 1U);
        if (remainder > halfway || (remainder == halfway && (rounded & 1U) != 0)) ++rounded;
        return static_cast<std::uint16_t>(sign | rounded);
    }
    if (exponent >= 31) {
        if ((bits & 0x7fffffffU) > 0x7f800000U) {
            return static_cast<std::uint16_t>(sign | 0x7e00U);
        }
        return static_cast<std::uint16_t>(sign | 0x7c00U);
    }
    auto rounded = mantissa + 0x00000fffU + ((mantissa >> 13U) & 1U);
    auto half_exponent = exponent;
    if ((rounded & 0x00800000U) != 0) {
        rounded = 0;
        ++half_exponent;
        if (half_exponent >= 31) return static_cast<std::uint16_t>(sign | 0x7c00U);
    }
    return static_cast<std::uint16_t>(
        sign | (static_cast<unsigned>(half_exponent) << 10U) | (rounded >> 13U));
}

[[nodiscard]] inline float float16_bits_to_float(std::uint16_t value) noexcept {
    const auto sign = static_cast<std::uint32_t>(value & 0x8000U) << 16U;
    auto exponent = static_cast<std::uint32_t>((value >> 10U) & 0x1fU);
    auto mantissa = static_cast<std::uint32_t>(value & 0x03ffU);
    std::uint32_t bits = 0;
    if (exponent == 0U) {
        if (mantissa == 0U) {
            bits = sign;
        } else {
            int normalized_exponent = -14;
            while ((mantissa & 0x0400U) == 0U) {
                mantissa <<= 1U;
                --normalized_exponent;
            }
            mantissa &= 0x03ffU;
            bits = sign |
                   (static_cast<std::uint32_t>(normalized_exponent + 127) << 23U) |
                   (mantissa << 13U);
        }
    } else if (exponent == 31U) {
        bits = sign | 0x7f800000U | (mantissa << 13U);
    } else {
        bits = sign | ((exponent + 112U) << 23U) | (mantissa << 13U);
    }
    return std::bit_cast<float>(bits);
}

[[nodiscard]] inline std::uint16_t float_to_bfloat16_bits(float value) noexcept {
    auto bits = std::bit_cast<std::uint32_t>(value);
    if ((bits & 0x7fffffffU) > 0x7f800000U) return 0x7fc0U;
    bits += 0x7fffU + ((bits >> 16U) & 1U);
    return static_cast<std::uint16_t>(bits >> 16U);
}

[[nodiscard]] inline float bfloat16_bits_to_float(std::uint16_t value) noexcept {
    return std::bit_cast<float>(static_cast<std::uint32_t>(value) << 16U);
}

struct Float16 {
    std::uint16_t bits = 0;
    Float16() = default;
    explicit Float16(float value) : bits(float_to_float16_bits(value)) {}
    [[nodiscard]] explicit operator float() const noexcept {
        return float16_bits_to_float(bits);
    }
};

struct BFloat16 {
    std::uint16_t bits = 0;
    BFloat16() = default;
    explicit BFloat16(float value) : bits(float_to_bfloat16_bits(value)) {}
    [[nodiscard]] explicit operator float() const noexcept {
        return bfloat16_bits_to_float(bits);
    }
};

static_assert(sizeof(Float16) == 2);
static_assert(sizeof(BFloat16) == 2);

}  // namespace microllm
