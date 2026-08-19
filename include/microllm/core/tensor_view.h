#pragma once

#include <cstddef>
#include <cstdint>
#include <span>

#include <microllm/base/device.h>
#include <microllm/base/dtype.h>

namespace microllm {

struct TensorView {
    void* data = nullptr;
    DType dtype = DType::Float32;
    Device device = Device::cpu();
    std::span<const std::int64_t> shape;
    std::span<const std::int64_t> strides;
};

struct ConstTensorView {
    const void* data = nullptr;
    DType dtype = DType::Float32;
    Device device = Device::cpu();
    std::span<const std::int64_t> shape;
    std::span<const std::int64_t> strides;
};

}  // namespace microllm
