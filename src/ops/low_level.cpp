#include <microllm/ops/low_level.h>

#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <string>

#include <microllm/core/tensor.h>
#include <microllm/runtime/runtime.h>

#if MICROLLM_HAS_HIP
#include "hip/kernels.h"
#endif

namespace microllm::ops {
namespace {

template <typename View>
void require_float32_contiguous(const View& view, const char* name) {
    if (view.dtype != DType::Float32) {
        throw std::invalid_argument(std::string(name) + " must be float32");
    }
    if (view.shape.size() != view.strides.size()) {
        throw std::invalid_argument(std::string(name) + " shape/stride rank mismatch");
    }
    const Shape shape(view.shape.begin(), view.shape.end());
    if (Strides(view.strides.begin(), view.strides.end()) != contiguous_strides(shape)) {
        throw std::invalid_argument(std::string(name) + " must be contiguous");
    }
    if (checked_numel(shape) != 0 && view.data == nullptr) {
        throw std::invalid_argument(std::string(name) + " data pointer is null");
    }
}

template <typename Launch>
void binary_out(TensorView output, ConstTensorView left, ConstTensorView right,
                const OpContext& context, Launch&& launch) {
    require_float32_contiguous(output, "output");
    require_float32_contiguous(left, "left");
    require_float32_contiguous(right, "right");
    if (output.shape.size() != left.shape.size() || output.shape.size() != right.shape.size() ||
        !std::equal(output.shape.begin(), output.shape.end(), left.shape.begin()) ||
        !std::equal(output.shape.begin(), output.shape.end(), right.shape.begin())) {
        throw std::invalid_argument("add/multiply TensorView shapes must match");
    }
    if (output.device != left.device || output.device != right.device) {
        throw std::invalid_argument("add/multiply TensorView devices must match");
    }
    const auto elements = checked_numel(Shape(output.shape.begin(), output.shape.end()));
    if (output.device.is_cpu()) {
        auto* destination = static_cast<float*>(output.data);
        const auto* left_values = static_cast<const float*>(left.data);
        const auto* right_values = static_cast<const float*>(right.data);
        for (std::int64_t index = 0; index < elements; ++index) {
            destination[index] = launch(left_values[index], right_values[index]);
        }
        return;
    }
#if MICROLLM_HAS_HIP
    runtime::set_device(output.device);
    launch(static_cast<float*>(output.data), static_cast<const float*>(left.data),
           static_cast<const float*>(right.data), elements,
           context.native_stream(output.device));
#else
    (void)context;
    throw std::runtime_error("HIP TensorView operator requested from a CPU-only build");
#endif
}

struct AddLaunch {
    float operator()(float left, float right) const { return left + right; }
#if MICROLLM_HAS_HIP
    void operator()(float* output, const float* left, const float* right,
                    std::int64_t elements, void* stream) const {
        hip::launch_add(left, right, output, elements, stream);
    }
#endif
};

struct MultiplyLaunch {
    float operator()(float left, float right) const { return left * right; }
#if MICROLLM_HAS_HIP
    void operator()(float* output, const float* left, const float* right,
                    std::int64_t elements, void* stream) const {
        hip::launch_multiply(left, right, output, elements, stream);
    }
#endif
};

}  // namespace

void add_out(TensorView output, ConstTensorView left, ConstTensorView right,
             const OpContext& context) {
    binary_out(output, left, right, context, AddLaunch{});
}

void multiply_out(TensorView output, ConstTensorView left, ConstTensorView right,
                  const OpContext& context) {
    binary_out(output, left, right, context, MultiplyLaunch{});
}

}  // namespace microllm::ops
