#include <microllm/ops/low_level.h>

#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <string>

#include <microllm/base/low_precision.h>
#include <microllm/core/tensor.h>
#include <microllm/runtime/runtime.h>

#if MICROLLM_HAS_HIP
#include "hip/kernels.h"
#endif

namespace microllm::ops {
namespace {

template <typename View>
void require_forward_float_contiguous(const View& view, const char* name) {
    if (view.dtype != DType::Float32 && view.dtype != DType::Float16 &&
        view.dtype != DType::BFloat16) {
        throw std::invalid_argument(
            std::string(name) + " must be float32, float16, or bfloat16");
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

template <typename Scalar, typename Operation>
void binary_cpu(void* output, const void* left, const void* right,
                std::int64_t elements, Operation&& operation) {
    auto* destination = static_cast<Scalar*>(output);
    const auto* left_values = static_cast<const Scalar*>(left);
    const auto* right_values = static_cast<const Scalar*>(right);
    for (std::int64_t index = 0; index < elements; ++index) {
        destination[index] = Scalar(operation(
            static_cast<float>(left_values[index]),
            static_cast<float>(right_values[index])));
    }
}

template <typename Launch>
void binary_out(TensorView output, ConstTensorView left, ConstTensorView right,
                const OpContext& context, Launch&& launch) {
    require_forward_float_contiguous(output, "output");
    require_forward_float_contiguous(left, "left");
    require_forward_float_contiguous(right, "right");
    if (output.dtype != left.dtype || output.dtype != right.dtype) {
        throw std::invalid_argument("add/multiply TensorView dtypes must match");
    }
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
        switch (output.dtype) {
            case DType::Float32:
                binary_cpu<float>(output.data, left.data, right.data, elements, launch);
                break;
            case DType::Float16:
                binary_cpu<Float16>(output.data, left.data, right.data, elements, launch);
                break;
            case DType::BFloat16:
                binary_cpu<BFloat16>(output.data, left.data, right.data, elements, launch);
                break;
            default:
                throw std::logic_error("validated TensorView dtype became invalid");
        }
        return;
    }
#if MICROLLM_HAS_HIP
    runtime::set_device(output.device);
    launch(output.data, left.data, right.data, output.dtype, elements,
           context.native_stream(output.device));
#else
    (void)context;
    throw std::runtime_error("HIP TensorView operator requested from a CPU-only build");
#endif
}

struct AddLaunch {
    float operator()(float left, float right) const { return left + right; }
#if MICROLLM_HAS_HIP
    void operator()(void* output, const void* left, const void* right,
                    DType dtype, std::int64_t elements, void* stream) const {
        hip::launch_add_typed(left, right, output, dtype, elements, stream);
    }
#endif
};

struct MultiplyLaunch {
    float operator()(float left, float right) const { return left * right; }
#if MICROLLM_HAS_HIP
    void operator()(void* output, const void* left, const void* right,
                    DType dtype, std::int64_t elements, void* stream) const {
        hip::launch_multiply_typed(left, right, output, dtype, elements, stream);
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
