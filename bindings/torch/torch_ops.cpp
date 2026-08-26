#include <cstdint>
#include <stdexcept>
#include <tuple>
#include <utility>

#include <ATen/ATen.h>
#include <torch/library.h>

#include <microllm/ops/low_level.h>
#include <microllm/ops/ops.h>

#if defined(USE_ROCM)
#include <c10/hip/HIPStream.h>
#endif

namespace {

microllm::Device microllm_device(const at::Tensor& tensor) {
    if (tensor.device().is_cpu()) return microllm::Device::cpu();
#if defined(USE_ROCM) && MICROLLM_HAS_HIP
    if (tensor.device().is_cuda()) return microllm::Device::hip(tensor.get_device());
#endif
    throw std::invalid_argument("microLLM Torch Op supports CPU or PyTorch ROCm tensors");
}

microllm::DType microllm_dtype(const at::Tensor& tensor) {
    switch (tensor.scalar_type()) {
        case at::kFloat: return microllm::DType::Float32;
        case at::kHalf: return microllm::DType::Float16;
        case at::kBFloat16: return microllm::DType::BFloat16;
        default:
            throw std::invalid_argument(
                "microLLM Torch Op supports float32, float16, or bfloat16 tensors");
    }
}

void* current_stream(const at::Tensor& tensor) {
    if (tensor.device().is_cpu()) return nullptr;
#if defined(USE_ROCM) && MICROLLM_HAS_HIP
    return reinterpret_cast<void*>(c10::hip::getCurrentHIPStream(tensor.get_device()).stream());
#else
    throw std::invalid_argument("microLLM was not built with PyTorch ROCm stream support");
#endif
}

microllm::ConstTensorView const_view(const at::Tensor& tensor) {
    return {tensor.const_data_ptr(),
            microllm_dtype(tensor),
            microllm_device(tensor),
            {tensor.sizes().data(), static_cast<std::size_t>(tensor.dim())},
            {tensor.strides().data(), static_cast<std::size_t>(tensor.dim())}};
}

microllm::TensorView mutable_view(at::Tensor& tensor) {
    return {tensor.mutable_data_ptr(),
            microllm_dtype(tensor),
            microllm_device(tensor),
            {tensor.sizes().data(), static_cast<std::size_t>(tensor.dim())},
            {tensor.strides().data(), static_cast<std::size_t>(tensor.dim())}};
}

microllm::Tensor external_tensor(const at::Tensor& tensor) {
    const auto bytes = static_cast<std::size_t>(tensor.numel()) *
                       static_cast<std::size_t>(tensor.element_size());
    auto storage = microllm::Storage::from_external(
        const_cast<void*>(tensor.const_data_ptr()), bytes,
        microllm_device(tensor));
    return microllm::Tensor::from_storage(
        std::move(storage),
        microllm::Shape(tensor.sizes().begin(), tensor.sizes().end()),
        microllm::Strides(tensor.strides().begin(), tensor.strides().end()), 0,
        microllm_dtype(tensor));
}

template <typename Operation>
at::Tensor binary(const at::Tensor& left, const at::Tensor& right,
                  Operation&& operation) {
    TORCH_CHECK(left.scalar_type() == at::kFloat ||
                    left.scalar_type() == at::kHalf ||
                    left.scalar_type() == at::kBFloat16,
                "left must be float32, float16, or bfloat16");
    TORCH_CHECK(left.scalar_type() == right.scalar_type(),
                "input dtypes must match");
    TORCH_CHECK(left.sizes() == right.sizes(), "input shapes must match");
    TORCH_CHECK(left.device() == right.device(), "input devices must match");
    TORCH_CHECK(left.is_contiguous() && right.is_contiguous(), "inputs must be contiguous");
    auto output = at::empty_like(left);
    const auto context = left.device().is_cpu()
                             ? microllm::ops::OpContext{}
                             : microllm::ops::OpContext::from_external_stream(
                                   microllm_device(left), current_stream(left));
    operation(mutable_view(output), const_view(left), const_view(right), context);
    return output;
}

at::Tensor meta_binary(const at::Tensor& left, const at::Tensor& right) {
    TORCH_CHECK(left.scalar_type() == at::kFloat ||
                    left.scalar_type() == at::kHalf ||
                    left.scalar_type() == at::kBFloat16,
                "left must be float32, float16, or bfloat16");
    TORCH_CHECK(left.scalar_type() == right.scalar_type(),
                "input dtypes must match");
    TORCH_CHECK(left.sizes() == right.sizes(), "input shapes must match");
    TORCH_CHECK(left.device() == right.device(), "input devices must match");
    TORCH_CHECK(left.is_contiguous() && right.is_contiguous(),
                "inputs must be contiguous");
    return at::empty_like(left);
}

struct AddOperation {
    void operator()(microllm::TensorView output, microllm::ConstTensorView left,
                    microllm::ConstTensorView right,
                    const microllm::ops::OpContext& context) const {
        microllm::ops::add_out(output, left, right, context);
    }
};

struct MultiplyOperation {
    void operator()(microllm::TensorView output, microllm::ConstTensorView left,
                    microllm::ConstTensorView right,
                    const microllm::ops::OpContext& context) const {
        microllm::ops::multiply_out(output, left, right, context);
    }
};

at::Tensor add(const at::Tensor& left, const at::Tensor& right) {
    return binary(left, right, AddOperation{});
}

at::Tensor multiply(const at::Tensor& left, const at::Tensor& right) {
    return binary(left, right, MultiplyOperation{});
}

at::Tensor swiglu(const at::Tensor& gate, const at::Tensor& up) {
    TORCH_CHECK(gate.scalar_type() == at::kFloat ||
                    gate.scalar_type() == at::kHalf ||
                    gate.scalar_type() == at::kBFloat16,
                "gate must be float32, float16, or bfloat16");
    TORCH_CHECK(gate.scalar_type() == up.scalar_type(),
                "input dtypes must match");
    TORCH_CHECK(gate.sizes() == up.sizes(), "input shapes must match");
    TORCH_CHECK(gate.device() == up.device(), "input devices must match");
    TORCH_CHECK(gate.is_contiguous() && up.is_contiguous(),
                "inputs must be contiguous");
    auto output = at::empty_like(gate);
    if (gate.numel() == 0) return output;
    auto output_tensor = external_tensor(output);
    const auto gate_tensor = external_tensor(gate);
    const auto up_tensor = external_tensor(up);
    const auto context = gate.device().is_cpu()
                             ? microllm::ops::OpContext{}
                             : microllm::ops::OpContext::from_external_stream(
                                   microllm_device(gate), current_stream(gate));
    microllm::ops::swiglu_out_(
        output_tensor, gate_tensor, up_tensor, context);
    return output;
}

std::tuple<at::Tensor, at::Tensor> swiglu_backward(
    const at::Tensor& gate, const at::Tensor& up,
    const at::Tensor& gradient) {
    TORCH_CHECK(gate.scalar_type() == at::kFloat &&
                    up.scalar_type() == at::kFloat &&
                    gradient.scalar_type() == at::kFloat,
                "SwiGLU fused backward requires float32 tensors");
    TORCH_CHECK(gate.sizes() == up.sizes() &&
                    gate.sizes() == gradient.sizes(),
                "SwiGLU backward shapes must match");
    TORCH_CHECK(gate.device() == up.device() &&
                    gate.device() == gradient.device(),
                "SwiGLU backward devices must match");
    TORCH_CHECK(gate.is_contiguous() && up.is_contiguous() &&
                    gradient.is_contiguous(),
                "SwiGLU backward tensors must be contiguous");
    auto gate_gradient = at::empty_like(gate);
    auto up_gradient = at::empty_like(up);
    if (gate.numel() == 0) {
        return {std::move(gate_gradient), std::move(up_gradient)};
    }
    auto gate_gradient_tensor = external_tensor(gate_gradient);
    auto up_gradient_tensor = external_tensor(up_gradient);
    const auto gate_tensor = external_tensor(gate);
    const auto up_tensor = external_tensor(up);
    const auto gradient_tensor = external_tensor(gradient);
    const auto context = gate.device().is_cpu()
                             ? microllm::ops::OpContext{}
                             : microllm::ops::OpContext::from_external_stream(
                                   microllm_device(gate), current_stream(gate));
    microllm::ops::swiglu_backward_out_(
        gate_gradient_tensor, up_gradient_tensor, gate_tensor, up_tensor,
        gradient_tensor, context);
    return {std::move(gate_gradient), std::move(up_gradient)};
}

std::tuple<at::Tensor, at::Tensor> meta_swiglu_backward(
    const at::Tensor& gate, const at::Tensor& up,
    const at::Tensor& gradient) {
    TORCH_CHECK(gate.scalar_type() == at::kFloat &&
                    up.scalar_type() == at::kFloat &&
                    gradient.scalar_type() == at::kFloat,
                "SwiGLU fused backward requires float32 tensors");
    TORCH_CHECK(gate.sizes() == up.sizes() &&
                    gate.sizes() == gradient.sizes(),
                "SwiGLU backward shapes must match");
    TORCH_CHECK(gate.device() == up.device() &&
                    gate.device() == gradient.device(),
                "SwiGLU backward devices must match");
    return {at::empty_like(gate), at::empty_like(up)};
}

}  // namespace

TORCH_LIBRARY(microllm, library) {
    library.def("add(Tensor left, Tensor right) -> Tensor");
    library.def("multiply(Tensor left, Tensor right) -> Tensor");
    library.def("swiglu(Tensor gate, Tensor up) -> Tensor");
    library.def(
        "swiglu_backward(Tensor gate, Tensor up, Tensor gradient) -> (Tensor, Tensor)");
}

TORCH_LIBRARY_IMPL(microllm, CPU, library) {
    library.impl("add", &add);
    library.impl("multiply", &multiply);
    library.impl("swiglu", &swiglu);
    library.impl("swiglu_backward", &swiglu_backward);
}

TORCH_LIBRARY_IMPL(microllm, CUDA, library) {
    library.impl("add", &add);
    library.impl("multiply", &multiply);
    library.impl("swiglu", &swiglu);
    library.impl("swiglu_backward", &swiglu_backward);
}

TORCH_LIBRARY_IMPL(microllm, Meta, library) {
    library.impl("add", &meta_binary);
    library.impl("multiply", &meta_binary);
    library.impl("swiglu", &meta_binary);
    library.impl("swiglu_backward", &meta_swiglu_backward);
}
