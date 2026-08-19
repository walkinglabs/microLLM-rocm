#include <cstdint>
#include <stdexcept>

#include <torch/extension.h>

#include <microllm/ops/low_level.h>

#if defined(USE_ROCM)
#include <c10/hip/HIPStream.h>
#endif

namespace {

microllm::Device device_of(const torch::Tensor& tensor) {
    if (tensor.device().is_cpu()) return microllm::Device::cpu();
#if defined(USE_ROCM) && MICROLLM_HAS_HIP
    if (tensor.device().is_cuda()) return microllm::Device::hip(tensor.get_device());
#endif
    throw std::invalid_argument("microLLM Torch Op supports CPU or PyTorch ROCm tensors");
}

void* current_stream(const torch::Tensor& tensor) {
    if (tensor.device().is_cpu()) return nullptr;
#if defined(USE_ROCM) && MICROLLM_HAS_HIP
    return reinterpret_cast<void*>(c10::hip::getCurrentHIPStream(tensor.get_device()).stream());
#else
    throw std::invalid_argument("microLLM was not built with PyTorch ROCm stream support");
#endif
}

microllm::ConstTensorView const_view(const torch::Tensor& tensor) {
    return {tensor.const_data_ptr(),
            microllm::DType::Float32,
            device_of(tensor),
            {tensor.sizes().data(), static_cast<std::size_t>(tensor.dim())},
            {tensor.strides().data(), static_cast<std::size_t>(tensor.dim())}};
}

microllm::TensorView mutable_view(torch::Tensor& tensor) {
    return {tensor.mutable_data_ptr(),
            microllm::DType::Float32,
            device_of(tensor),
            {tensor.sizes().data(), static_cast<std::size_t>(tensor.dim())},
            {tensor.strides().data(), static_cast<std::size_t>(tensor.dim())}};
}

template <typename Operation>
torch::Tensor binary(const torch::Tensor& left, const torch::Tensor& right,
                     Operation&& operation) {
    TORCH_CHECK(left.scalar_type() == torch::kFloat32, "left must be float32");
    TORCH_CHECK(right.scalar_type() == torch::kFloat32, "right must be float32");
    TORCH_CHECK(left.sizes() == right.sizes(), "input shapes must match");
    TORCH_CHECK(left.device() == right.device(), "input devices must match");
    TORCH_CHECK(left.is_contiguous() && right.is_contiguous(), "inputs must be contiguous");
    auto output = torch::empty_like(left);
    const auto context = left.device().is_cpu()
                             ? microllm::ops::OpContext{}
                             : microllm::ops::OpContext::from_external_stream(
                                   device_of(left), current_stream(left));
    operation(mutable_view(output), const_view(left), const_view(right), context);
    return output;
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

torch::Tensor add(const torch::Tensor& left, const torch::Tensor& right) {
    return binary(left, right, AddOperation{});
}

torch::Tensor multiply(const torch::Tensor& left, const torch::Tensor& right) {
    return binary(left, right, MultiplyOperation{});
}

}  // namespace

TORCH_LIBRARY(microllm, library) {
    library.def("add(Tensor left, Tensor right) -> Tensor");
    library.def("multiply(Tensor left, Tensor right) -> Tensor");
}

TORCH_LIBRARY_IMPL(microllm, CPU, library) {
    library.impl("add", &add);
    library.impl("multiply", &multiply);
}

TORCH_LIBRARY_IMPL(microllm, CUDA, library) {
    library.impl("add", &add);
    library.impl("multiply", &multiply);
}
