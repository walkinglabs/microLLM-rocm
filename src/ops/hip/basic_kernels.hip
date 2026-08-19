#include "kernels.h"

#include <stdexcept>
#include <string>

#include <hip/hip_runtime.h>

namespace microllm::ops::hip {
namespace {

constexpr unsigned kBlockSize = 256;

void check_launch(const char* operation) {
    const auto status = hipGetLastError();
    if (status != hipSuccess) {
        throw std::runtime_error(std::string(operation) + ": " + hipGetErrorString(status));
    }
}

hipStream_t as_stream(void* stream) { return reinterpret_cast<hipStream_t>(stream); }

unsigned grid_size(std::int64_t elements) {
    const auto block_size = static_cast<std::int64_t>(kBlockSize);
    return static_cast<unsigned>((elements + block_size - 1) / block_size);
}

__global__ void fill_kernel(float* output, std::int64_t elements, float value) {
    const auto index = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index < elements) output[index] = value;
}

__global__ void add_kernel(const float* left, const float* right, float* output,
                           std::int64_t elements) {
    const auto index = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index < elements) output[index] = left[index] + right[index];
}

__global__ void multiply_kernel(const float* left, const float* right, float* output,
                                std::int64_t elements) {
    const auto index = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index < elements) output[index] = left[index] * right[index];
}

__global__ void scale_kernel(const float* input, float* output, std::int64_t elements,
                             float factor) {
    const auto index = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index < elements) output[index] = input[index] * factor;
}

__global__ void matmul_kernel(const float* left, const float* right, float* output,
                              std::int64_t batches, std::int64_t rows,
                              std::int64_t inner, std::int64_t columns) {
    const auto index = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const auto elements = batches * rows * columns;
    if (index >= elements) return;
    const auto column = index % columns;
    const auto row_and_batch = index / columns;
    const auto row = row_and_batch % rows;
    const auto batch = row_and_batch / rows;
    const auto left_base = batch * rows * inner;
    const auto right_base = batch * inner * columns;
    float sum = 0.0F;
    for (std::int64_t reduction = 0; reduction < inner; ++reduction) {
        sum += left[left_base + row * inner + reduction] *
               right[right_base + reduction * columns + column];
    }
    output[index] = sum;
}

}  // namespace

void launch_fill(float* output, std::int64_t elements, float value, void* stream) {
    if (elements == 0) return;
    hipLaunchKernelGGL(fill_kernel, dim3(grid_size(elements)), dim3(kBlockSize), 0,
                       as_stream(stream), output, elements, value);
    check_launch("fill_kernel");
}

void launch_add(const float* left, const float* right, float* output,
                std::int64_t elements, void* stream) {
    if (elements == 0) return;
    hipLaunchKernelGGL(add_kernel, dim3(grid_size(elements)), dim3(kBlockSize), 0,
                       as_stream(stream), left, right, output, elements);
    check_launch("add_kernel");
}

void launch_multiply(const float* left, const float* right, float* output,
                     std::int64_t elements, void* stream) {
    if (elements == 0) return;
    hipLaunchKernelGGL(multiply_kernel, dim3(grid_size(elements)), dim3(kBlockSize), 0,
                       as_stream(stream), left, right, output, elements);
    check_launch("multiply_kernel");
}

void launch_scale(const float* input, float* output, std::int64_t elements, float factor,
                  void* stream) {
    if (elements == 0) return;
    hipLaunchKernelGGL(scale_kernel, dim3(grid_size(elements)), dim3(kBlockSize), 0,
                       as_stream(stream), input, output, elements, factor);
    check_launch("scale_kernel");
}

void launch_matmul(const float* left, const float* right, float* output,
                   std::int64_t batches, std::int64_t rows, std::int64_t inner,
                   std::int64_t columns, void* stream) {
    const auto elements = batches * rows * columns;
    if (elements == 0) return;
    hipLaunchKernelGGL(matmul_kernel, dim3(grid_size(elements)), dim3(kBlockSize), 0,
                       as_stream(stream), left, right, output, batches, rows, inner, columns);
    check_launch("matmul_kernel");
}

}  // namespace microllm::ops::hip
