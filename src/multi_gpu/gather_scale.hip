#include "gather_scale.h"

#include <hip/hip_runtime.h>

#include <stdexcept>
#include <string>

namespace microllm::multi_gpu::hip {
namespace {

constexpr int kBlockSize = 256;

__global__ void gather_scale_kernel(
    const GatherScaleDescriptor* descriptors,
    std::size_t descriptor_count,
    float* destination,
    std::int64_t destination_elements,
    float scale) {
    const auto index = static_cast<std::int64_t>(
        blockIdx.x * blockDim.x + threadIdx.x);
    if (index >= destination_elements) return;

    std::size_t first = 0;
    std::size_t count = descriptor_count;
    while (count > 1) {
        const auto step = count / 2;
        const auto candidate = first + step;
        if (index >= descriptors[candidate].destination_begin) {
            first = candidate;
            count -= step;
        } else {
            count = step;
        }
    }
    const auto descriptor = descriptors[first];
    if (index >= descriptor.destination_end) return;
    destination[index] =
        descriptor.source[index - descriptor.destination_begin] * scale;
}

}  // namespace

void launch_gather_scale(
    const GatherScaleDescriptor* descriptors,
    std::size_t descriptor_count,
    float* destination,
    std::int64_t destination_elements,
    float scale,
    void* stream) {
    if (descriptor_count == 0 || destination_elements == 0) return;
    const auto blocks = static_cast<unsigned>(
        (destination_elements + kBlockSize - 1) / kBlockSize);
    hipLaunchKernelGGL(
        gather_scale_kernel, dim3(blocks), dim3(kBlockSize), 0,
        reinterpret_cast<hipStream_t>(stream), descriptors, descriptor_count,
        destination, destination_elements, scale);
    const auto status = hipGetLastError();
    if (status != hipSuccess) {
        throw std::runtime_error(
            std::string("gather_scale_kernel: ") + hipGetErrorString(status));
    }
}

}  // namespace microllm::multi_gpu::hip
