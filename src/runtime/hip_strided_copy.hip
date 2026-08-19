#include "hip_strided_copy.h"

#include <stdexcept>
#include <string>

#include <hip/hip_runtime.h>

namespace microllm::runtime::detail {
namespace {

constexpr unsigned kBlockSize = 256;
constexpr std::int64_t kMaximumRank = 8;

struct Layout {
    std::int64_t rank = 0;
    std::int64_t shape[kMaximumRank]{};
    std::int64_t strides[kMaximumRank]{};
};

__global__ void strided_copy_kernel(float* destination, const float* source,
                                    std::int64_t elements, Layout layout) {
    const auto logical = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (logical >= elements) return;
    auto remainder = logical;
    std::int64_t source_index = 0;
    for (std::int64_t reversed = layout.rank; reversed > 0; --reversed) {
        const auto dim = reversed - 1;
        const auto coordinate = remainder % layout.shape[dim];
        remainder /= layout.shape[dim];
        source_index += coordinate * layout.strides[dim];
    }
    destination[logical] = source[source_index];
}

}  // namespace

void launch_strided_copy_float32(float* destination, const float* source,
                                 std::int64_t elements, std::int64_t rank,
                                 const std::int64_t* shape, const std::int64_t* strides) {
    if (rank < 0 || rank > kMaximumRank) {
        throw std::invalid_argument("HIP strided copy supports ranks up to eight");
    }
    if (elements == 0) return;
    Layout layout;
    layout.rank = rank;
    for (std::int64_t dim = 0; dim < rank; ++dim) {
        layout.shape[dim] = shape[dim];
        layout.strides[dim] = strides[dim];
    }
    const auto block_size = static_cast<std::int64_t>(kBlockSize);
    const auto blocks = static_cast<unsigned>((elements + block_size - 1) / block_size);
    hipLaunchKernelGGL(strided_copy_kernel, dim3(blocks), dim3(kBlockSize), 0, nullptr,
                       destination, source, elements, layout);
    const auto status = hipGetLastError();
    if (status != hipSuccess) {
        throw std::runtime_error(std::string("strided_copy_kernel: ") + hipGetErrorString(status));
    }
}

}  // namespace microllm::runtime::detail
