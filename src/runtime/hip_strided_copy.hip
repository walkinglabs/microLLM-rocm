#include "hip_strided_copy.h"

#include <cstddef>
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

__global__ void strided_copy_kernel(std::byte* destination, const std::byte* source,
                                    std::size_t element_bytes, std::int64_t elements,
                                    Layout layout) {
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
    for (std::size_t byte = 0; byte < element_bytes; ++byte) {
        destination[static_cast<std::size_t>(logical) * element_bytes + byte] =
            source[static_cast<std::size_t>(source_index) * element_bytes + byte];
    }
}

}  // namespace

void launch_strided_copy(void* destination, const void* source,
                         std::size_t element_bytes, std::int64_t elements,
                         std::int64_t rank, const std::int64_t* shape,
                         const std::int64_t* strides, void* stream) {
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
    hipLaunchKernelGGL(strided_copy_kernel, dim3(blocks), dim3(kBlockSize), 0,
                       reinterpret_cast<hipStream_t>(stream),
                       static_cast<std::byte*>(destination),
                       static_cast<const std::byte*>(source), element_bytes, elements, layout);
    const auto status = hipGetLastError();
    if (status != hipSuccess) {
        throw std::runtime_error(std::string("strided_copy_kernel: ") + hipGetErrorString(status));
    }
}

}  // namespace microllm::runtime::detail
