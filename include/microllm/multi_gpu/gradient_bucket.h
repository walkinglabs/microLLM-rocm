#pragma once

#include <cstddef>
#include <vector>

#include <microllm/autograd/autograd.h>
#include <microllm/multi_gpu/communicator.h>

namespace microllm::multi_gpu {

struct BucketStats {
    std::size_t bucket_count = 0;
    std::size_t parameter_count = 0;
    std::size_t total_elements = 0;
    std::size_t bucket_tensor_count = 0;
    std::size_t average_tensor_count = 0;
    std::size_t unpacked_tensor_count = 0;
    std::size_t pack_copy_calls = 0;
    std::size_t unpack_copy_calls = 0;
    std::size_t temporary_elements = 0;
    std::size_t temporary_bytes = 0;
};

[[nodiscard]] BucketStats all_reduce_gradients(
    Communicator& communicator,
    const std::vector<std::vector<autograd::Value*>>& rank_parameters,
    std::size_t maximum_bucket_bytes);

}  // namespace microllm::multi_gpu
