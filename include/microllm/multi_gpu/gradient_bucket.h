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
};

[[nodiscard]] BucketStats all_reduce_gradients(
    Communicator& communicator,
    const std::vector<std::vector<autograd::Value*>>& rank_parameters,
    std::size_t maximum_bucket_bytes);

}  // namespace microllm::multi_gpu
