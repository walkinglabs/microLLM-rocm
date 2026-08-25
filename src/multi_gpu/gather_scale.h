#pragma once

#include <cstddef>
#include <cstdint>

namespace microllm::multi_gpu::hip {

struct GatherScaleDescriptor {
    const float* source = nullptr;
    std::int64_t destination_begin = 0;
    std::int64_t destination_end = 0;
};

static_assert(sizeof(GatherScaleDescriptor) % sizeof(float) == 0);

void launch_gather_scale(
    const GatherScaleDescriptor* descriptors,
    std::size_t descriptor_count,
    float* destination,
    std::int64_t destination_elements,
    float scale,
    void* stream);

}  // namespace microllm::multi_gpu::hip
