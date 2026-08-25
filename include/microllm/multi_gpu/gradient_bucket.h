#pragma once

#include <cstddef>
#include <memory>
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
    std::size_t gradient_view_count = 0;
    std::size_t pack_copy_calls = 0;
    std::size_t unpack_copy_calls = 0;
    std::size_t temporary_elements = 0;
    std::size_t temporary_bytes = 0;
    bool persistent_storage = false;
    bool plan_reused = false;
    bool overlap_enabled = false;
    std::size_t overlapped_bucket_count = 0;
    std::size_t plan_capacity_elements = 0;
    std::size_t plan_capacity_bytes = 0;
};

struct RankBucketStats {
    std::size_t bucket_count = 0;
    std::size_t parameter_count = 0;
    std::size_t total_elements = 0;
    std::size_t pack_copy_calls = 0;
    std::size_t unpack_copy_calls = 0;
    std::size_t gradient_view_count = 0;
    bool persistent_storage = false;
    bool plan_reused = false;
    std::size_t plan_capacity_elements = 0;
    std::size_t plan_capacity_bytes = 0;
};

// Owns reusable bucket and optional unpacked-gradient Storage for one process/rank.
// The plan binds to rank identity/device, parameter identity/order/shapes and
// one bucket limit and the view policy. clear() is required before changing
// that contract.
class RankGradientBucketPlan {
public:
    RankGradientBucketPlan();
    ~RankGradientBucketPlan();
    RankGradientBucketPlan(RankGradientBucketPlan&&) noexcept;
    RankGradientBucketPlan& operator=(RankGradientBucketPlan&&) noexcept;
    RankGradientBucketPlan(const RankGradientBucketPlan&) = delete;
    RankGradientBucketPlan& operator=(const RankGradientBucketPlan&) = delete;

    [[nodiscard]] bool initialized() const noexcept;
    void clear() noexcept;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;

    friend RankBucketStats all_reduce_rank_gradients(
        RankCommunicator& communicator,
        const std::vector<autograd::Value*>& parameters,
        std::size_t maximum_bucket_bytes,
        RankGradientBucketPlan* persistent_plan,
        bool gradient_views);
};

// Owns reusable rank-local bucket and unpacked-gradient Storage. A plan binds to
// one communicator, parameter identity/order, shape set, and bucket limit. Call
// clear() before deliberately changing any part of that contract.
class GradientBucketPlan {
public:
    GradientBucketPlan();
    ~GradientBucketPlan();
    GradientBucketPlan(GradientBucketPlan&&) noexcept;
    GradientBucketPlan& operator=(GradientBucketPlan&&) noexcept;
    GradientBucketPlan(const GradientBucketPlan&) = delete;
    GradientBucketPlan& operator=(const GradientBucketPlan&) = delete;

    [[nodiscard]] bool initialized() const noexcept;
    [[nodiscard]] bool overlap_active() const noexcept;
    void clear() noexcept;
    void begin_overlap_step(
        Communicator& communicator,
        const std::vector<std::vector<autograd::Value*>>& rank_parameters);
    void mark_parameter_ready(std::size_t rank, std::size_t parameter_index);
    [[nodiscard]] BucketStats finish_overlap_step();

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;

    friend BucketStats all_reduce_gradients(
        Communicator& communicator,
        const std::vector<std::vector<autograd::Value*>>& rank_parameters,
        std::size_t maximum_bucket_bytes, bool in_place_average,
        GradientBucketPlan* persistent_plan, bool gradient_views);
};

[[nodiscard]] BucketStats all_reduce_gradients(
    Communicator& communicator,
    const std::vector<std::vector<autograd::Value*>>& rank_parameters,
    std::size_t maximum_bucket_bytes, bool in_place_average = true,
    GradientBucketPlan* persistent_plan = nullptr,
    bool gradient_views = false);

[[nodiscard]] RankBucketStats all_reduce_rank_gradients(
    RankCommunicator& communicator,
    const std::vector<autograd::Value*>& parameters,
    std::size_t maximum_bucket_bytes,
    RankGradientBucketPlan* persistent_plan = nullptr,
    bool gradient_views = false);

}  // namespace microllm::multi_gpu
