#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

#include <microllm/io/token_dataset.h>
#include <microllm/model/model.h>
#include <microllm/multi_gpu/communicator.h>
#include <microllm/multi_gpu/gradient_bucket.h>
#include <microllm/training/optimizer.h>

namespace microllm::multi_gpu {

struct DataParallelConfig {
    std::vector<int> device_indices;
    std::size_t maximum_bucket_bytes = 25U * 1024U * 1024U;
    // 1 preserves the correctness-first every-step audit. 0 explicitly
    // disables it; N checks steps divisible by N.
    std::size_t parameter_check_interval = 1;
    bool in_place_bucket_average = true;
    training::AdamWConfig optimizer;
};

struct DistributedStepMetrics {
    std::uint64_t step = 0;
    float mean_loss = 0.0F;
    std::vector<float> rank_losses;
    BucketStats buckets;
    bool parameter_check_performed = false;
    float maximum_parameter_difference = 0.0F;
    double forward_backward_ms = 0.0;
    double communication_ms = 0.0;
    std::size_t communication_allocation_calls = 0;
    std::size_t communication_backend_allocation_calls = 0;
    std::size_t communication_cache_reuse_calls = 0;
    std::size_t communication_total_allocated_bytes = 0;
    double optimizer_ms = 0.0;
    double verification_ms = 0.0;
    double total_ms = 0.0;
};

class DataParallelTrainer {
public:
    DataParallelTrainer(model::ModelConfig model_config, std::uint64_t seed,
                        DataParallelConfig config);
    ~DataParallelTrainer();
    DataParallelTrainer(DataParallelTrainer&&) noexcept;
    DataParallelTrainer& operator=(DataParallelTrainer&&) noexcept;
    DataParallelTrainer(const DataParallelTrainer&) = delete;
    DataParallelTrainer& operator=(const DataParallelTrainer&) = delete;

    [[nodiscard]] std::size_t world_size() const noexcept;
    [[nodiscard]] const DataParallelConfig& config() const noexcept;
    [[nodiscard]] model::TransformerModel& model(std::size_t rank);
    [[nodiscard]] const model::TransformerModel& model(std::size_t rank) const;
    [[nodiscard]] DistributedStepMetrics step(
        const std::vector<io::TokenBatch>& rank_batches,
        std::uint64_t step_number);
    void abort() noexcept;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace microllm::multi_gpu
