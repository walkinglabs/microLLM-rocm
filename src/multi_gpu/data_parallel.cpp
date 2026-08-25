#include <microllm/multi_gpu/data_parallel.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <stdexcept>
#include <utility>

#include <microllm/profiling/trace.h>
#include <microllm/runtime/runtime.h>

namespace microllm::multi_gpu {
namespace {

using Clock = std::chrono::steady_clock;

double elapsed_ms(Clock::time_point start, Clock::time_point finish) {
    return std::chrono::duration<double, std::milli>(finish - start).count();
}

Tensor scalar(float value) { return Tensor::from_vector({value}, {}); }

void validate_batches(const std::vector<io::TokenBatch>& batches,
                      std::size_t world_size) {
    if (batches.size() != world_size) {
        throw std::invalid_argument("data parallel step needs one batch per rank");
    }
    if (batches.empty()) return;
    const auto elements = batches.front().targets.numel();
    const auto valid_targets = [](const Tensor& targets) {
        const auto values = targets.to_int32_vector();
        return static_cast<std::size_t>(std::count_if(
            values.begin(), values.end(), [](std::int32_t value) { return value != -100; }));
    };
    const auto reference_valid_targets = valid_targets(batches.front().targets);
    if (reference_valid_targets == 0) {
        throw std::invalid_argument("rank batches need at least one non-ignored target");
    }
    for (const auto& batch : batches) {
        if (batch.inputs.shape() != batch.targets.shape() ||
            batch.inputs.dtype() != DType::Int32 ||
            batch.targets.dtype() != DType::Int32) {
            throw std::invalid_argument("rank inputs and targets must be matching int32 tensors");
        }
        if (batch.targets.numel() != elements) {
            throw std::invalid_argument(
                "all ranks need equal target counts for unweighted gradient averaging");
        }
        if (valid_targets(batch.targets) != reference_valid_targets) {
            throw std::invalid_argument(
                "all ranks need equal non-ignored target counts for gradient averaging");
        }
    }
}

float maximum_parameter_difference(
    const std::vector<std::unique_ptr<model::TransformerModel>>& models) {
    if (models.size() < 2) return 0.0F;
    const auto reference = models.front()->parameters();
    float maximum = 0.0F;
    for (std::size_t rank = 1; rank < models.size(); ++rank) {
        const auto parameters = models[rank]->parameters();
        if (parameters.size() != reference.size()) {
            throw std::logic_error("rank model parameter counts diverged");
        }
        for (std::size_t parameter = 0; parameter < reference.size(); ++parameter) {
            const auto left = reference[parameter]->data().to_vector();
            const auto right = parameters[parameter]->data().to_vector();
            for (std::size_t index = 0; index < left.size(); ++index) {
                maximum = std::max(maximum, std::abs(left[index] - right[index]));
            }
        }
    }
    return maximum;
}

}  // namespace

struct DataParallelTrainer::Impl {
    model::ModelConfig model_config;
    DataParallelConfig config;
    std::vector<std::unique_ptr<model::TransformerModel>> models;
    std::vector<std::unique_ptr<training::AdamW>> optimizers;
    Communicator communicator;

    Impl(model::ModelConfig model_configuration, std::uint64_t seed,
         DataParallelConfig trainer_config)
        : model_config(std::move(model_configuration)),
          config(std::move(trainer_config)),
          communicator(config.device_indices) {
        if (config.device_indices.size() < 2) {
            throw std::invalid_argument("data parallel training requires at least two devices");
        }
        if (config.maximum_bucket_bytes < sizeof(float)) {
            throw std::invalid_argument("data parallel bucket must hold at least one float");
        }
        models.reserve(config.device_indices.size());
        optimizers.reserve(config.device_indices.size());
        for (const auto device_index : config.device_indices) {
            auto rank_model = std::make_unique<model::TransformerModel>(model_config, seed);
            rank_model->to(Device::hip(device_index));
            models.push_back(std::move(rank_model));
        }
        for (auto& rank_model : models) {
            optimizers.push_back(std::make_unique<training::AdamW>(
                rank_model->parameters(), config.optimizer));
        }
    }
};

DataParallelTrainer::DataParallelTrainer(model::ModelConfig model_config,
                                         std::uint64_t seed,
                                         DataParallelConfig config)
    : impl_(std::make_unique<Impl>(std::move(model_config), seed,
                                   std::move(config))) {}
DataParallelTrainer::~DataParallelTrainer() = default;
DataParallelTrainer::DataParallelTrainer(DataParallelTrainer&&) noexcept = default;
DataParallelTrainer& DataParallelTrainer::operator=(DataParallelTrainer&&) noexcept = default;

std::size_t DataParallelTrainer::world_size() const noexcept { return impl_->models.size(); }
const DataParallelConfig& DataParallelTrainer::config() const noexcept { return impl_->config; }
model::TransformerModel& DataParallelTrainer::model(std::size_t rank) {
    return *impl_->models.at(rank);
}
const model::TransformerModel& DataParallelTrainer::model(std::size_t rank) const {
    return *impl_->models.at(rank);
}

DistributedStepMetrics DataParallelTrainer::step(
    const std::vector<io::TokenBatch>& rank_batches,
    std::uint64_t step_number) {
    validate_batches(rank_batches, world_size());
    DistributedStepMetrics metrics;
    metrics.step = step_number;
    metrics.rank_losses.resize(world_size());
    const auto total_start = Clock::now();
    profiling::TraceTimer total_timer(profiling::TraceKind::Model,
                                      "data_parallel.step", Device::cpu());

    const auto forward_start = Clock::now();
    for (std::size_t rank = 0; rank < world_size(); ++rank) {
        impl_->optimizers[rank]->zero_grad();
        const auto loss = impl_->models[rank]->loss(rank_batches[rank].inputs,
                                                   rank_batches[rank].targets);
        metrics.rank_losses[rank] = loss.data().to_vector()[0];
        loss.backward();
    }
    // Correctness baseline: communication streams must not read gradients before
    // default-stream backward work is complete. Gradient-ready event overlap is a
    // separate milestone.
    for (const auto device : impl_->config.device_indices) {
        runtime::synchronize(Device::hip(device));
    }
    const auto forward_finish = Clock::now();
    metrics.forward_backward_ms = elapsed_ms(forward_start, forward_finish);
    for (const auto loss : metrics.rank_losses) metrics.mean_loss += loss;
    metrics.mean_loss /= static_cast<float>(world_size());
    if (auto* trace = profiling::TraceSession::current(); trace != nullptr) {
        trace->record(profiling::TraceKind::Layer, "data_parallel.forward_backward",
                      scalar(metrics.mean_loss), metrics.forward_backward_ms);
    }

    const auto communication_start = Clock::now();
    std::vector<std::vector<autograd::Value*>> rank_parameters;
    rank_parameters.reserve(world_size());
    for (auto& rank_model : impl_->models) rank_parameters.push_back(rank_model->parameters());
    metrics.buckets = all_reduce_gradients(
        impl_->communicator, rank_parameters, impl_->config.maximum_bucket_bytes);
    const auto communication_finish = Clock::now();
    metrics.communication_ms = elapsed_ms(communication_start, communication_finish);
    if (auto* trace = profiling::TraceSession::current(); trace != nullptr) {
        trace->record(profiling::TraceKind::Layer, "data_parallel.all_reduce",
                      scalar(static_cast<float>(metrics.buckets.bucket_count)),
                      metrics.communication_ms);
    }

    const auto optimizer_start = Clock::now();
    for (auto& optimizer : impl_->optimizers) optimizer->step();
    // Parameter verification previously supplied an accidental device wait via
    // to_vector(). A skipped audit must not change step completion or temporary
    // lifetimes, so optimizer completion is an explicit stage boundary.
    for (const auto device : impl_->config.device_indices) {
        runtime::synchronize(Device::hip(device));
    }
    const auto optimizer_finish = Clock::now();
    metrics.optimizer_ms = elapsed_ms(optimizer_start, optimizer_finish);
    metrics.parameter_check_performed =
        impl_->config.parameter_check_interval != 0 &&
        step_number % impl_->config.parameter_check_interval == 0;
    if (metrics.parameter_check_performed) {
        const auto verification_start = Clock::now();
        metrics.maximum_parameter_difference =
            maximum_parameter_difference(impl_->models);
        metrics.verification_ms = elapsed_ms(
            verification_start, Clock::now());
    }
    if (auto* trace = profiling::TraceSession::current(); trace != nullptr) {
        trace->record(profiling::TraceKind::Layer, "data_parallel.optimizer",
                      scalar(static_cast<float>(impl_->optimizers.size())),
                      metrics.optimizer_ms);
        trace->record(profiling::TraceKind::Layer,
                      "data_parallel.parameter_verification",
                      scalar(metrics.maximum_parameter_difference),
                      metrics.verification_ms);
    }
    metrics.total_ms = elapsed_ms(total_start, Clock::now());
    total_timer.finish(scalar(metrics.mean_loss));
    return metrics;
}

void DataParallelTrainer::abort() noexcept {
    if (impl_) impl_->communicator.abort();
}

}  // namespace microllm::multi_gpu
