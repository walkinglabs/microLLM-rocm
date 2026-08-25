#include <microllm/multi_gpu/gradient_bucket.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

#include <microllm/runtime/memory.h>

namespace microllm::multi_gpu {
namespace {

struct BucketRange {
    std::size_t first_parameter = 0;
    std::size_t end_parameter = 0;
    std::size_t elements = 0;
};

void validate(const Communicator& communicator,
              const std::vector<std::vector<autograd::Value*>>& ranks) {
    if (ranks.size() != communicator.size()) {
        throw std::invalid_argument("gradient buckets need one parameter list per rank");
    }
    if (ranks.empty()) return;
    const auto parameter_count = ranks.front().size();
    for (std::size_t rank = 0; rank < ranks.size(); ++rank) {
        if (ranks[rank].size() != parameter_count) {
            throw std::invalid_argument("all ranks need the same parameter count");
        }
        for (std::size_t index = 0; index < parameter_count; ++index) {
            const auto* parameter = ranks[rank][index];
            if (parameter == nullptr || !parameter->has_grad() ||
                parameter->grad().dtype() != DType::Float32 ||
                !parameter->grad().is_contiguous() ||
                parameter->grad().device() != Device::hip(communicator.devices()[rank])) {
                throw std::invalid_argument(
                    "bucket parameters need matching contiguous rank-local HIP gradients");
            }
            if (rank != 0 && parameter->grad().shape() != ranks[0][index]->grad().shape()) {
                throw std::invalid_argument("bucket gradient shapes differ across ranks");
            }
        }
    }
}

std::vector<BucketRange> make_ranges(
    const std::vector<autograd::Value*>& parameters,
    std::size_t maximum_elements) {
    std::vector<BucketRange> ranges;
    std::size_t first_parameter = 0;
    while (first_parameter < parameters.size()) {
        std::size_t end_parameter = first_parameter;
        std::size_t bucket_elements = 0;
        while (end_parameter < parameters.size()) {
            const auto elements = static_cast<std::size_t>(
                parameters[end_parameter]->grad().numel());
            if (elements > std::numeric_limits<std::size_t>::max() - bucket_elements) {
                throw std::overflow_error("gradient bucket element count overflow");
            }
            if (end_parameter != first_parameter &&
                bucket_elements + elements > maximum_elements) {
                break;
            }
            bucket_elements += elements;
            ++end_parameter;
            if (bucket_elements >= maximum_elements) break;
        }
        ranges.push_back({first_parameter, end_parameter, bucket_elements});
        first_parameter = end_parameter;
    }
    return ranges;
}

void add_elements(std::size_t& destination, std::size_t elements) {
    if (elements > std::numeric_limits<std::size_t>::max() - destination) {
        throw std::overflow_error("gradient bucket storage element count overflow");
    }
    destination += elements;
}

std::size_t bytes_for(std::size_t elements, const char* description) {
    if (elements > std::numeric_limits<std::size_t>::max() / sizeof(float)) {
        throw std::overflow_error(description);
    }
    return elements * sizeof(float);
}

}  // namespace

struct GradientBucketPlan::Impl {
    struct PersistentRange {
        std::size_t first_parameter = 0;
        std::size_t end_parameter = 0;
        std::size_t elements = 0;
        std::vector<Tensor> buckets;
        std::vector<std::vector<Tensor>> gradients;
        std::vector<runtime::Event> ready_events;
        std::vector<std::size_t> remaining_parameters;
        std::vector<std::vector<bool>> parameter_ready;
        bool overlap_enqueued = false;
    };

    bool initialized = false;
    bool gradient_views = false;
    std::size_t maximum_bucket_bytes = 0;
    std::vector<int> devices;
    std::vector<std::vector<autograd::Value*>> parameters;
    std::vector<PersistentRange> ranges;
    std::size_t capacity_elements = 0;
    bool overlap_active = false;
    Communicator* overlap_communicator = nullptr;
    std::vector<std::vector<autograd::Value*>> overlap_parameters;
    std::size_t overlap_enqueued_buckets = 0;
};

struct RankGradientBucketPlan::Impl {
    struct PersistentRange {
        std::size_t first_parameter = 0;
        std::size_t end_parameter = 0;
        std::size_t elements = 0;
        Tensor bucket;
        std::vector<Tensor> gradients;
    };

    bool initialized = false;
    int rank = -1;
    int world_size = 0;
    Device device = Device::cpu();
    std::size_t maximum_bucket_bytes = 0;
    std::vector<autograd::Value*> parameters;
    std::vector<PersistentRange> ranges;
    std::size_t capacity_elements = 0;
};

GradientBucketPlan::GradientBucketPlan() : impl_(std::make_unique<Impl>()) {}
GradientBucketPlan::~GradientBucketPlan() = default;
GradientBucketPlan::GradientBucketPlan(GradientBucketPlan&&) noexcept = default;
GradientBucketPlan& GradientBucketPlan::operator=(GradientBucketPlan&&) noexcept = default;

bool GradientBucketPlan::initialized() const noexcept {
    return impl_ != nullptr && impl_->initialized;
}

bool GradientBucketPlan::overlap_active() const noexcept {
    return impl_ != nullptr && impl_->overlap_active;
}

void GradientBucketPlan::clear() noexcept {
    if (!impl_) return;
    impl_->ranges.clear();
    impl_->parameters.clear();
    impl_->devices.clear();
    impl_->maximum_bucket_bytes = 0;
    impl_->gradient_views = false;
    impl_->capacity_elements = 0;
    impl_->initialized = false;
    impl_->overlap_active = false;
    impl_->overlap_communicator = nullptr;
    impl_->overlap_parameters.clear();
    impl_->overlap_enqueued_buckets = 0;
}

RankGradientBucketPlan::RankGradientBucketPlan()
    : impl_(std::make_unique<Impl>()) {}
RankGradientBucketPlan::~RankGradientBucketPlan() = default;
RankGradientBucketPlan::RankGradientBucketPlan(
    RankGradientBucketPlan&&) noexcept = default;
RankGradientBucketPlan& RankGradientBucketPlan::operator=(
    RankGradientBucketPlan&&) noexcept = default;

bool RankGradientBucketPlan::initialized() const noexcept {
    return impl_ != nullptr && impl_->initialized;
}

void RankGradientBucketPlan::clear() noexcept {
    if (!impl_) return;
    impl_->ranges.clear();
    impl_->parameters.clear();
    impl_->rank = -1;
    impl_->world_size = 0;
    impl_->device = Device::cpu();
    impl_->maximum_bucket_bytes = 0;
    impl_->capacity_elements = 0;
    impl_->initialized = false;
}

void GradientBucketPlan::begin_overlap_step(
    Communicator& communicator,
    const std::vector<std::vector<autograd::Value*>>& rank_parameters) {
    if (!impl_ || !impl_->initialized || !impl_->gradient_views) {
        throw std::logic_error(
            "gradient overlap requires an initialized bucket-view plan");
    }
    if (impl_->overlap_active) {
        throw std::logic_error("gradient overlap step is already active");
    }
    if (communicator.devices() != impl_->devices ||
        rank_parameters != impl_->parameters) {
        throw std::invalid_argument(
            "gradient overlap communicator or parameter contract changed");
    }
    impl_->overlap_active = true;
    impl_->overlap_communicator = &communicator;
    impl_->overlap_parameters = rank_parameters;
    impl_->overlap_enqueued_buckets = 0;
    for (auto& range : impl_->ranges) {
        if (range.ready_events.empty()) {
            range.ready_events.reserve(communicator.size());
            for (const auto device : communicator.devices()) {
                range.ready_events.emplace_back(Device::hip(device), false);
            }
        }
        range.remaining_parameters.assign(
            communicator.size(),
            range.end_parameter - range.first_parameter);
        range.parameter_ready.assign(
            communicator.size(),
            std::vector<bool>(range.end_parameter - range.first_parameter,
                              false));
        range.overlap_enqueued = false;
    }
}

void GradientBucketPlan::mark_parameter_ready(
    std::size_t rank, std::size_t parameter_index) {
    if (!impl_ || !impl_->overlap_active ||
        impl_->overlap_communicator == nullptr) {
        throw std::logic_error("gradient overlap step is not active");
    }
    if (rank >= impl_->overlap_parameters.size()) {
        throw std::out_of_range("gradient overlap rank is out of range");
    }
    auto found = std::find_if(
        impl_->ranges.begin(), impl_->ranges.end(),
        [&](const auto& range) {
            return parameter_index >= range.first_parameter &&
                   parameter_index < range.end_parameter;
        });
    if (found == impl_->ranges.end()) {
        throw std::out_of_range("gradient overlap parameter is out of range");
    }
    const auto local = parameter_index - found->first_parameter;
    if (found->parameter_ready[rank][local]) {
        throw std::logic_error("gradient overlap parameter readiness was duplicated");
    }
    found->parameter_ready[rank][local] = true;
    --found->remaining_parameters[rank];
    if (found->remaining_parameters[rank] == 0) {
        found->ready_events[rank].record_default_stream();
    }
    if (found->overlap_enqueued ||
        std::any_of(found->remaining_parameters.begin(),
                    found->remaining_parameters.end(),
                    [](std::size_t remaining) { return remaining != 0; })) {
        return;
    }

    auto& communicator = *impl_->overlap_communicator;
    for (std::size_t current_rank = 0;
         current_rank < communicator.size(); ++current_rank) {
        found->ready_events[current_rank].wait(
            communicator.stream(current_rank));
        std::size_t offset = 0;
        for (std::size_t parameter = found->first_parameter;
             parameter < found->end_parameter; ++parameter) {
            const auto& gradient =
                impl_->overlap_parameters[current_rank][parameter]->grad();
            const auto elements = static_cast<std::size_t>(gradient.numel());
            const auto bytes = bytes_for(
                elements, "gradient overlap pack bytes overflow");
            auto* destination =
                static_cast<std::byte*>(found->buckets[current_rank].data()) +
                offset * sizeof(float);
            runtime::copy_bytes_async(
                destination, found->buckets[current_rank].device(),
                gradient.data(), gradient.device(), bytes,
                communicator.stream(current_rank));
            offset += elements;
        }
    }
    communicator.enqueue_all_reduce_average_in_place(found->buckets);
    found->overlap_enqueued = true;
    ++impl_->overlap_enqueued_buckets;
}

BucketStats GradientBucketPlan::finish_overlap_step() {
    if (!impl_ || !impl_->overlap_active ||
        impl_->overlap_communicator == nullptr) {
        throw std::logic_error("gradient overlap step is not active");
    }
    if (impl_->overlap_enqueued_buckets != impl_->ranges.size()) {
        throw std::logic_error(
            "gradient overlap finished before every bucket was ready");
    }
    impl_->overlap_communicator->synchronize();
    BucketStats stats;
    stats.parameter_count = impl_->parameters.front().size();
    stats.bucket_count = impl_->ranges.size();
    stats.persistent_storage = true;
    stats.plan_reused = true;
    stats.overlap_enabled = true;
    stats.overlapped_bucket_count = impl_->overlap_enqueued_buckets;
    stats.plan_capacity_elements = impl_->capacity_elements;
    stats.plan_capacity_bytes = bytes_for(
        impl_->capacity_elements, "gradient overlap plan capacity bytes overflow");
    for (auto& range : impl_->ranges) {
        add_elements(stats.total_elements, range.elements);
        stats.bucket_tensor_count += impl_->overlap_communicator->size();
        for (std::size_t rank = 0; rank < range.gradients.size(); ++rank) {
            stats.gradient_view_count += range.gradients[rank].size();
            stats.pack_copy_calls += range.gradients[rank].size();
            for (std::size_t local = 0;
                 local < range.gradients[rank].size(); ++local) {
                impl_->overlap_parameters[rank]
                    [range.first_parameter + local]->set_grad(
                        range.gradients[rank][local]);
            }
        }
    }
    impl_->overlap_active = false;
    impl_->overlap_communicator = nullptr;
    impl_->overlap_parameters.clear();
    return stats;
}

BucketStats all_reduce_gradients(
    Communicator& communicator,
    const std::vector<std::vector<autograd::Value*>>& rank_parameters,
    std::size_t maximum_bucket_bytes, bool in_place_average,
    GradientBucketPlan* persistent_plan, bool gradient_views) {
    if (maximum_bucket_bytes < sizeof(float)) {
        throw std::invalid_argument("maximum bucket size must hold at least one float");
    }
    if (persistent_plan != nullptr && !in_place_average) {
        throw std::invalid_argument(
            "persistent gradient buckets require in-place averaging");
    }
    if (gradient_views && persistent_plan == nullptr) {
        throw std::invalid_argument(
            "gradient bucket views require persistent gradient buckets");
    }
    validate(communicator, rank_parameters);
    if (rank_parameters.empty() || rank_parameters.front().empty()) return {};
    const auto maximum_elements = maximum_bucket_bytes / sizeof(float);
    const auto ranges = make_ranges(rank_parameters.front(), maximum_elements);

    BucketStats stats;
    stats.parameter_count = rank_parameters.front().size();
    stats.bucket_count = ranges.size();
    for (const auto& range : ranges) add_elements(stats.total_elements, range.elements);

    if (persistent_plan != nullptr) {
        if (!persistent_plan->impl_) {
            persistent_plan->impl_ = std::make_unique<GradientBucketPlan::Impl>();
        }
        auto& plan = *persistent_plan->impl_;
        const bool reused = plan.initialized;
        if (reused) {
            if (plan.maximum_bucket_bytes != maximum_bucket_bytes ||
                plan.gradient_views != gradient_views ||
                plan.devices != communicator.devices() ||
                plan.parameters != rank_parameters ||
                plan.ranges.size() != ranges.size()) {
                throw std::invalid_argument(
                    "persistent gradient bucket contract changed; clear the plan first");
            }
            for (std::size_t index = 0; index < ranges.size(); ++index) {
                const auto& expected = ranges[index];
                const auto& actual = plan.ranges[index];
                if (actual.first_parameter != expected.first_parameter ||
                    actual.end_parameter != expected.end_parameter ||
                    actual.elements != expected.elements) {
                    throw std::invalid_argument(
                        "persistent gradient bucket layout changed; clear the plan first");
                }
            }
        } else {
            GradientBucketPlan::Impl candidate;
            candidate.maximum_bucket_bytes = maximum_bucket_bytes;
            candidate.gradient_views = gradient_views;
            candidate.devices = communicator.devices();
            candidate.parameters = rank_parameters;
            candidate.ranges.reserve(ranges.size());
            for (const auto& range : ranges) {
                if (range.elements > static_cast<std::size_t>(
                        std::numeric_limits<std::int64_t>::max())) {
                    throw std::overflow_error(
                        "gradient bucket shape exceeds int64");
                }
                GradientBucketPlan::Impl::PersistentRange persistent_range;
                persistent_range.first_parameter = range.first_parameter;
                persistent_range.end_parameter = range.end_parameter;
                persistent_range.elements = range.elements;
                persistent_range.buckets.reserve(communicator.size());
                persistent_range.gradients.resize(communicator.size());
                for (std::size_t rank = 0; rank < communicator.size(); ++rank) {
                    persistent_range.buckets.emplace_back(
                        Shape{static_cast<std::int64_t>(range.elements)},
                        DType::Float32, Device::hip(communicator.devices()[rank]));
                    add_elements(candidate.capacity_elements, range.elements);
                    auto& gradients = persistent_range.gradients[rank];
                    gradients.reserve(range.end_parameter - range.first_parameter);
                    std::size_t offset = 0;
                    for (std::size_t parameter = range.first_parameter;
                         parameter < range.end_parameter; ++parameter) {
                        const auto& gradient = rank_parameters[rank][parameter]->grad();
                        const auto elements = static_cast<std::size_t>(gradient.numel());
                        if (gradient_views) {
                            gradients.emplace_back(Tensor::from_storage(
                                persistent_range.buckets.back().storage(),
                                gradient.shape(), contiguous_strides(gradient.shape()),
                                static_cast<std::int64_t>(offset), DType::Float32));
                        } else {
                            gradients.emplace_back(gradient.shape(), DType::Float32,
                                                   gradient.device());
                            add_elements(candidate.capacity_elements, elements);
                        }
                        add_elements(offset, elements);
                    }
                }
                candidate.ranges.push_back(std::move(persistent_range));
            }
            candidate.initialized = true;
            plan = std::move(candidate);
        }

        stats.persistent_storage = true;
        stats.plan_reused = reused;
        stats.plan_capacity_elements = plan.capacity_elements;
        stats.plan_capacity_bytes = bytes_for(
            plan.capacity_elements, "gradient bucket plan capacity bytes overflow");
        for (auto& range : plan.ranges) {
            stats.bucket_tensor_count += communicator.size();
            for (std::size_t rank = 0; rank < communicator.size(); ++rank) {
                std::size_t offset = 0;
                for (std::size_t parameter = range.first_parameter;
                     parameter < range.end_parameter; ++parameter) {
                    const auto& gradient = rank_parameters[rank][parameter]->grad();
                    const auto elements = static_cast<std::size_t>(gradient.numel());
                    const auto bytes = bytes_for(
                        elements, "gradient bucket pack bytes overflow");
                    auto* destination =
                        static_cast<std::byte*>(range.buckets[rank].data()) +
                        offset * sizeof(float);
                    runtime::copy_bytes_async(
                        destination, range.buckets[rank].device(), gradient.data(),
                        gradient.device(), bytes, communicator.stream(rank));
                    ++stats.pack_copy_calls;
                    offset += elements;
                }
            }
            communicator.all_reduce(range.buckets, true, true);
            if (plan.gradient_views) {
                for (const auto& gradients : range.gradients) {
                    stats.gradient_view_count += gradients.size();
                }
            } else {
                for (std::size_t rank = 0; rank < communicator.size(); ++rank) {
                    std::size_t offset = 0;
                    auto& gradients = range.gradients[rank];
                    stats.unpacked_tensor_count += gradients.size();
                    for (auto& gradient : gradients) {
                        const auto elements = static_cast<std::size_t>(gradient.numel());
                        const auto bytes = bytes_for(
                            elements, "gradient bucket unpack bytes overflow");
                        const auto* source =
                            static_cast<const std::byte*>(range.buckets[rank].data()) +
                            offset * sizeof(float);
                        runtime::copy_bytes_async(
                            gradient.data(), gradient.device(), source,
                            range.buckets[rank].device(), bytes,
                            communicator.stream(rank));
                        ++stats.unpack_copy_calls;
                        offset += elements;
                    }
                }
                communicator.synchronize();
            }
            for (std::size_t rank = 0; rank < communicator.size(); ++rank) {
                for (std::size_t local = 0;
                     local < range.gradients[rank].size(); ++local) {
                    rank_parameters[rank][range.first_parameter + local]->set_grad(
                        range.gradients[rank][local]);
                }
            }
        }
        return stats;
    }

    for (const auto& range : ranges) {
        std::vector<Tensor> buckets;
        buckets.reserve(communicator.size());
        for (std::size_t rank = 0; rank < communicator.size(); ++rank) {
            buckets.emplace_back(Shape{static_cast<std::int64_t>(range.elements)},
                                 DType::Float32,
                                 Device::hip(communicator.devices()[rank]));
            ++stats.bucket_tensor_count;
            add_elements(stats.temporary_elements, range.elements);
            std::size_t offset = 0;
            for (std::size_t index = range.first_parameter;
                 index < range.end_parameter; ++index) {
                const auto& gradient = rank_parameters[rank][index]->grad();
                const auto elements = static_cast<std::size_t>(gradient.numel());
                const auto bytes = bytes_for(
                    elements, "gradient bucket pack bytes overflow");
                auto* destination = static_cast<std::byte*>(buckets.back().data()) +
                                    offset * sizeof(float);
                runtime::copy_bytes_async(
                    destination, buckets.back().device(), gradient.data(),
                    gradient.device(), bytes, communicator.stream(rank));
                ++stats.pack_copy_calls;
                offset += elements;
            }
        }
        communicator.all_reduce(buckets, true, in_place_average);
        if (!in_place_average) {
            stats.average_tensor_count += communicator.size();
            for (std::size_t rank = 0; rank < communicator.size(); ++rank) {
                add_elements(stats.temporary_elements, range.elements);
            }
        }

        std::vector<std::vector<Tensor>> unpacked(communicator.size());
        for (std::size_t rank = 0; rank < communicator.size(); ++rank) {
            std::size_t offset = 0;
            unpacked[rank].reserve(range.end_parameter - range.first_parameter);
            for (std::size_t index = range.first_parameter;
                 index < range.end_parameter; ++index) {
                const auto shape = rank_parameters[rank][index]->grad().shape();
                unpacked[rank].emplace_back(
                    shape, DType::Float32,
                    Device::hip(communicator.devices()[rank]));
                ++stats.unpacked_tensor_count;
                const auto elements = static_cast<std::size_t>(
                    unpacked[rank].back().numel());
                add_elements(stats.temporary_elements, elements);
                const auto bytes = bytes_for(
                    elements, "gradient bucket unpack bytes overflow");
                const auto* source =
                    static_cast<const std::byte*>(buckets[rank].data()) +
                    offset * sizeof(float);
                runtime::copy_bytes_async(
                    unpacked[rank].back().data(), unpacked[rank].back().device(),
                    source, buckets[rank].device(), bytes,
                    communicator.stream(rank));
                ++stats.unpack_copy_calls;
                offset += elements;
            }
        }
        communicator.synchronize();
        for (std::size_t rank = 0; rank < communicator.size(); ++rank) {
            for (std::size_t local = 0; local < unpacked[rank].size(); ++local) {
                rank_parameters[rank][range.first_parameter + local]->set_grad(
                    std::move(unpacked[rank][local]));
            }
        }
    }
    stats.temporary_bytes = bytes_for(
        stats.temporary_elements, "gradient bucket temporary bytes overflow");
    return stats;
}

RankBucketStats all_reduce_rank_gradients(
    RankCommunicator& communicator,
    const std::vector<autograd::Value*>& parameters,
    std::size_t maximum_bucket_bytes,
    RankGradientBucketPlan* persistent_plan) {
    if (maximum_bucket_bytes < sizeof(float)) {
        throw std::invalid_argument(
            "rank gradient bucket must hold at least one float");
    }
    for (const auto* parameter : parameters) {
        if (parameter == nullptr || !parameter->has_grad() ||
            parameter->grad().dtype() != DType::Float32 ||
            parameter->grad().device() != communicator.device() ||
            !parameter->grad().is_contiguous()) {
            throw std::invalid_argument(
                "rank gradient buckets require contiguous local float32 gradients");
        }
    }
    RankBucketStats stats;
    stats.parameter_count = parameters.size();
    if (parameters.empty()) return stats;
    const auto ranges = make_ranges(
        parameters, maximum_bucket_bytes / sizeof(float));
    if (persistent_plan != nullptr) {
        if (!persistent_plan->impl_) {
            persistent_plan->impl_ =
                std::make_unique<RankGradientBucketPlan::Impl>();
        }
        auto& plan = *persistent_plan->impl_;
        const bool reused = plan.initialized;
        if (reused) {
            if (plan.rank != communicator.rank() ||
                plan.world_size != communicator.world_size() ||
                plan.device != communicator.device() ||
                plan.maximum_bucket_bytes != maximum_bucket_bytes ||
                plan.parameters != parameters ||
                plan.ranges.size() != ranges.size()) {
                throw std::invalid_argument(
                    "persistent rank gradient bucket contract changed; "
                    "clear the plan first");
            }
            for (std::size_t index = 0; index < ranges.size(); ++index) {
                const auto& expected = ranges[index];
                const auto& actual = plan.ranges[index];
                if (actual.first_parameter != expected.first_parameter ||
                    actual.end_parameter != expected.end_parameter ||
                    actual.elements != expected.elements) {
                    throw std::invalid_argument(
                        "persistent rank gradient bucket layout changed; "
                        "clear the plan first");
                }
                for (std::size_t local = 0;
                     local < actual.gradients.size(); ++local) {
                    const auto parameter = actual.first_parameter + local;
                    if (actual.gradients[local].shape() !=
                        parameters[parameter]->grad().shape()) {
                        throw std::invalid_argument(
                            "persistent rank gradient shape changed; "
                            "clear the plan first");
                    }
                }
            }
        } else {
            RankGradientBucketPlan::Impl candidate;
            candidate.rank = communicator.rank();
            candidate.world_size = communicator.world_size();
            candidate.device = communicator.device();
            candidate.maximum_bucket_bytes = maximum_bucket_bytes;
            candidate.parameters = parameters;
            candidate.ranges.reserve(ranges.size());
            for (const auto& range : ranges) {
                if (range.elements > static_cast<std::size_t>(
                        std::numeric_limits<std::int64_t>::max())) {
                    throw std::overflow_error(
                        "persistent rank gradient bucket shape exceeds int64");
                }
                RankGradientBucketPlan::Impl::PersistentRange persistent;
                persistent.first_parameter = range.first_parameter;
                persistent.end_parameter = range.end_parameter;
                persistent.elements = range.elements;
                persistent.bucket = Tensor(
                    Shape{static_cast<std::int64_t>(range.elements)},
                    DType::Float32, communicator.device());
                add_elements(candidate.capacity_elements, range.elements);
                persistent.gradients.reserve(
                    range.end_parameter - range.first_parameter);
                for (std::size_t index = range.first_parameter;
                     index < range.end_parameter; ++index) {
                    const auto& gradient = parameters[index]->grad();
                    persistent.gradients.emplace_back(
                        gradient.shape(), DType::Float32,
                        communicator.device());
                    add_elements(candidate.capacity_elements,
                                 static_cast<std::size_t>(gradient.numel()));
                }
                candidate.ranges.push_back(std::move(persistent));
            }
            candidate.initialized = true;
            plan = std::move(candidate);
        }

        stats.persistent_storage = true;
        stats.plan_reused = reused;
        stats.plan_capacity_elements = plan.capacity_elements;
        stats.plan_capacity_bytes = bytes_for(
            plan.capacity_elements,
            "persistent rank gradient bucket capacity bytes overflow");
        for (auto& range : plan.ranges) {
            std::size_t offset = 0;
            for (std::size_t index = range.first_parameter;
                 index < range.end_parameter; ++index) {
                const auto& gradient = parameters[index]->grad();
                const auto elements = static_cast<std::size_t>(gradient.numel());
                const auto bytes = bytes_for(
                    elements, "persistent rank gradient pack bytes overflow");
                auto* destination =
                    static_cast<std::byte*>(range.bucket.data()) +
                    offset * sizeof(float);
                runtime::copy_bytes_async(
                    destination, range.bucket.device(), gradient.data(),
                    gradient.device(), bytes, communicator.stream());
                ++stats.pack_copy_calls;
                offset += elements;
            }
            communicator.enqueue_all_reduce_average_in_place(range.bucket);
            offset = 0;
            for (auto& gradient : range.gradients) {
                const auto elements = static_cast<std::size_t>(gradient.numel());
                const auto bytes = bytes_for(
                    elements, "persistent rank gradient unpack bytes overflow");
                const auto* source =
                    static_cast<const std::byte*>(range.bucket.data()) +
                    offset * sizeof(float);
                runtime::copy_bytes_async(
                    gradient.data(), gradient.device(), source,
                    range.bucket.device(), bytes, communicator.stream());
                ++stats.unpack_copy_calls;
                offset += elements;
            }
            communicator.synchronize();
            for (std::size_t local = 0;
                 local < range.gradients.size(); ++local) {
                parameters[range.first_parameter + local]->set_grad(
                    range.gradients[local]);
            }
            ++stats.bucket_count;
            add_elements(stats.total_elements, range.elements);
        }
        return stats;
    }
    for (const auto& range : ranges) {
        if (range.elements > static_cast<std::size_t>(
                std::numeric_limits<std::int64_t>::max())) {
            throw std::overflow_error("rank gradient bucket shape exceeds int64");
        }
        Tensor bucket(
            Shape{static_cast<std::int64_t>(range.elements)},
            DType::Float32, communicator.device());
        std::vector<Tensor> unpacked;
        unpacked.reserve(range.end_parameter - range.first_parameter);
        std::size_t offset = 0;
        for (std::size_t index = range.first_parameter;
             index < range.end_parameter; ++index) {
            const auto& gradient = parameters[index]->grad();
            const auto elements = static_cast<std::size_t>(gradient.numel());
            const auto bytes = bytes_for(
                elements, "rank gradient pack bytes overflow");
            auto* destination = static_cast<std::byte*>(bucket.data()) +
                                offset * sizeof(float);
            runtime::copy_bytes_async(
                destination, bucket.device(), gradient.data(),
                gradient.device(), bytes, communicator.stream());
            unpacked.emplace_back(
                gradient.shape(), DType::Float32, communicator.device());
            ++stats.pack_copy_calls;
            offset += elements;
        }
        communicator.enqueue_all_reduce_average_in_place(bucket);
        offset = 0;
        for (auto& gradient : unpacked) {
            const auto elements = static_cast<std::size_t>(gradient.numel());
            const auto bytes = bytes_for(
                elements, "rank gradient unpack bytes overflow");
            const auto* source = static_cast<const std::byte*>(bucket.data()) +
                                 offset * sizeof(float);
            runtime::copy_bytes_async(
                gradient.data(), gradient.device(), source, bucket.device(),
                bytes, communicator.stream());
            ++stats.unpack_copy_calls;
            offset += elements;
        }
        communicator.synchronize();
        for (std::size_t local = 0; local < unpacked.size(); ++local) {
            parameters[range.first_parameter + local]->set_grad(
                std::move(unpacked[local]));
        }
        ++stats.bucket_count;
        add_elements(stats.total_elements, range.elements);
    }
    return stats;
}

}  // namespace microllm::multi_gpu
