#include <microllm/multi_gpu/gradient_bucket.h>

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
        std::vector<std::vector<Tensor>> unpacked;
    };

    bool initialized = false;
    std::size_t maximum_bucket_bytes = 0;
    std::vector<int> devices;
    std::vector<std::vector<autograd::Value*>> parameters;
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

void GradientBucketPlan::clear() noexcept {
    if (!impl_) return;
    impl_->ranges.clear();
    impl_->parameters.clear();
    impl_->devices.clear();
    impl_->maximum_bucket_bytes = 0;
    impl_->capacity_elements = 0;
    impl_->initialized = false;
}

BucketStats all_reduce_gradients(
    Communicator& communicator,
    const std::vector<std::vector<autograd::Value*>>& rank_parameters,
    std::size_t maximum_bucket_bytes, bool in_place_average,
    GradientBucketPlan* persistent_plan) {
    if (maximum_bucket_bytes < sizeof(float)) {
        throw std::invalid_argument("maximum bucket size must hold at least one float");
    }
    if (persistent_plan != nullptr && !in_place_average) {
        throw std::invalid_argument(
            "persistent gradient buckets require in-place averaging");
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
            candidate.devices = communicator.devices();
            candidate.parameters = rank_parameters;
            candidate.ranges.reserve(ranges.size());
            for (const auto& range : ranges) {
                GradientBucketPlan::Impl::PersistentRange persistent_range;
                persistent_range.first_parameter = range.first_parameter;
                persistent_range.end_parameter = range.end_parameter;
                persistent_range.elements = range.elements;
                persistent_range.buckets.reserve(communicator.size());
                persistent_range.unpacked.resize(communicator.size());
                for (std::size_t rank = 0; rank < communicator.size(); ++rank) {
                    persistent_range.buckets.emplace_back(
                        Shape{static_cast<std::int64_t>(range.elements)},
                        DType::Float32, Device::hip(communicator.devices()[rank]));
                    add_elements(candidate.capacity_elements, range.elements);
                    auto& unpacked = persistent_range.unpacked[rank];
                    unpacked.reserve(range.end_parameter - range.first_parameter);
                    for (std::size_t parameter = range.first_parameter;
                         parameter < range.end_parameter; ++parameter) {
                        const auto& gradient = rank_parameters[rank][parameter]->grad();
                        unpacked.emplace_back(gradient.shape(), DType::Float32,
                                              gradient.device());
                        add_elements(candidate.capacity_elements,
                                     static_cast<std::size_t>(gradient.numel()));
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
            for (std::size_t rank = 0; rank < communicator.size(); ++rank) {
                std::size_t offset = 0;
                auto& unpacked = range.unpacked[rank];
                stats.unpacked_tensor_count += unpacked.size();
                for (auto& gradient : unpacked) {
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
            for (std::size_t rank = 0; rank < communicator.size(); ++rank) {
                for (std::size_t local = 0;
                     local < range.unpacked[rank].size(); ++local) {
                    rank_parameters[rank][range.first_parameter + local]->set_grad(
                        range.unpacked[rank][local]);
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

}  // namespace microllm::multi_gpu
