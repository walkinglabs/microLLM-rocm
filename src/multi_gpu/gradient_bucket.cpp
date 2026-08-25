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

}  // namespace

BucketStats all_reduce_gradients(
    Communicator& communicator,
    const std::vector<std::vector<autograd::Value*>>& rank_parameters,
    std::size_t maximum_bucket_bytes, bool in_place_average) {
    if (maximum_bucket_bytes < sizeof(float)) {
        throw std::invalid_argument("maximum bucket size must hold at least one float");
    }
    validate(communicator, rank_parameters);
    if (rank_parameters.empty() || rank_parameters.front().empty()) return {};
    const auto maximum_elements = maximum_bucket_bytes / sizeof(float);
    BucketStats stats;
    stats.parameter_count = rank_parameters.front().size();
    std::size_t first_parameter = 0;
    while (first_parameter < stats.parameter_count) {
        std::size_t end_parameter = first_parameter;
        std::size_t bucket_elements = 0;
        while (end_parameter < stats.parameter_count) {
            const auto elements = static_cast<std::size_t>(
                rank_parameters[0][end_parameter]->grad().numel());
            if (end_parameter != first_parameter && bucket_elements + elements > maximum_elements) {
                break;
            }
            bucket_elements += elements;
            ++end_parameter;
            if (bucket_elements >= maximum_elements) break;
        }

        std::vector<Tensor> buckets;
        buckets.reserve(communicator.size());
        for (std::size_t rank = 0; rank < communicator.size(); ++rank) {
            buckets.emplace_back(Shape{static_cast<std::int64_t>(bucket_elements)},
                                 DType::Float32, Device::hip(communicator.devices()[rank]));
            ++stats.bucket_tensor_count;
            stats.temporary_elements += bucket_elements;
            std::size_t offset = 0;
            for (std::size_t index = first_parameter; index < end_parameter; ++index) {
                const auto& gradient = rank_parameters[rank][index]->grad();
                const auto bytes = static_cast<std::size_t>(gradient.numel()) * sizeof(float);
                auto* destination = static_cast<std::byte*>(buckets.back().data()) +
                                    offset * sizeof(float);
                runtime::copy_bytes_async(destination, buckets.back().device(), gradient.data(),
                                          gradient.device(), bytes, communicator.stream(rank));
                ++stats.pack_copy_calls;
                offset += static_cast<std::size_t>(gradient.numel());
            }
        }
        communicator.all_reduce(buckets, true, in_place_average);
        if (!in_place_average) {
            stats.average_tensor_count += communicator.size();
            stats.temporary_elements += bucket_elements * communicator.size();
        }

        std::vector<std::vector<Tensor>> unpacked(communicator.size());
        for (std::size_t rank = 0; rank < communicator.size(); ++rank) {
            std::size_t offset = 0;
            unpacked[rank].reserve(end_parameter - first_parameter);
            for (std::size_t index = first_parameter; index < end_parameter; ++index) {
                const auto shape = rank_parameters[rank][index]->grad().shape();
                unpacked[rank].emplace_back(shape, DType::Float32,
                                            Device::hip(communicator.devices()[rank]));
                ++stats.unpacked_tensor_count;
                stats.temporary_elements += static_cast<std::size_t>(
                    unpacked[rank].back().numel());
                const auto bytes = static_cast<std::size_t>(unpacked[rank].back().numel()) *
                                   sizeof(float);
                const auto* source = static_cast<const std::byte*>(buckets[rank].data()) +
                                     offset * sizeof(float);
                runtime::copy_bytes_async(unpacked[rank].back().data(),
                                          unpacked[rank].back().device(), source,
                                          buckets[rank].device(), bytes,
                                          communicator.stream(rank));
                ++stats.unpack_copy_calls;
                offset += static_cast<std::size_t>(unpacked[rank].back().numel());
            }
        }
        communicator.synchronize();
        for (std::size_t rank = 0; rank < communicator.size(); ++rank) {
            for (std::size_t local = 0; local < unpacked[rank].size(); ++local) {
                rank_parameters[rank][first_parameter + local]->set_grad(
                    std::move(unpacked[rank][local]));
            }
        }
        ++stats.bucket_count;
        stats.total_elements += bucket_elements;
        first_parameter = end_parameter;
    }
    if (stats.temporary_elements >
        std::numeric_limits<std::size_t>::max() / sizeof(float)) {
        throw std::overflow_error("gradient bucket temporary bytes overflow");
    }
    stats.temporary_bytes = stats.temporary_elements * sizeof(float);
    return stats;
}

}  // namespace microllm::multi_gpu
