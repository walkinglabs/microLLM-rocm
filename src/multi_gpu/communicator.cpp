#include <microllm/multi_gpu/communicator.h>

#include <stdexcept>
#include <string>
#include <utility>

#include <rccl/rccl.h>

#include <microllm/ops/ops.h>
#include <microllm/runtime/runtime.h>

namespace microllm::multi_gpu {
namespace {

void check_rccl(ncclResult_t result, const char* operation) {
    if (result != ncclSuccess) {
        throw std::runtime_error(std::string(operation) + ": " + ncclGetErrorString(result));
    }
}

}  // namespace

struct Communicator::Impl {
    std::vector<int> devices;
    std::vector<ncclComm_t> communicators;
    std::vector<runtime::Stream> streams;
    bool aborted = false;
};

Communicator::Communicator(std::vector<int> device_indices)
    : impl_(std::make_unique<Impl>()) {
    if (device_indices.empty()) throw std::invalid_argument("communicator needs at least one device");
    const auto visible = runtime::hip_device_count();
    std::vector<bool> seen(static_cast<std::size_t>(visible), false);
    for (const auto device : device_indices) {
        if (device < 0 || device >= visible) throw std::out_of_range("communicator device is not visible");
        if (seen[static_cast<std::size_t>(device)]) {
            throw std::invalid_argument("communicator devices must be unique");
        }
        seen[static_cast<std::size_t>(device)] = true;
    }
    impl_->devices = std::move(device_indices);
    impl_->communicators.resize(impl_->devices.size(), nullptr);
    const auto initialization = ncclCommInitAll(impl_->communicators.data(),
                                                static_cast<int>(impl_->devices.size()),
                                                impl_->devices.data());
    if (initialization != ncclSuccess) {
        for (const auto communicator : impl_->communicators) {
            if (communicator != nullptr) (void)ncclCommAbort(communicator);
        }
        check_rccl(initialization, "ncclCommInitAll");
    }
    impl_->streams.reserve(impl_->devices.size());
    for (const auto device : impl_->devices) {
        impl_->streams.emplace_back(Device::hip(device));
    }
}

Communicator::~Communicator() {
    if (!impl_) return;
    if (!impl_->aborted) {
        for (const auto communicator : impl_->communicators) {
            if (communicator != nullptr) (void)ncclCommDestroy(communicator);
        }
    }
}
Communicator::Communicator(Communicator&&) noexcept = default;
Communicator& Communicator::operator=(Communicator&&) noexcept = default;
std::size_t Communicator::size() const noexcept { return impl_->devices.size(); }
const std::vector<int>& Communicator::devices() const noexcept { return impl_->devices; }
bool Communicator::aborted() const noexcept { return impl_->aborted; }
runtime::Stream& Communicator::stream(std::size_t rank) { return impl_->streams.at(rank); }

void Communicator::all_reduce(std::vector<Tensor>& tensors, bool average,
                              bool in_place_average) {
    if (tensors.empty()) return;
    enqueue_all_reduce_sum(tensors);
    synchronize();
    if (average) {
        const auto factor = 1.0F / static_cast<float>(tensors.size());
        for (std::size_t rank = 0; rank < tensors.size(); ++rank) {
            const ops::OpContext context{&impl_->streams[rank], nullptr, 0};
            if (in_place_average) {
                ops::scale_in_place_(tensors[rank], factor, context);
            } else {
                tensors[rank] = ops::scale(tensors[rank], factor, context);
            }
        }
        synchronize();
    }
}

void Communicator::enqueue_all_reduce_sum(std::vector<Tensor>& tensors) {
    if (impl_->aborted) throw std::logic_error("communicator has been aborted");
    if (tensors.size() != impl_->devices.size()) {
        throw std::invalid_argument("all-reduce needs one Tensor per rank");
    }
    if (tensors.empty()) return;
    const auto shape = tensors.front().shape();
    const auto elements = tensors.front().numel();
    for (std::size_t rank = 0; rank < tensors.size(); ++rank) {
        const auto& tensor = tensors[rank];
        if (tensor.device() != Device::hip(impl_->devices[rank]) ||
            tensor.dtype() != DType::Float32 || !tensor.is_contiguous() ||
            tensor.shape() != shape) {
            throw std::invalid_argument(
                "all-reduce tensors must be matching contiguous float32 rank-local HIP tensors");
        }
    }
    bool group_started = false;
    try {
        check_rccl(ncclGroupStart(), "ncclGroupStart");
        group_started = true;
        for (std::size_t rank = 0; rank < tensors.size(); ++rank) {
            check_rccl(ncclAllReduce(tensors[rank].data(), tensors[rank].data(),
                                     static_cast<std::size_t>(elements), ncclFloat32, ncclSum,
                                     impl_->communicators[rank],
                                     reinterpret_cast<hipStream_t>(
                                         impl_->streams[rank].native_handle())),
                       "ncclAllReduce");
        }
        check_rccl(ncclGroupEnd(), "ncclGroupEnd");
        group_started = false;
    } catch (...) {
        if (group_started) (void)ncclGroupEnd();
        abort();
        throw;
    }
}

void Communicator::enqueue_all_reduce_average_in_place(
    std::vector<Tensor>& tensors) {
    enqueue_all_reduce_sum(tensors);
    const auto factor = 1.0F / static_cast<float>(tensors.size());
    for (std::size_t rank = 0; rank < tensors.size(); ++rank) {
        const ops::OpContext context{&impl_->streams[rank], nullptr, 0};
        ops::scale_in_place_(tensors[rank], factor, context);
    }
}

void Communicator::synchronize() {
    if (impl_->aborted) throw std::logic_error("communicator has been aborted");
    try {
        for (const auto& stream : impl_->streams) stream.synchronize();
    } catch (...) {
        abort();
        throw;
    }
}

void Communicator::abort() noexcept {
    if (!impl_ || impl_->aborted) return;
    for (const auto communicator : impl_->communicators) {
        if (communicator != nullptr) (void)ncclCommAbort(communicator);
    }
    impl_->aborted = true;
}

}  // namespace microllm::multi_gpu
