#include <microllm/runtime/memory.h>
#include <microllm/runtime/runtime.h>

#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>

#if MICROLLM_HAS_HIP
#include "hip_strided_copy.h"
#endif

#if MICROLLM_HAS_HIP
#include <hip/hip_runtime_api.h>
#endif

namespace microllm::runtime {
namespace {

#if MICROLLM_HAS_HIP
void check_hip(hipError_t status, const char* operation) {
    if (status != hipSuccess) {
        throw std::runtime_error(std::string(operation) + ": " + hipGetErrorString(status));
    }
}

hipStream_t as_stream(void* handle) { return reinterpret_cast<hipStream_t>(handle); }
hipEvent_t as_event(void* handle) { return reinterpret_cast<hipEvent_t>(handle); }

void select_copy_device(Device destination, Device source) {
    if (destination.is_hip()) {
        check_hip(hipSetDevice(destination.index()), "hipSetDevice(destination)");
    } else if (source.is_hip()) {
        check_hip(hipSetDevice(source.index()), "hipSetDevice(source)");
    }
}

hipMemcpyKind copy_kind(Device destination, Device source) {
    if (destination.is_cpu() && source.is_cpu()) return hipMemcpyHostToHost;
    if (destination.is_hip() && source.is_cpu()) return hipMemcpyHostToDevice;
    if (destination.is_cpu() && source.is_hip()) return hipMemcpyDeviceToHost;
    return hipMemcpyDeviceToDevice;
}
#endif

void require_same_hip_device(Device destination, Device source) {
    if (destination.is_hip() && source.is_hip() && destination != source) {
        throw std::invalid_argument("cross-GPU copy requires the distributed runtime milestone");
    }
}

}  // namespace

bool hip_compiled() noexcept { return MICROLLM_HAS_HIP != 0; }

int hip_device_count() {
#if MICROLLM_HAS_HIP
    int count = 0;
    const auto status = hipGetDeviceCount(&count);
    if (status == hipErrorNoDevice) return 0;
    check_hip(status, "hipGetDeviceCount");
    return count;
#else
    return 0;
#endif
}

DeviceInfo device_info(Device device) {
    if (device.is_cpu()) {
        return {device, "host CPU", "host", 0, 0, 0};
    }
#if MICROLLM_HAS_HIP
    hipDeviceProp_t properties{};
    check_hip(hipGetDeviceProperties(&properties, device.index()), "hipGetDeviceProperties");
    return {device,
            properties.name,
            properties.gcnArchName,
            properties.totalGlobalMem,
            properties.multiProcessorCount,
            properties.warpSize};
#else
    throw std::runtime_error("microLLM was built without HIP support");
#endif
}

MemoryInfo memory_info(Device device) {
    if (device.is_cpu()) return {};
#if MICROLLM_HAS_HIP
    set_device(device);
    std::size_t free_bytes = 0;
    std::size_t total_bytes = 0;
    check_hip(hipMemGetInfo(&free_bytes, &total_bytes), "hipMemGetInfo");
    return {free_bytes, total_bytes};
#else
    throw std::runtime_error("microLLM was built without HIP support");
#endif
}

int hip_runtime_version() {
#if MICROLLM_HAS_HIP
    int version = 0;
    check_hip(hipRuntimeGetVersion(&version), "hipRuntimeGetVersion");
    return version;
#else
    return 0;
#endif
}

int hip_driver_version() {
#if MICROLLM_HAS_HIP
    int version = 0;
    check_hip(hipDriverGetVersion(&version), "hipDriverGetVersion");
    return version;
#else
    return 0;
#endif
}

void set_device(Device device) {
    if (device.is_cpu()) return;
#if MICROLLM_HAS_HIP
    check_hip(hipSetDevice(device.index()), "hipSetDevice");
#else
    throw std::runtime_error("microLLM was built without HIP support");
#endif
}

void synchronize(Device device) {
    if (device.is_cpu()) return;
#if MICROLLM_HAS_HIP
    set_device(device);
    check_hip(hipDeviceSynchronize(), "hipDeviceSynchronize");
#else
    throw std::runtime_error("microLLM was built without HIP support");
#endif
}

void* allocate(std::size_t num_bytes, Device device) {
    if (num_bytes == 0) return nullptr;
    if (device.is_cpu()) return ::operator new(num_bytes);
#if MICROLLM_HAS_HIP
    set_device(device);
    void* pointer = nullptr;
    check_hip(hipMalloc(&pointer, num_bytes), "hipMalloc");
    return pointer;
#else
    throw std::runtime_error("HIP allocation requested from a CPU-only build");
#endif
}

void deallocate(void* pointer, Device device) noexcept {
    if (pointer == nullptr) return;
    if (device.is_cpu()) {
        ::operator delete(pointer);
        return;
    }
#if MICROLLM_HAS_HIP
    if (hipSetDevice(device.index()) == hipSuccess) (void)hipFree(pointer);
#else
    (void)device;
#endif
}

void copy_bytes(void* destination, Device destination_device, const void* source,
                Device source_device, std::size_t num_bytes) {
    if (num_bytes == 0) return;
    if (destination == nullptr || source == nullptr) {
        throw std::invalid_argument("copy pointers must be non-null for a non-empty copy");
    }
    require_same_hip_device(destination_device, source_device);
    if (destination_device.is_cpu() && source_device.is_cpu()) {
        std::memcpy(destination, source, num_bytes);
        return;
    }
#if MICROLLM_HAS_HIP
    select_copy_device(destination_device, source_device);
    check_hip(hipMemcpy(destination, source, num_bytes,
                        copy_kind(destination_device, source_device)),
              "hipMemcpy");
#else
    throw std::runtime_error("HIP copy requested from a CPU-only build");
#endif
}

void copy_strided_float32(void* contiguous_destination, const void* strided_source,
                          Device device, std::span<const std::int64_t> shape,
                          std::span<const std::int64_t> strides) {
    if (shape.size() != strides.size()) {
        throw std::invalid_argument("strided copy shape/stride rank mismatch");
    }
    std::int64_t elements = 1;
    for (const auto dimension : shape) {
        if (dimension < 0 ||
            (elements != 0 && dimension > std::numeric_limits<std::int64_t>::max() / elements)) {
            throw std::overflow_error("strided copy shape is invalid");
        }
        elements *= dimension;
    }
    if (elements == 0) return;
    if (contiguous_destination == nullptr || strided_source == nullptr) {
        throw std::invalid_argument("strided copy pointers must be non-null");
    }
    if (device.is_cpu()) {
        auto* destination = static_cast<float*>(contiguous_destination);
        const auto* source = static_cast<const float*>(strided_source);
        for (std::int64_t logical = 0; logical < elements; ++logical) {
            auto remainder = logical;
            std::int64_t source_index = 0;
            for (std::size_t reversed = shape.size(); reversed > 0; --reversed) {
                const auto dim = reversed - 1;
                const auto coordinate = remainder % shape[dim];
                remainder /= shape[dim];
                source_index += coordinate * strides[dim];
            }
            destination[logical] = source[source_index];
        }
        return;
    }
#if MICROLLM_HAS_HIP
    set_device(device);
    detail::launch_strided_copy_float32(static_cast<float*>(contiguous_destination),
                                        static_cast<const float*>(strided_source), elements,
                                        static_cast<std::int64_t>(shape.size()), shape.data(),
                                        strides.data());
#else
    throw std::runtime_error("HIP strided copy requested from a CPU-only build");
#endif
}

struct Stream::Impl {
    Device device = Device::cpu();
    void* handle = nullptr;
};

Stream::Stream(Device device, bool non_blocking) : impl_(std::make_unique<Impl>()) {
    impl_->device = device;
    if (device.is_cpu()) return;
#if MICROLLM_HAS_HIP
    set_device(device);
    hipStream_t stream = nullptr;
    check_hip(hipStreamCreateWithFlags(&stream, non_blocking ? hipStreamNonBlocking : hipStreamDefault),
              "hipStreamCreateWithFlags");
    impl_->handle = reinterpret_cast<void*>(stream);
#else
    (void)non_blocking;
    throw std::runtime_error("HIP stream requested from a CPU-only build");
#endif
}

Stream::~Stream() {
#if MICROLLM_HAS_HIP
    if (impl_ && impl_->handle != nullptr) {
        (void)hipSetDevice(impl_->device.index());
        (void)hipStreamDestroy(as_stream(impl_->handle));
    }
#endif
}
Stream::Stream(Stream&&) noexcept = default;
Stream& Stream::operator=(Stream&&) noexcept = default;
Device Stream::device() const noexcept { return impl_->device; }
void* Stream::native_handle() const noexcept { return impl_->handle; }
void Stream::synchronize() const {
    if (impl_->device.is_cpu()) return;
#if MICROLLM_HAS_HIP
    check_hip(hipStreamSynchronize(as_stream(impl_->handle)), "hipStreamSynchronize");
#else
    throw std::runtime_error("microLLM was built without HIP support");
#endif
}

struct Event::Impl {
    Device device = Device::cpu();
    void* handle = nullptr;
    bool timing = true;
    bool cpu_recorded = false;
};

Event::Event(Device device, bool enable_timing) : impl_(std::make_unique<Impl>()) {
    impl_->device = device;
    impl_->timing = enable_timing;
    if (device.is_cpu()) return;
#if MICROLLM_HAS_HIP
    set_device(device);
    hipEvent_t event = nullptr;
    const unsigned flags = enable_timing ? hipEventDefault : hipEventDisableTiming;
    check_hip(hipEventCreateWithFlags(&event, flags), "hipEventCreateWithFlags");
    impl_->handle = reinterpret_cast<void*>(event);
#else
    throw std::runtime_error("HIP event requested from a CPU-only build");
#endif
}

Event::~Event() {
#if MICROLLM_HAS_HIP
    if (impl_ && impl_->handle != nullptr) {
        (void)hipSetDevice(impl_->device.index());
        (void)hipEventDestroy(as_event(impl_->handle));
    }
#endif
}
Event::Event(Event&&) noexcept = default;
Event& Event::operator=(Event&&) noexcept = default;
Device Event::device() const noexcept { return impl_->device; }
void* Event::native_handle() const noexcept { return impl_->handle; }

void Event::record(const Stream& stream) {
    if (stream.device() != impl_->device) throw std::invalid_argument("event/stream device mismatch");
    if (impl_->device.is_cpu()) {
        impl_->cpu_recorded = true;
        return;
    }
#if MICROLLM_HAS_HIP
    check_hip(hipEventRecord(as_event(impl_->handle), as_stream(stream.native_handle())),
              "hipEventRecord");
#endif
}

void Event::wait(const Stream& stream) const {
    if (stream.device() != impl_->device) throw std::invalid_argument("event/stream device mismatch");
    if (impl_->device.is_cpu()) return;
#if MICROLLM_HAS_HIP
    check_hip(hipStreamWaitEvent(as_stream(stream.native_handle()), as_event(impl_->handle), 0),
              "hipStreamWaitEvent");
#endif
}

void Event::synchronize() const {
    if (impl_->device.is_cpu()) return;
#if MICROLLM_HAS_HIP
    check_hip(hipEventSynchronize(as_event(impl_->handle)), "hipEventSynchronize");
#endif
}

bool Event::ready() const {
    if (impl_->device.is_cpu()) return impl_->cpu_recorded;
#if MICROLLM_HAS_HIP
    const auto status = hipEventQuery(as_event(impl_->handle));
    if (status == hipSuccess) return true;
    if (status == hipErrorNotReady) return false;
    check_hip(status, "hipEventQuery");
#endif
    return false;
}

float Event::elapsed_ms_since(const Event& start) const {
    if (impl_->device != start.impl_->device) {
        throw std::invalid_argument("elapsed events must use the same device");
    }
    if (!impl_->timing || !start.impl_->timing) {
        throw std::invalid_argument("elapsed time requires timing-enabled events");
    }
    if (impl_->device.is_cpu()) {
        throw std::runtime_error("CPU events do not provide device elapsed time");
    }
#if MICROLLM_HAS_HIP
    float milliseconds = 0.0F;
    check_hip(hipEventElapsedTime(&milliseconds, as_event(start.impl_->handle),
                                  as_event(impl_->handle)),
              "hipEventElapsedTime");
    return milliseconds;
#else
    throw std::runtime_error("microLLM was built without HIP support");
#endif
}

void copy_bytes_async(void* destination, Device destination_device, const void* source,
                      Device source_device, std::size_t num_bytes, const Stream& stream) {
    if (num_bytes == 0) return;
    if (destination == nullptr || source == nullptr) {
        throw std::invalid_argument("copy pointers must be non-null for a non-empty copy");
    }
    require_same_hip_device(destination_device, source_device);
    if (destination_device.is_cpu() && source_device.is_cpu()) {
        if (!stream.device().is_cpu()) throw std::invalid_argument("CPU copy requires a CPU stream");
        std::memcpy(destination, source, num_bytes);
        return;
    }
    const auto copy_device = destination_device.is_hip() ? destination_device : source_device;
    if (stream.device() != copy_device) throw std::invalid_argument("copy/stream device mismatch");
#if MICROLLM_HAS_HIP
    select_copy_device(destination_device, source_device);
    check_hip(hipMemcpyAsync(destination, source, num_bytes,
                             copy_kind(destination_device, source_device),
                             as_stream(stream.native_handle())),
              "hipMemcpyAsync");
#else
    throw std::runtime_error("HIP copy requested from a CPU-only build");
#endif
}

}  // namespace microllm::runtime
