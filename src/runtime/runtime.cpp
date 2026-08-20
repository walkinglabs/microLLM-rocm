#include <microllm/runtime/memory.h>
#include <microllm/runtime/runtime.h>

#include <cstring>
#include <algorithm>
#include <atomic>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#if MICROLLM_HAS_HIP
#include "hip_strided_copy.h"
#endif

#if MICROLLM_HAS_HIP
#include <hip/hip_runtime_api.h>
#endif

namespace microllm::runtime {
namespace {

struct AllocationCounters {
    std::atomic<std::size_t> current{0};
    std::atomic<std::size_t> peak{0};
    std::atomic<std::size_t> total{0};
    std::atomic<std::size_t> allocation_calls{0};
    std::atomic<std::size_t> deallocation_calls{0};
    std::atomic<std::size_t> backend_allocation_calls{0};
    std::atomic<std::size_t> backend_deallocation_calls{0};
    std::atomic<std::size_t> cache_reuse_calls{0};
    std::atomic<std::size_t> cached_bytes{0};
    std::atomic<std::size_t> reserved_bytes{0};
};

AllocationCounters cpu_counters;
AllocationCounters hip_counters;

struct TransferCounters {
    std::atomic<std::size_t> host_to_device_calls{0};
    std::atomic<std::size_t> device_to_host_calls{0};
    std::atomic<std::size_t> device_to_device_calls{0};
    std::atomic<std::size_t> host_to_device_bytes{0};
    std::atomic<std::size_t> device_to_host_bytes{0};
    std::atomic<std::size_t> device_to_device_bytes{0};
};

TransferCounters transfer_counters;

void record_transfer(Device destination, Device source, std::size_t bytes) {
    if (destination.is_cpu() && source.is_cpu()) return;
    if (destination.is_hip() && source.is_cpu()) {
        transfer_counters.host_to_device_calls.fetch_add(1);
        transfer_counters.host_to_device_bytes.fetch_add(bytes);
    } else if (destination.is_cpu() && source.is_hip()) {
        transfer_counters.device_to_host_calls.fetch_add(1);
        transfer_counters.device_to_host_bytes.fetch_add(bytes);
    } else {
        transfer_counters.device_to_device_calls.fetch_add(1);
        transfer_counters.device_to_device_bytes.fetch_add(bytes);
    }
}

AllocationCounters& counters(Device device) {
    return device.is_cpu() ? cpu_counters : hip_counters;
}

void record_allocation(Device device, std::size_t bytes) {
    auto& values = counters(device);
    const auto current = values.current.fetch_add(bytes) + bytes;
    values.total.fetch_add(bytes);
    values.allocation_calls.fetch_add(1);
    auto peak = values.peak.load();
    while (current > peak && !values.peak.compare_exchange_weak(peak, current)) {
    }
}

#if MICROLLM_HAS_HIP
struct SharedReadyEvent {
    hipEvent_t value = nullptr;
    ~SharedReadyEvent() { if (value != nullptr) (void)hipEventDestroy(value); }
};

struct RetiredBlock {
    void* pointer = nullptr;
    std::shared_ptr<SharedReadyEvent> ready;
};

struct PendingBlock {
    void* pointer = nullptr;
    std::size_t bytes = 0;
};

struct HipExactSizePool {
    std::mutex mutex;
    std::map<int, bool> enabled;
    std::map<int, bool> forbidden;
    std::map<std::pair<int, std::size_t>, std::vector<RetiredBlock>> retired;
    std::map<int, std::vector<PendingBlock>> pending;
};

HipExactSizePool& hip_pool() {
    // Deliberately process-lifetime: HIP may already be shutting down when C++ static
    // destructors run. The driver reclaims retained blocks at process exit.
    static auto* pool = new HipExactSizePool();
    return *pool;
}

constexpr std::size_t kMaximumCachedBytesPerDevice = 8ULL * 1024ULL * 1024ULL * 1024ULL;
constexpr std::size_t kRetirementBatchSize = 8;

void check_hip(hipError_t status, const char* operation) {
    if (status != hipSuccess) {
        throw std::runtime_error(std::string(operation) + ": " + hipGetErrorString(status));
    }
}

hipStream_t as_stream(void* handle) { return reinterpret_cast<hipStream_t>(handle); }
hipEvent_t as_event(void* handle) { return reinterpret_cast<hipEvent_t>(handle); }

void flush_pending(HipExactSizePool& pool, Device device) noexcept {
    auto& pending = pool.pending[device.index()];
    if (pending.empty()) return;
    hipEvent_t event = nullptr;
    const auto created = hipEventCreateWithFlags(&event, hipEventDisableTiming);
    const auto recorded = created == hipSuccess ? hipEventRecord(event, nullptr) : created;
    if (created == hipSuccess && recorded == hipSuccess) {
        auto ready = std::make_shared<SharedReadyEvent>();
        ready->value = event;
        for (const auto& block : pending) {
            pool.retired[{device.index(), block.bytes}].push_back({block.pointer, ready});
        }
        pending.clear();
        return;
    }
    if (event != nullptr) (void)hipEventDestroy(event);
    for (const auto& block : pending) {
        if (hipFree(block.pointer) == hipSuccess) {
            counters(device).backend_deallocation_calls.fetch_add(1);
            counters(device).cached_bytes.fetch_sub(block.bytes);
            counters(device).reserved_bytes.fetch_sub(block.bytes);
        }
    }
    pending.clear();
}

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

PrecisionCapabilities precision_capabilities(Device device) {
    const auto info = device_info(device);
    PrecisionCapabilities capabilities;
    capabilities.architecture = info.architecture;
    capabilities.fp32 = true;
    if (device.is_cpu()) return capabilities;
    const auto is_cdna3 = info.architecture.rfind("gfx942", 0) == 0;
    const auto is_cdna4 = info.architecture.rfind("gfx950", 0) == 0;
    capabilities.fp64 = is_cdna3 || is_cdna4;
    capabilities.tf32_hardware = is_cdna3;
    capabilities.fp16 = is_cdna3 || is_cdna4;
    capabilities.bfloat16 = is_cdna3 || is_cdna4;
    capabilities.fp8_fnuz = is_cdna3;
    capabilities.fp8_ocp = is_cdna4;
    capabilities.int8_matrix = is_cdna3 || is_cdna4;
    capabilities.mxfp8 = is_cdna4;
    capabilities.mxfp6 = is_cdna4;
    capabilities.mxfp4 = is_cdna4;
    // The published MI300/MI350 precision table lists INT8 Matrix but not INT4 Matrix.
    // Packed INT4 remains a software dequantization path until a probed backend says otherwise.
    capabilities.int4_matrix = false;
    return capabilities;
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
    {
        auto& pool = hip_pool();
        const std::lock_guard<std::mutex> lock(pool.mutex);
        flush_pending(pool, device);
    }
    check_hip(hipDeviceSynchronize(), "hipDeviceSynchronize");
#else
    throw std::runtime_error("microLLM was built without HIP support");
#endif
}

TransferStats transfer_stats() noexcept {
    return {transfer_counters.host_to_device_calls.load(),
            transfer_counters.device_to_host_calls.load(),
            transfer_counters.device_to_device_calls.load(),
            transfer_counters.host_to_device_bytes.load(),
            transfer_counters.device_to_host_bytes.load(),
            transfer_counters.device_to_device_bytes.load()};
}

void reset_transfer_stats() noexcept {
    transfer_counters.host_to_device_calls.store(0);
    transfer_counters.device_to_host_calls.store(0);
    transfer_counters.device_to_device_calls.store(0);
    transfer_counters.host_to_device_bytes.store(0);
    transfer_counters.device_to_host_bytes.store(0);
    transfer_counters.device_to_device_bytes.store(0);
}

void notify_non_default_stream(Device device) noexcept {
    if (device.is_cpu()) return;
#if MICROLLM_HAS_HIP
    if (hipSetDevice(device.index()) != hipSuccess) return;
    auto& pool = hip_pool();
    const std::lock_guard<std::mutex> lock(pool.mutex);
    flush_pending(pool, device);
    pool.enabled[device.index()] = false;
    pool.forbidden[device.index()] = true;
#else
    (void)device;
#endif
}

void enable_hip_caching_allocator(Device device) {
    if (device.is_cpu()) return;
#if MICROLLM_HAS_HIP
    set_device(device);
    auto& pool = hip_pool();
    const std::lock_guard<std::mutex> lock(pool.mutex);
    if (pool.forbidden[device.index()]) {
        throw std::logic_error(
            "HIP caching allocator cannot be enabled after non-default Stream use");
    }
    pool.enabled[device.index()] = true;
#else
    (void)device;
    throw std::runtime_error("HIP caching allocator requested from a CPU-only build");
#endif
}

bool hip_caching_allocator_enabled(Device device) noexcept {
    if (device.is_cpu()) return false;
#if MICROLLM_HAS_HIP
    auto& pool = hip_pool();
    const std::lock_guard<std::mutex> lock(pool.mutex);
    return pool.enabled[device.index()];
#else
    (void)device;
    return false;
#endif
}

void* allocate(std::size_t num_bytes, Device device) {
    if (num_bytes == 0) return nullptr;
    if (device.is_cpu()) {
        auto* pointer = ::operator new(num_bytes);
        counters(device).backend_allocation_calls.fetch_add(1);
        counters(device).reserved_bytes.fetch_add(num_bytes);
        record_allocation(device, num_bytes);
        return pointer;
    }
#if MICROLLM_HAS_HIP
    set_device(device);
    {
        auto& pool = hip_pool();
        const std::lock_guard<std::mutex> lock(pool.mutex);
        if (pool.enabled[device.index()]) {
            auto& candidates = pool.retired[{device.index(), num_bytes}];
            for (auto candidate = candidates.begin(); candidate != candidates.end(); ++candidate) {
                const auto status = hipEventQuery(candidate->ready->value);
                if (status == hipSuccess) {
                    void* pointer = candidate->pointer;
                    candidates.erase(candidate);
                    counters(device).cached_bytes.fetch_sub(num_bytes);
                    counters(device).cache_reuse_calls.fetch_add(1);
                    record_allocation(device, num_bytes);
                    return pointer;
                }
                if (status != hipErrorNotReady) (void)hipGetLastError();
            }
        }
    }
    void* pointer = nullptr;
    check_hip(hipMalloc(&pointer, num_bytes), "hipMalloc");
    counters(device).backend_allocation_calls.fetch_add(1);
    counters(device).reserved_bytes.fetch_add(num_bytes);
    record_allocation(device, num_bytes);
    return pointer;
#else
    throw std::runtime_error("HIP allocation requested from a CPU-only build");
#endif
}

void deallocate(void* pointer, Device device, std::size_t num_bytes) noexcept {
    if (pointer == nullptr) return;
    if (device.is_cpu()) {
        ::operator delete(pointer);
        counters(device).current.fetch_sub(num_bytes);
        counters(device).deallocation_calls.fetch_add(1);
        counters(device).backend_deallocation_calls.fetch_add(1);
        counters(device).reserved_bytes.fetch_sub(num_bytes);
        return;
    }
#if MICROLLM_HAS_HIP
    if (hipSetDevice(device.index()) != hipSuccess) return;
    bool cached = false;
    {
        auto& pool = hip_pool();
        const std::lock_guard<std::mutex> lock(pool.mutex);
        const auto cached_bytes = counters(device).cached_bytes.load();
        if (pool.enabled[device.index()] &&
            num_bytes <= kMaximumCachedBytesPerDevice -
                             std::min(cached_bytes, kMaximumCachedBytesPerDevice)) {
            pool.pending[device.index()].push_back({pointer, num_bytes});
            counters(device).cached_bytes.fetch_add(num_bytes);
            cached = true;
            if (pool.pending[device.index()].size() >= kRetirementBatchSize) {
                flush_pending(pool, device);
            }
        }
    }
    if (cached) {
        counters(device).current.fetch_sub(num_bytes);
        counters(device).deallocation_calls.fetch_add(1);
        return;
    }
    if (hipFree(pointer) == hipSuccess) {
        counters(device).current.fetch_sub(num_bytes);
        counters(device).deallocation_calls.fetch_add(1);
        counters(device).backend_deallocation_calls.fetch_add(1);
        counters(device).reserved_bytes.fetch_sub(num_bytes);
    }
#else
    (void)device;
#endif
}

AllocationStats allocation_stats(Device device) noexcept {
    auto& values = counters(device);
    return {values.current.load(), values.peak.load(), values.total.load(),
            values.allocation_calls.load(), values.deallocation_calls.load(),
            values.backend_allocation_calls.load(), values.backend_deallocation_calls.load(),
            values.cache_reuse_calls.load(), values.cached_bytes.load(),
            values.reserved_bytes.load()};
}

void reset_allocation_peak(Device device) noexcept {
    auto& values = counters(device);
    values.peak.store(values.current.load());
    values.total.store(0);
    values.allocation_calls.store(0);
    values.deallocation_calls.store(0);
    values.backend_allocation_calls.store(0);
    values.backend_deallocation_calls.store(0);
    values.cache_reuse_calls.store(0);
}

void copy_bytes(void* destination, Device destination_device, const void* source,
                Device source_device, std::size_t num_bytes) {
    if (num_bytes == 0) return;
    if (destination == nullptr || source == nullptr) {
        throw std::invalid_argument("copy pointers must be non-null for a non-empty copy");
    }
    require_same_hip_device(destination_device, source_device);
    record_transfer(destination_device, source_device, num_bytes);
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

void copy_strided(void* contiguous_destination, const void* strided_source,
                  std::size_t element_bytes, Device device,
                  std::span<const std::int64_t> shape,
                  std::span<const std::int64_t> strides) {
    if (shape.size() != strides.size()) {
        throw std::invalid_argument("strided copy shape/stride rank mismatch");
    }
    if (element_bytes == 0) throw std::invalid_argument("strided copy element size is zero");
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
        auto* destination = static_cast<std::byte*>(contiguous_destination);
        const auto* source = static_cast<const std::byte*>(strided_source);
        for (std::int64_t logical = 0; logical < elements; ++logical) {
            auto remainder = logical;
            std::int64_t source_index = 0;
            for (std::size_t reversed = shape.size(); reversed > 0; --reversed) {
                const auto dim = reversed - 1;
                const auto coordinate = remainder % shape[dim];
                remainder /= shape[dim];
                source_index += coordinate * strides[dim];
            }
            std::memcpy(destination + static_cast<std::size_t>(logical) * element_bytes,
                        source + static_cast<std::size_t>(source_index) * element_bytes,
                        element_bytes);
        }
        return;
    }
#if MICROLLM_HAS_HIP
    set_device(device);
    detail::launch_strided_copy(contiguous_destination, strided_source, element_bytes,
                                elements, static_cast<std::int64_t>(shape.size()),
                                shape.data(), strides.data());
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
    notify_non_default_stream(device);
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
    record_transfer(destination_device, source_device, num_bytes);
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
