#include <microllm/runtime/memory.h>
#include <microllm/runtime/diagnostics.h>
#include <microllm/runtime/runtime.h>

#include <cstring>
#include <algorithm>
#include <array>
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

thread_local bool strided_copy_diagnostics_enabled = false;
thread_local std::vector<StridedCopyRecord> strided_copy_diagnostic_records;
thread_local bool allocation_source_diagnostics_enabled = false;
thread_local AllocationSource active_allocation_source =
    AllocationSource::Unspecified;
thread_local std::vector<AllocationSourceRecord>
    allocation_source_diagnostic_records;
thread_local DeferredHipDeallocationScope* active_deferred_scope = nullptr;
thread_local ScopedDeferredHipStream* active_scoped_deferred_stream = nullptr;

void record_strided_copy(std::size_t element_bytes, Device device,
                         std::span<const std::int64_t> shape,
                         std::span<const std::int64_t> strides,
                         std::int64_t elements) {
    if (!strided_copy_diagnostics_enabled) return;
    const auto found = std::find_if(
        strided_copy_diagnostic_records.begin(),
        strided_copy_diagnostic_records.end(), [&](const auto& record) {
            return record.source == active_allocation_source &&
                   record.element_bytes == element_bytes && record.device == device &&
                   std::equal(record.shape.begin(), record.shape.end(),
                              shape.begin(), shape.end()) &&
                   std::equal(record.strides.begin(), record.strides.end(),
                              strides.begin(), strides.end());
        });
    const auto new_record = found == strided_copy_diagnostic_records.end();
    auto* record = new_record ? &strided_copy_diagnostic_records.emplace_back()
                              : &*found;
    if (new_record) {
        record->source = active_allocation_source;
        record->shape.assign(shape.begin(), shape.end());
        record->strides.assign(strides.begin(), strides.end());
        record->element_bytes = element_bytes;
        record->device = device;
    }
    ++record->calls;
    record->elements += static_cast<std::uint64_t>(elements);
    record->bytes += static_cast<std::uint64_t>(elements) * element_bytes;
}

void record_allocation_source(Device device, std::size_t bytes) {
    if (!allocation_source_diagnostics_enabled) return;
    const auto found = std::find_if(
        allocation_source_diagnostic_records.begin(),
        allocation_source_diagnostic_records.end(),
        [&](const auto& record) {
            return record.source == active_allocation_source &&
                   record.device == device &&
                   record.allocation_bytes == bytes;
        });
    const auto is_new = found == allocation_source_diagnostic_records.end();
    auto* record = is_new
                       ? &allocation_source_diagnostic_records.emplace_back()
                       : &*found;
    if (is_new) {
        record->source = active_allocation_source;
        record->device = device;
        record->allocation_bytes = bytes;
    }
    ++record->calls;
    record->total_bytes += static_cast<std::uint64_t>(bytes);
}

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
struct HipExactSizePool {
    std::mutex mutex;
    std::map<int, bool> enabled;
    std::map<int, bool> forbidden;
    // Immediate address reuse is safe only while every engine operation uses
    // the legacy default stream: later work is ordered after the last use of
    // the retired address.  notify_non_default_stream permanently disables it.
    std::map<std::pair<int, std::size_t>, std::vector<void*>> retired;
};

HipExactSizePool& hip_pool() {
    // Deliberately process-lifetime: HIP may already be shutting down when C++ static
    // destructors run. The driver reclaims retained blocks at process exit.
    static auto* pool = new HipExactSizePool();
    return *pool;
}

constexpr std::size_t kMaximumCachedBytesPerDevice = 8ULL * 1024ULL * 1024ULL * 1024ULL;

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

bool stream_ordered_allocator_supported(Device device) {
    if (device.is_cpu()) return false;
#if MICROLLM_HAS_HIP
    int supported = 0;
    check_hip(hipDeviceGetAttribute(
                  &supported, hipDeviceAttributeMemoryPoolsSupported,
                  device.index()),
              "hipDeviceGetAttribute(memory pools)");
    return supported != 0;
#else
    (void)device;
    return false;
#endif
}

HipMemoryPoolStats default_hip_memory_pool_stats(Device device) {
    if (device.is_cpu()) return {};
#if MICROLLM_HAS_HIP
    HipMemoryPoolStats result;
    result.supported = stream_ordered_allocator_supported(device);
    if (!result.supported) return result;
    set_device(device);
    hipMemPool_t pool = nullptr;
    check_hip(hipDeviceGetDefaultMemPool(&pool, device.index()),
              "hipDeviceGetDefaultMemPool");
    check_hip(hipMemPoolGetAttribute(
                  pool, hipMemPoolAttrReservedMemCurrent,
                  &result.reserved_current_bytes),
              "hipMemPoolGetAttribute(reserved current)");
    check_hip(hipMemPoolGetAttribute(
                  pool, hipMemPoolAttrReservedMemHigh,
                  &result.reserved_high_bytes),
              "hipMemPoolGetAttribute(reserved high)");
    check_hip(hipMemPoolGetAttribute(
                  pool, hipMemPoolAttrUsedMemCurrent,
                  &result.used_current_bytes),
              "hipMemPoolGetAttribute(used current)");
    check_hip(hipMemPoolGetAttribute(
                  pool, hipMemPoolAttrUsedMemHigh,
                  &result.used_high_bytes),
              "hipMemPoolGetAttribute(used high)");
    check_hip(hipMemPoolGetAttribute(
                  pool, hipMemPoolAttrReleaseThreshold,
                  &result.release_threshold_bytes),
              "hipMemPoolGetAttribute(release threshold)");
    return result;
#else
    (void)device;
    throw std::runtime_error("HIP memory pool queried from a CPU-only build");
#endif
}

void set_default_hip_memory_pool_release_threshold(
    Device device, std::uint64_t bytes) {
    if (device.is_cpu()) {
        throw std::invalid_argument("HIP memory pool requires a HIP device");
    }
#if MICROLLM_HAS_HIP
    if (!stream_ordered_allocator_supported(device)) {
        throw std::runtime_error("HIP Stream ordered allocator is unsupported");
    }
    set_device(device);
    hipMemPool_t pool = nullptr;
    check_hip(hipDeviceGetDefaultMemPool(&pool, device.index()),
              "hipDeviceGetDefaultMemPool");
    check_hip(hipMemPoolSetAttribute(
                  pool, hipMemPoolAttrReleaseThreshold, &bytes),
              "hipMemPoolSetAttribute(release threshold)");
#else
    (void)bytes;
    throw std::runtime_error("HIP memory pool configured from a CPU-only build");
#endif
}

void trim_default_hip_memory_pool(Device device,
                                  std::size_t minimum_bytes_to_hold) {
    if (device.is_cpu()) {
        throw std::invalid_argument("HIP memory pool requires a HIP device");
    }
#if MICROLLM_HAS_HIP
    if (!stream_ordered_allocator_supported(device)) {
        throw std::runtime_error("HIP Stream ordered allocator is unsupported");
    }
    set_device(device);
    hipMemPool_t pool = nullptr;
    check_hip(hipDeviceGetDefaultMemPool(&pool, device.index()),
              "hipDeviceGetDefaultMemPool");
    check_hip(hipMemPoolTrimTo(pool, minimum_bytes_to_hold),
              "hipMemPoolTrimTo");
#else
    (void)minimum_bytes_to_hold;
    throw std::runtime_error("HIP memory pool trimmed from a CPU-only build");
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

void* resolve_deferred_hip_stream(
    Device device, void* explicitly_requested_stream) {
    if (device.is_cpu() || active_scoped_deferred_stream == nullptr) {
        return explicitly_requested_stream;
    }
    if (active_scoped_deferred_stream->device() != device) {
        throw std::invalid_argument(
            "active deferred HIP Stream uses a different tensor device");
    }
    const auto active_handle =
        active_scoped_deferred_stream->stream().native_handle();
    if (explicitly_requested_stream != nullptr &&
        explicitly_requested_stream != active_handle) {
        throw std::logic_error(
            "operator Stream conflicts with active deferred HIP Stream");
    }
    return active_handle;
}

void* allocate(std::size_t num_bytes, Device device) {
    if (num_bytes == 0) return nullptr;
    record_allocation_source(device, num_bytes);
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
            if (!candidates.empty()) {
                void* pointer = candidates.back();
                candidates.pop_back();
                counters(device).cached_bytes.fetch_sub(num_bytes);
                counters(device).cache_reuse_calls.fetch_add(1);
                record_allocation(device, num_bytes);
                return pointer;
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
    if (active_deferred_scope != nullptr &&
        active_deferred_scope->device() == device) {
        active_deferred_scope->defer(pointer, num_bytes);
        counters(device).current.fetch_sub(num_bytes);
        counters(device).deallocation_calls.fetch_add(1);
        return;
    }
    if (hipSetDevice(device.index()) != hipSuccess) return;
    bool cached = false;
    {
        auto& pool = hip_pool();
        const std::lock_guard<std::mutex> lock(pool.mutex);
        const auto cached_bytes = counters(device).cached_bytes.load();
        if (pool.enabled[device.index()] &&
            num_bytes <= kMaximumCachedBytesPerDevice -
                             std::min(cached_bytes, kMaximumCachedBytesPerDevice)) {
            // No Event batch is needed under the pool's default-stream-only
            // contract.  Delaying this block until a batch boundary makes
            // reuse depend on unrelated allocation counts.
            pool.retired[{device.index(), num_bytes}].push_back(pointer);
            counters(device).cached_bytes.fetch_add(num_bytes);
            cached = true;
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

void enable_strided_copy_diagnostics(bool enabled) noexcept {
    strided_copy_diagnostics_enabled = enabled;
}

const char* allocation_source_name(AllocationSource source) noexcept {
    switch (source) {
        case AllocationSource::Unspecified: return "unspecified";
        case AllocationSource::ModelEmbedding: return "model.embedding";
        case AllocationSource::AttentionNorm: return "attention.norm";
        case AllocationSource::AttentionProjection: return "attention.projection";
        case AllocationSource::AttentionLayout: return "attention.layout";
        case AllocationSource::AttentionCore: return "attention.core";
        case AllocationSource::AttentionOutput: return "attention.output";
        case AllocationSource::AttentionResidual: return "attention.residual";
        case AllocationSource::FfnNorm: return "ffn.norm";
        case AllocationSource::Ffn: return "ffn";
        case AllocationSource::FfnResidual: return "ffn.residual";
        case AllocationSource::ModelFinalNorm: return "model.final_norm";
        case AllocationSource::ModelOutput: return "model.output";
    }
    return "unknown";
}

ScopedAllocationSource::ScopedAllocationSource(
    AllocationSource source) noexcept {
    if (!allocation_source_diagnostics_enabled &&
        !strided_copy_diagnostics_enabled) {
        return;
    }
    previous_ = active_allocation_source;
    active_ = true;
    active_allocation_source = source;
}

ScopedAllocationSource::~ScopedAllocationSource() {
    if (active_) active_allocation_source = previous_;
}

void enable_allocation_source_diagnostics(bool enabled) noexcept {
    allocation_source_diagnostics_enabled = enabled;
}

void reset_allocation_source_diagnostics() noexcept {
    allocation_source_diagnostic_records.clear();
}

AllocationSourceDiagnostics allocation_source_diagnostics() {
    AllocationSourceDiagnostics result;
    result.records = allocation_source_diagnostic_records;
    for (const auto& record : result.records) {
        result.calls += record.calls;
        result.bytes += record.total_bytes;
    }
    return result;
}

void reset_strided_copy_diagnostics() noexcept {
    strided_copy_diagnostic_records.clear();
}

StridedCopyDiagnostics strided_copy_diagnostics() {
    StridedCopyDiagnostics result;
    result.records = strided_copy_diagnostic_records;
    for (const auto& record : result.records) {
        result.calls += record.calls;
        result.elements += record.elements;
        result.bytes += record.bytes;
    }
    return result;
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
    record_strided_copy(element_bytes, device, shape, strides, elements);
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
                                shape.data(), strides.data(),
                                resolve_deferred_hip_stream(device));
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

struct StreamOrderedHipBuffer::Impl {
    const Stream* stream = nullptr;
    Device device = Device::cpu();
    void* pointer = nullptr;
    std::size_t num_bytes = 0;
};

StreamOrderedHipBuffer::StreamOrderedHipBuffer(
    const Stream& stream, std::size_t bytes)
    : impl_(std::make_unique<Impl>()) {
    if (stream.device().is_cpu()) {
        throw std::invalid_argument("Stream ordered allocation requires a HIP Stream");
    }
    if (bytes == 0) {
        throw std::invalid_argument("Stream ordered allocation size must be positive");
    }
#if MICROLLM_HAS_HIP
    set_device(stream.device());
    void* pointer = nullptr;
    check_hip(hipMallocAsync(&pointer, bytes, as_stream(stream.native_handle())),
              "hipMallocAsync");
    impl_->stream = &stream;
    impl_->device = stream.device();
    impl_->pointer = pointer;
    impl_->num_bytes = bytes;
#else
    (void)stream;
    (void)bytes;
    throw std::runtime_error("Stream ordered allocation requires HIP support");
#endif
}

StreamOrderedHipBuffer::~StreamOrderedHipBuffer() {
    try {
        release();
    } catch (...) {
    }
}

StreamOrderedHipBuffer::StreamOrderedHipBuffer(
    StreamOrderedHipBuffer&&) noexcept = default;

StreamOrderedHipBuffer& StreamOrderedHipBuffer::operator=(
    StreamOrderedHipBuffer&& other) noexcept {
    if (this == &other) return *this;
    try {
        release();
    } catch (...) {
    }
    impl_ = std::move(other.impl_);
    return *this;
}

bool StreamOrderedHipBuffer::defined() const noexcept {
    return impl_ && impl_->pointer != nullptr;
}

void* StreamOrderedHipBuffer::data() noexcept {
    return impl_ ? impl_->pointer : nullptr;
}

const void* StreamOrderedHipBuffer::data() const noexcept {
    return impl_ ? impl_->pointer : nullptr;
}

std::size_t StreamOrderedHipBuffer::bytes() const noexcept {
    return impl_ ? impl_->num_bytes : 0;
}

Device StreamOrderedHipBuffer::device() const noexcept {
    return impl_ ? impl_->device : Device::cpu();
}

void StreamOrderedHipBuffer::release() {
    if (!defined()) return;
#if MICROLLM_HAS_HIP
    set_device(impl_->device);
    check_hip(hipFreeAsync(
                  impl_->pointer, as_stream(impl_->stream->native_handle())),
              "hipFreeAsync");
    impl_->pointer = nullptr;
    impl_->num_bytes = 0;
#else
    throw std::runtime_error("Stream ordered release requires HIP support");
#endif
}

struct HipActivationArena::Impl {
    const Stream* stream = nullptr;
    Device device = Device::cpu();
    void* pointer = nullptr;
    std::size_t capacity = 0;
    std::size_t cursor = 0;
};

HipActivationArena::HipActivationArena(
    const Stream& stream, std::size_t capacity_bytes)
    : impl_(std::make_unique<Impl>()) {
    if (stream.device().is_cpu()) {
        throw std::invalid_argument("activation arena requires a HIP Stream");
    }
    if (capacity_bytes == 0) {
        throw std::invalid_argument("activation arena capacity must be positive");
    }
    impl_->stream = &stream;
    impl_->device = stream.device();
    impl_->pointer = runtime::allocate(capacity_bytes, stream.device());
    impl_->capacity = capacity_bytes;
}

HipActivationArena::~HipActivationArena() {
    if (!impl_ || impl_->pointer == nullptr) return;
    try {
        impl_->stream->synchronize();
    } catch (...) {
    }
    runtime::deallocate(impl_->pointer, impl_->device, impl_->capacity);
}

void* HipActivationArena::allocate_slice(
    std::size_t bytes, std::size_t alignment) {
    if (bytes == 0) {
        throw std::invalid_argument("activation arena slice must be non-empty");
    }
    if (alignment == 0 || (alignment & (alignment - 1)) != 0) {
        throw std::invalid_argument("activation arena alignment must be a power of two");
    }
    const auto padding = (alignment - (impl_->cursor & (alignment - 1))) &
                         (alignment - 1);
    const auto remaining = impl_->capacity - impl_->cursor;
    if (padding > remaining || bytes > remaining - padding) {
        throw std::overflow_error("activation arena capacity exceeded");
    }
    impl_->cursor += padding;
    auto* result = static_cast<std::byte*>(impl_->pointer) + impl_->cursor;
    impl_->cursor += bytes;
    return result;
}

void HipActivationArena::reset_plan() noexcept {
    if (impl_) impl_->cursor = 0;
}

void* HipActivationArena::data() noexcept {
    return impl_ ? impl_->pointer : nullptr;
}

const void* HipActivationArena::data() const noexcept {
    return impl_ ? impl_->pointer : nullptr;
}

Device HipActivationArena::device() const noexcept {
    return impl_ ? impl_->device : Device::cpu();
}

std::size_t HipActivationArena::capacity_bytes() const noexcept {
    return impl_ ? impl_->capacity : 0;
}

std::size_t HipActivationArena::planned_bytes() const noexcept {
    return impl_ ? impl_->cursor : 0;
}

struct HipGraphExecutable::Impl {
    Device device = Device::cpu();
    void* executable = nullptr;
    std::size_t nodes = 0;
};

HipGraphExecutable::HipGraphExecutable() = default;

HipGraphExecutable::HipGraphExecutable(std::unique_ptr<Impl> impl)
    : impl_(std::move(impl)) {}

HipGraphExecutable::~HipGraphExecutable() {
#if MICROLLM_HAS_HIP
    if (impl_ && impl_->executable != nullptr) {
        (void)hipSetDevice(impl_->device.index());
        (void)hipGraphExecDestroy(
            reinterpret_cast<hipGraphExec_t>(impl_->executable));
    }
#endif
}

HipGraphExecutable::HipGraphExecutable(HipGraphExecutable&&) noexcept = default;
HipGraphExecutable& HipGraphExecutable::operator=(HipGraphExecutable&&) noexcept = default;

HipGraphExecutable HipGraphExecutable::capture(
    const Stream& stream, const std::function<void()>& capture_work) {
    if (!capture_work) throw std::invalid_argument("HIP graph capture work is empty");
    if (stream.device().is_cpu()) {
        throw std::runtime_error("HIP graph capture requires a HIP Stream");
    }
#if MICROLLM_HAS_HIP
    set_device(stream.device());
    const auto native_stream = as_stream(stream.native_handle());
    check_hip(hipStreamBeginCapture(native_stream, hipStreamCaptureModeThreadLocal),
              "hipStreamBeginCapture");
    try {
        capture_work();
    } catch (...) {
        hipGraph_t abandoned = nullptr;
        if (hipStreamEndCapture(native_stream, &abandoned) == hipSuccess &&
            abandoned != nullptr) {
            (void)hipGraphDestroy(abandoned);
        }
        // Capture-unsafe calls such as synchronous hipMalloc can leave a
        // sticky launch error even after EndCapture resets the Stream state.
        // Preserve the original C++ exception while making the Stream usable
        // by an eager fallback or a later valid capture.
        (void)hipGetLastError();
        throw;
    }

    hipGraph_t graph = nullptr;
    const auto end_status = hipStreamEndCapture(native_stream, &graph);
    if (end_status != hipSuccess) {
        if (graph != nullptr) (void)hipGraphDestroy(graph);
        (void)hipGetLastError();
        check_hip(end_status, "hipStreamEndCapture");
    }

    std::size_t nodes = 0;
    const auto nodes_status = hipGraphGetNodes(graph, nullptr, &nodes);
    if (nodes_status != hipSuccess) {
        (void)hipGraphDestroy(graph);
        check_hip(nodes_status, "hipGraphGetNodes");
    }
    if (nodes == 0) {
        (void)hipGraphDestroy(graph);
        throw std::invalid_argument("HIP graph capture produced no device nodes");
    }

    hipGraphExec_t executable = nullptr;
    hipGraphNode_t error_node = nullptr;
    std::array<char, 2048> log{};
    const auto instantiate_status = hipGraphInstantiate(
        &executable, graph, &error_node, log.data(), log.size());
    (void)error_node;
    (void)hipGraphDestroy(graph);
    if (instantiate_status != hipSuccess) {
        if (executable != nullptr) (void)hipGraphExecDestroy(executable);
        std::string message = "hipGraphInstantiate: ";
        message += hipGetErrorString(instantiate_status);
        if (log.front() != '\0') message += std::string("; ") + log.data();
        throw std::runtime_error(message);
    }

    auto impl = std::make_unique<Impl>();
    impl->device = stream.device();
    impl->executable = reinterpret_cast<void*>(executable);
    impl->nodes = nodes;
    return HipGraphExecutable(std::move(impl));
#else
    (void)stream;
    throw std::runtime_error("microLLM was built without HIP graph support");
#endif
}

bool HipGraphExecutable::defined() const noexcept {
    return impl_ && impl_->executable != nullptr;
}

Device HipGraphExecutable::device() const {
    if (!defined()) throw std::logic_error("undefined HIP graph has no device");
    return impl_->device;
}

std::size_t HipGraphExecutable::node_count() const {
    if (!defined()) throw std::logic_error("undefined HIP graph has no nodes");
    return impl_->nodes;
}

void HipGraphExecutable::launch(const Stream& stream) const {
    if (!defined()) throw std::logic_error("cannot launch an undefined HIP graph");
    if (stream.device() != impl_->device) {
        throw std::invalid_argument("HIP graph launch Stream uses a different device");
    }
#if MICROLLM_HAS_HIP
    set_device(impl_->device);
    check_hip(hipGraphLaunch(
                  reinterpret_cast<hipGraphExec_t>(impl_->executable),
                  as_stream(stream.native_handle())),
              "hipGraphLaunch");
#else
    (void)stream;
    throw std::runtime_error("microLLM was built without HIP graph support");
#endif
}

struct DeferredHipDeallocationScope::Impl {
    struct Record {
        void* pointer = nullptr;
        std::size_t bytes = 0;
    };
    const Stream* stream = nullptr;
    Device device = Device::cpu();
    std::unique_ptr<Record[]> records;
    std::size_t capacity = 0;
    std::size_t pending = 0;
    std::size_t pending_bytes = 0;
    std::size_t total_blocks = 0;
    std::size_t total_bytes = 0;
    std::size_t overflow_count = 0;
    bool is_finished = false;
    int failure = 0;
};

DeferredHipDeallocationScope::DeferredHipDeallocationScope(
    const Stream& stream, std::size_t maximum_blocks)
    : impl_(std::make_unique<Impl>()) {
    if (stream.device().is_cpu()) {
        throw std::invalid_argument("deferred HIP deallocation requires a HIP Stream");
    }
    if (maximum_blocks == 0) {
        throw std::invalid_argument("deferred HIP deallocation capacity must be positive");
    }
    if (active_deferred_scope != nullptr) {
        throw std::logic_error("deferred HIP deallocation scopes cannot be nested");
    }
#if MICROLLM_HAS_HIP
    impl_->stream = &stream;
    impl_->device = stream.device();
    impl_->capacity = maximum_blocks;
    impl_->records = std::make_unique<Impl::Record[]>(maximum_blocks);
    active_deferred_scope = this;
#else
    (void)stream;
    (void)maximum_blocks;
    throw std::runtime_error("microLLM was built without deferred HIP deallocation");
#endif
}

DeferredHipDeallocationScope::~DeferredHipDeallocationScope() {
    try {
        finish();
    } catch (...) {
        if (active_deferred_scope == this) active_deferred_scope = nullptr;
    }
}

Device DeferredHipDeallocationScope::device() const noexcept {
    return impl_ ? impl_->device : Device::cpu();
}

bool DeferredHipDeallocationScope::finished() const noexcept {
    return !impl_ || impl_->is_finished;
}

std::size_t DeferredHipDeallocationScope::pending_blocks() const noexcept {
    return impl_ ? impl_->pending : 0;
}

std::size_t DeferredHipDeallocationScope::pending_bytes() const noexcept {
    return impl_ ? impl_->pending_bytes : 0;
}

std::size_t DeferredHipDeallocationScope::total_deferred_blocks() const noexcept {
    return impl_ ? impl_->total_blocks : 0;
}

std::size_t DeferredHipDeallocationScope::total_deferred_bytes() const noexcept {
    return impl_ ? impl_->total_bytes : 0;
}

std::size_t DeferredHipDeallocationScope::overflow_flushes() const noexcept {
    return impl_ ? impl_->overflow_count : 0;
}

void DeferredHipDeallocationScope::defer(
    void* pointer, std::size_t num_bytes) noexcept {
#if MICROLLM_HAS_HIP
    if (!impl_ || impl_->is_finished || pointer == nullptr) return;
    if (impl_->pending == impl_->capacity) {
        const auto synchronize_status = hipStreamSynchronize(
            as_stream(impl_->stream->native_handle()));
        if (synchronize_status != hipSuccess) {
            impl_->failure = static_cast<int>(synchronize_status);
            return;
        }
        for (std::size_t index = 0; index < impl_->pending; ++index) {
            const auto& record = impl_->records[index];
            const auto free_status = hipFree(record.pointer);
            if (free_status != hipSuccess) {
                impl_->failure = static_cast<int>(free_status);
                continue;
            }
            counters(impl_->device).backend_deallocation_calls.fetch_add(1);
            counters(impl_->device).reserved_bytes.fetch_sub(record.bytes);
        }
        impl_->pending = 0;
        impl_->pending_bytes = 0;
        ++impl_->overflow_count;
    }
    impl_->records[impl_->pending++] = {pointer, num_bytes};
    impl_->pending_bytes += num_bytes;
    ++impl_->total_blocks;
    impl_->total_bytes += num_bytes;
#else
    (void)pointer;
    (void)num_bytes;
#endif
}

void DeferredHipDeallocationScope::finish() {
    if (!impl_ || impl_->is_finished) return;
    if (active_deferred_scope != nullptr && active_deferred_scope != this) {
        throw std::logic_error("another deferred HIP deallocation scope is active");
    }
    if (active_deferred_scope == this) active_deferred_scope = nullptr;
#if MICROLLM_HAS_HIP
    impl_->stream->synchronize();
    set_device(impl_->device);
    for (std::size_t index = 0; index < impl_->pending; ++index) {
        auto& record = impl_->records[index];
        const auto status = hipFree(record.pointer);
        if (status != hipSuccess) {
            if (impl_->failure == 0) impl_->failure = static_cast<int>(status);
            continue;
        }
        counters(impl_->device).backend_deallocation_calls.fetch_add(1);
        counters(impl_->device).reserved_bytes.fetch_sub(record.bytes);
        record.pointer = nullptr;
    }
    impl_->pending = 0;
    impl_->pending_bytes = 0;
    impl_->is_finished = true;
    if (impl_->failure != 0) {
        check_hip(static_cast<hipError_t>(impl_->failure),
                  "deferred HIP deallocation release");
    }
#else
    throw std::runtime_error("microLLM was built without deferred HIP deallocation");
#endif
}

struct ScopedDeferredHipStream::Impl {
    const Stream* stream = nullptr;
    std::unique_ptr<DeferredHipDeallocationScope> lifetime;
    bool is_finished = false;
};

ScopedDeferredHipStream::ScopedDeferredHipStream(
    const Stream& stream, std::size_t maximum_blocks)
    : impl_(std::make_unique<Impl>()) {
    if (stream.device().is_cpu()) {
        throw std::invalid_argument("scoped deferred Stream requires a HIP Stream");
    }
    if (active_scoped_deferred_stream != nullptr) {
        throw std::logic_error("scoped deferred HIP Streams cannot be nested");
    }
    impl_->stream = &stream;
    impl_->lifetime = std::make_unique<DeferredHipDeallocationScope>(
        stream, maximum_blocks);
    active_scoped_deferred_stream = this;
}

ScopedDeferredHipStream::~ScopedDeferredHipStream() {
    try {
        finish();
    } catch (...) {
        if (active_scoped_deferred_stream == this) {
            active_scoped_deferred_stream = nullptr;
        }
    }
}

Device ScopedDeferredHipStream::device() const noexcept {
    return impl_ && impl_->stream != nullptr ? impl_->stream->device()
                                             : Device::cpu();
}

const Stream& ScopedDeferredHipStream::stream() const {
    if (!impl_ || impl_->stream == nullptr) {
        throw std::logic_error("scoped deferred HIP Stream is undefined");
    }
    return *impl_->stream;
}

bool ScopedDeferredHipStream::finished() const noexcept {
    return !impl_ || impl_->is_finished;
}

std::size_t ScopedDeferredHipStream::pending_blocks() const noexcept {
    return impl_ && impl_->lifetime ? impl_->lifetime->pending_blocks() : 0;
}

std::size_t ScopedDeferredHipStream::pending_bytes() const noexcept {
    return impl_ && impl_->lifetime ? impl_->lifetime->pending_bytes() : 0;
}

std::size_t ScopedDeferredHipStream::total_deferred_blocks() const noexcept {
    return impl_ && impl_->lifetime
               ? impl_->lifetime->total_deferred_blocks()
               : 0;
}

std::size_t ScopedDeferredHipStream::total_deferred_bytes() const noexcept {
    return impl_ && impl_->lifetime
               ? impl_->lifetime->total_deferred_bytes()
               : 0;
}

std::size_t ScopedDeferredHipStream::overflow_flushes() const noexcept {
    return impl_ && impl_->lifetime ? impl_->lifetime->overflow_flushes() : 0;
}

void ScopedDeferredHipStream::finish() {
    if (!impl_ || impl_->is_finished) return;
    if (active_scoped_deferred_stream != this) {
        throw std::logic_error("scoped deferred HIP Stream is not active");
    }
    active_scoped_deferred_stream = nullptr;
    try {
        impl_->lifetime->finish();
        impl_->is_finished = true;
    } catch (...) {
        impl_->is_finished = impl_->lifetime->finished();
        throw;
    }
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

void Event::record_default_stream() {
    if (impl_->device.is_cpu()) {
        throw std::runtime_error("CPU events do not record device work");
    }
#if MICROLLM_HAS_HIP
    set_device(impl_->device);
    check_hip(hipEventRecord(as_event(impl_->handle), nullptr),
              "hipEventRecord(default stream)");
#else
    throw std::runtime_error("microLLM was built without HIP support");
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
    const auto resolved_stream = resolve_deferred_hip_stream(
        stream.device(), stream.native_handle());
    check_hip(hipMemcpyAsync(destination, source, num_bytes,
                             copy_kind(destination_device, source_device),
                             as_stream(resolved_stream)),
              "hipMemcpyAsync");
#else
    throw std::runtime_error("HIP copy requested from a CPU-only build");
#endif
}

}  // namespace microllm::runtime
