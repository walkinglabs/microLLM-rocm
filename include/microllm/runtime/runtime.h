#pragma once

#include <cstddef>
#include <functional>
#include <memory>
#include <string>

#include <microllm/base/device.h>

namespace microllm::runtime {

struct DeviceInfo {
    Device device = Device::cpu();
    std::string name;
    std::string architecture;
    std::size_t total_memory = 0;
    int multiprocessor_count = 0;
    int warp_size = 0;
};

struct MemoryInfo {
    std::size_t free_bytes = 0;
    std::size_t total_bytes = 0;
};

struct HipMemoryPoolStats {
    bool supported = false;
    std::size_t reserved_current_bytes = 0;
    std::size_t reserved_high_bytes = 0;
    std::size_t used_current_bytes = 0;
    std::size_t used_high_bytes = 0;
    std::uint64_t release_threshold_bytes = 0;
};

struct TransferStats {
    std::size_t host_to_device_calls = 0;
    std::size_t device_to_host_calls = 0;
    std::size_t device_to_device_calls = 0;
    std::size_t host_to_device_bytes = 0;
    std::size_t device_to_host_bytes = 0;
    std::size_t device_to_device_bytes = 0;
};

struct PrecisionCapabilities {
    std::string architecture;
    bool fp64 = false;
    bool fp32 = false;
    bool tf32_hardware = false;
    bool fp16 = false;
    bool bfloat16 = false;
    bool fp8_fnuz = false;
    bool fp8_ocp = false;
    bool int8_matrix = false;
    bool mxfp8 = false;
    bool mxfp6 = false;
    bool mxfp4 = false;
    bool int4_matrix = false;
    bool packed_int4_software = true;
};

[[nodiscard]] bool hip_compiled() noexcept;
[[nodiscard]] int hip_device_count();
[[nodiscard]] DeviceInfo device_info(Device device);
[[nodiscard]] PrecisionCapabilities precision_capabilities(Device device);
[[nodiscard]] MemoryInfo memory_info(Device device);
[[nodiscard]] bool stream_ordered_allocator_supported(Device device);
[[nodiscard]] HipMemoryPoolStats default_hip_memory_pool_stats(Device device);
void set_default_hip_memory_pool_release_threshold(
    Device device, std::uint64_t bytes);
void trim_default_hip_memory_pool(Device device, std::size_t minimum_bytes_to_hold = 0);
[[nodiscard]] int hip_runtime_version();
[[nodiscard]] int hip_driver_version();
void set_device(Device device);
void synchronize(Device device);
[[nodiscard]] TransferStats transfer_stats() noexcept;
void reset_transfer_stats() noexcept;
// The exact-size allocator cache is only safe for legacy-default-stream work.
// Creating or passing any non-default stream permanently disables reuse on that device.
void notify_non_default_stream(Device device) noexcept;
void enable_hip_caching_allocator(Device device);
[[nodiscard]] bool hip_caching_allocator_enabled(Device device) noexcept;

class Stream {
public:
    explicit Stream(Device device = Device::cpu(), bool non_blocking = true);
    ~Stream();
    Stream(Stream&&) noexcept;
    Stream& operator=(Stream&&) noexcept;
    Stream(const Stream&) = delete;
    Stream& operator=(const Stream&) = delete;

    [[nodiscard]] Device device() const noexcept;
    [[nodiscard]] void* native_handle() const noexcept;
    void synchronize() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

// A raw allocation whose creation and release are ordered on one explicit HIP
// Stream. The Stream must outlive this object. This beta HIP mechanism is
// opt-in and does not change ordinary Storage allocation.
class StreamOrderedHipBuffer {
public:
    StreamOrderedHipBuffer(const Stream& stream, std::size_t bytes);
    ~StreamOrderedHipBuffer();
    StreamOrderedHipBuffer(StreamOrderedHipBuffer&&) noexcept;
    StreamOrderedHipBuffer& operator=(StreamOrderedHipBuffer&&) noexcept;
    StreamOrderedHipBuffer(const StreamOrderedHipBuffer&) = delete;
    StreamOrderedHipBuffer& operator=(const StreamOrderedHipBuffer&) = delete;

    [[nodiscard]] bool defined() const noexcept;
    [[nodiscard]] void* data() noexcept;
    [[nodiscard]] const void* data() const noexcept;
    [[nodiscard]] std::size_t bytes() const noexcept;
    [[nodiscard]] Device device() const noexcept;
    void release();

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

// A captured HIP execution graph. Every pointer referenced by capture_work must
// remain valid until the last launch completes. Capture and replay use an
// explicit Stream; CPU and legacy-default-Stream capture are intentionally out
// of this contract.
class HipGraphExecutable {
public:
    HipGraphExecutable();
    ~HipGraphExecutable();
    HipGraphExecutable(HipGraphExecutable&&) noexcept;
    HipGraphExecutable& operator=(HipGraphExecutable&&) noexcept;
    HipGraphExecutable(const HipGraphExecutable&) = delete;
    HipGraphExecutable& operator=(const HipGraphExecutable&) = delete;

    [[nodiscard]] static HipGraphExecutable capture(
        const Stream& stream, const std::function<void()>& capture_work);
    [[nodiscard]] bool defined() const noexcept;
    [[nodiscard]] Device device() const;
    [[nodiscard]] std::size_t node_count() const;
    void launch(const Stream& stream) const;

private:
    struct Impl;
    explicit HipGraphExecutable(std::unique_ptr<Impl> impl);
    std::unique_ptr<Impl> impl_;
};

// Defers HIP frees produced by the current thread until one explicit Stream
// has completed. This extends raw allocation lifetime; it does not route
// operator work to the Stream and it is intentionally non-nestable.
class DeferredHipDeallocationScope {
public:
    explicit DeferredHipDeallocationScope(
        const Stream& stream, std::size_t maximum_blocks = 8192);
    ~DeferredHipDeallocationScope();
    DeferredHipDeallocationScope(const DeferredHipDeallocationScope&) = delete;
    DeferredHipDeallocationScope& operator=(const DeferredHipDeallocationScope&) = delete;
    DeferredHipDeallocationScope(DeferredHipDeallocationScope&&) = delete;
    DeferredHipDeallocationScope& operator=(DeferredHipDeallocationScope&&) = delete;

    [[nodiscard]] Device device() const noexcept;
    [[nodiscard]] bool finished() const noexcept;
    [[nodiscard]] std::size_t pending_blocks() const noexcept;
    [[nodiscard]] std::size_t pending_bytes() const noexcept;
    [[nodiscard]] std::size_t total_deferred_blocks() const noexcept;
    [[nodiscard]] std::size_t total_deferred_bytes() const noexcept;
    [[nodiscard]] std::size_t overflow_flushes() const noexcept;
    void finish();

private:
    friend void deallocate(void* pointer, Device device,
                           std::size_t num_bytes) noexcept;
    void defer(void* pointer, std::size_t num_bytes) noexcept;
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

// Routes otherwise-default HIP operator/runtime work to one Stream and keeps
// destroyed temporary allocations alive until that same Stream completes.
// Routing work without routing lifetime is unsafe for eager Tensor temporaries.
class ScopedDeferredHipStream {
public:
    explicit ScopedDeferredHipStream(
        const Stream& stream, std::size_t maximum_blocks = 8192);
    ~ScopedDeferredHipStream();
    ScopedDeferredHipStream(const ScopedDeferredHipStream&) = delete;
    ScopedDeferredHipStream& operator=(const ScopedDeferredHipStream&) = delete;
    ScopedDeferredHipStream(ScopedDeferredHipStream&&) = delete;
    ScopedDeferredHipStream& operator=(ScopedDeferredHipStream&&) = delete;

    [[nodiscard]] Device device() const noexcept;
    [[nodiscard]] const Stream& stream() const;
    [[nodiscard]] bool finished() const noexcept;
    [[nodiscard]] std::size_t pending_blocks() const noexcept;
    [[nodiscard]] std::size_t pending_bytes() const noexcept;
    [[nodiscard]] std::size_t total_deferred_blocks() const noexcept;
    [[nodiscard]] std::size_t total_deferred_bytes() const noexcept;
    [[nodiscard]] std::size_t overflow_flushes() const noexcept;
    void finish();

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

// Internal routing seam shared by OpContext and runtime layout copies. When a
// safe scope is active, an explicit different Stream is rejected.
[[nodiscard]] void* resolve_deferred_hip_stream(
    Device device, void* explicitly_requested_stream = nullptr);

class Event {
public:
    explicit Event(Device device = Device::cpu(), bool enable_timing = true);
    ~Event();
    Event(Event&&) noexcept;
    Event& operator=(Event&&) noexcept;
    Event(const Event&) = delete;
    Event& operator=(const Event&) = delete;

    [[nodiscard]] Device device() const noexcept;
    [[nodiscard]] void* native_handle() const noexcept;
    void record(const Stream& stream);
    // Records on the legacy default Stream without creating a non-default
    // Stream or disabling the exact-size allocator contract.
    void record_default_stream();
    void wait(const Stream& stream) const;
    void synchronize() const;
    [[nodiscard]] bool ready() const;
    [[nodiscard]] float elapsed_ms_since(const Event& start) const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

void copy_bytes_async(void* destination, Device destination_device, const void* source,
                      Device source_device, std::size_t num_bytes, const Stream& stream);

}  // namespace microllm::runtime
