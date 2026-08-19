#pragma once

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

[[nodiscard]] bool hip_compiled() noexcept;
[[nodiscard]] int hip_device_count();
[[nodiscard]] DeviceInfo device_info(Device device);
void set_device(Device device);
void synchronize(Device device);

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
