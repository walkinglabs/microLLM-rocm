#pragma once

#include <cstdint>
#include <string>

namespace microllm {

enum class DeviceType : std::uint8_t { CPU, HIP };

class Device {
public:
    [[nodiscard]] static Device cpu(int index = 0);
    [[nodiscard]] static Device hip(int index = 0);

    [[nodiscard]] DeviceType type() const noexcept { return type_; }
    [[nodiscard]] int index() const noexcept { return index_; }
    [[nodiscard]] bool is_cpu() const noexcept { return type_ == DeviceType::CPU; }
    [[nodiscard]] bool is_hip() const noexcept { return type_ == DeviceType::HIP; }
    [[nodiscard]] std::string str() const;

    friend bool operator==(const Device&, const Device&) = default;

private:
    Device(DeviceType type, int index) : type_(type), index_(index) {}

    DeviceType type_;
    int index_;
};

}  // namespace microllm
