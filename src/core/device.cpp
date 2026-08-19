#include <microllm/base/device.h>

#include <stdexcept>

namespace microllm {

Device Device::cpu(int index) {
    if (index < 0) throw std::invalid_argument("device index must be non-negative");
    return {DeviceType::CPU, index};
}

Device Device::hip(int index) {
    if (index < 0) throw std::invalid_argument("device index must be non-negative");
    return {DeviceType::HIP, index};
}

std::string Device::str() const {
    return std::string(is_cpu() ? "cpu:" : "hip:") + std::to_string(index_);
}

}  // namespace microllm
