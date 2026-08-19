#pragma once

#include <cstddef>
#include <memory>

#include <microllm/base/device.h>

namespace microllm {

class Storage {
public:
    Storage() = default;
    explicit Storage(std::size_t num_bytes, Device device = Device::cpu());

    [[nodiscard]] void* data() noexcept;
    [[nodiscard]] const void* data() const noexcept;
    [[nodiscard]] std::size_t num_bytes() const noexcept;
    [[nodiscard]] Device device() const noexcept;
    [[nodiscard]] bool empty() const noexcept;
    [[nodiscard]] long use_count() const noexcept;

private:
    struct Allocation;
    std::shared_ptr<Allocation> allocation_;
};

}  // namespace microllm
