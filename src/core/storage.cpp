#include <microllm/core/storage.h>

#include <new>
#include <stdexcept>
#include <utility>

namespace microllm {

struct Storage::Allocation {
    void* data = nullptr;
    std::size_t num_bytes = 0;
    Device device = Device::cpu();

    ~Allocation() { ::operator delete(data); }
};

Storage::Storage(std::size_t num_bytes, Device device) {
    if (!device.is_cpu()) {
        throw std::runtime_error("HIP Storage is not available until the N1 runtime milestone");
    }
    auto allocation = std::make_shared<Allocation>();
    allocation->num_bytes = num_bytes;
    allocation->device = device;
    if (num_bytes != 0) {
        allocation->data = ::operator new(num_bytes);
    }
    allocation_ = std::move(allocation);
}

void* Storage::data() noexcept { return allocation_ ? allocation_->data : nullptr; }
const void* Storage::data() const noexcept { return allocation_ ? allocation_->data : nullptr; }
std::size_t Storage::num_bytes() const noexcept {
    return allocation_ ? allocation_->num_bytes : 0;
}
Device Storage::device() const noexcept {
    return allocation_ ? allocation_->device : Device::cpu();
}
bool Storage::empty() const noexcept { return num_bytes() == 0; }
long Storage::use_count() const noexcept { return allocation_.use_count(); }

}  // namespace microllm
