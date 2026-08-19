#include <microllm/core/storage.h>
#include <microllm/runtime/memory.h>

#include <utility>

namespace microllm {

struct Storage::Allocation {
    void* data = nullptr;
    std::size_t num_bytes = 0;
    Device device = Device::cpu();

    ~Allocation() { runtime::deallocate(data, device); }
};

Storage::Storage(std::size_t num_bytes, Device device) {
    auto allocation = std::make_shared<Allocation>();
    allocation->num_bytes = num_bytes;
    allocation->device = device;
    if (num_bytes != 0) {
        allocation->data = runtime::allocate(num_bytes, device);
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
