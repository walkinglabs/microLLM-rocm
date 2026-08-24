#include <microllm/core/storage.h>
#include <microllm/runtime/memory.h>

#include <stdexcept>
#include <utility>

namespace microllm {

struct Storage::Allocation {
    void* data = nullptr;
    std::size_t num_bytes = 0;
    Device device = Device::cpu();
    bool owns_memory = true;

    ~Allocation() {
        if (owns_memory) runtime::deallocate(data, device, num_bytes);
    }
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

Storage Storage::from_external(
    void* pointer, std::size_t num_bytes, Device device) {
    if (num_bytes != 0 && pointer == nullptr) {
        throw std::invalid_argument(
            "external Storage pointer is null for nonzero bytes");
    }
    Storage result;
    auto allocation = std::make_shared<Allocation>();
    allocation->data = pointer;
    allocation->num_bytes = num_bytes;
    allocation->device = device;
    allocation->owns_memory = false;
    result.allocation_ = std::move(allocation);
    return result;
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
