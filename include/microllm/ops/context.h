#pragma once

#include <cstddef>
#include <stdexcept>

#include <microllm/base/device.h>
#include <microllm/runtime/runtime.h>

namespace microllm::ops {

struct OpContext {
    runtime::Stream* stream = nullptr;
    void* workspace = nullptr;
    std::size_t workspace_bytes = 0;

    [[nodiscard]] void* native_stream(Device device) const {
        if (stream == nullptr) return nullptr;
        if (stream->device() != device) {
            throw std::invalid_argument("operator stream does not match tensor device");
        }
        return stream->native_handle();
    }
};

}  // namespace microllm::ops
