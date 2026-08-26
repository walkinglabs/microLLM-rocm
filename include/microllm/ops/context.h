#pragma once

#include <cstddef>
#include <stdexcept>

#include <microllm/base/device.h>
#include <microllm/runtime/runtime.h>

namespace microllm::ops {

enum class OpMode { Unspecified, Inference, Training };
enum class Fp32SolutionScope {
    General,
    PrefillQueryProjection,
    PrefillKeyValueProjection,
};

struct OpContext {
    runtime::Stream* stream = nullptr;
    void* workspace = nullptr;
    std::size_t workspace_bytes = 0;
    void* external_stream = nullptr;
    Device external_stream_device = Device::cpu();
    OpMode mode = OpMode::Unspecified;
    Fp32SolutionScope fp32_solution_scope = Fp32SolutionScope::General;

    [[nodiscard]] static OpContext from_external_stream(Device device, void* native_stream) {
        OpContext context;
        context.external_stream = native_stream;
        context.external_stream_device = device;
        return context;
    }

    [[nodiscard]] void* native_stream(Device device) const {
        if (device.is_hip()) runtime::set_device(device);
        if (stream != nullptr && external_stream != nullptr) {
            throw std::invalid_argument("operator context cannot contain two streams");
        }
        void* requested = nullptr;
        if (stream != nullptr) {
            if (stream->device() != device) {
                throw std::invalid_argument("operator stream does not match tensor device");
            }
            runtime::notify_non_default_stream(device);
            requested = stream->native_handle();
        } else if (external_stream != nullptr) {
            if (external_stream_device != device) {
                throw std::invalid_argument("external stream does not match tensor device");
            }
            runtime::notify_non_default_stream(device);
            requested = external_stream;
        }
        return runtime::resolve_deferred_hip_stream(device, requested);
    }
};

}  // namespace microllm::ops
