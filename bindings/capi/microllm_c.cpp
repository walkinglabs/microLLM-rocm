#include <microllm/capi/microllm.h>

#include <algorithm>
#include <exception>
#include <memory>
#include <new>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <microllm/ops/ops.h>
#include <microllm/runtime/runtime.h>

#if defined(_WIN32)
#define ML_EXPORT __declspec(dllexport)
#else
#define ML_EXPORT __attribute__((visibility("default")))
#endif

struct ml_tensor {
    microllm::Tensor value;
};

struct ml_event {
    ml_event(microllm::Device event_device, bool enable_timing)
        : device(event_device), value(event_device, enable_timing) {}
    microllm::Device device;
    microllm::runtime::Event value;
};

namespace {

thread_local std::string last_error;

microllm::Device device_from_c(ml_device_type type, int index) {
    switch (type) {
        case ML_DEVICE_CPU: return microllm::Device::cpu(index);
        case ML_DEVICE_HIP: return microllm::Device::hip(index);
    }
    throw std::invalid_argument("unknown C API device type");
}

microllm::Shape shape_from_c(const int64_t* shape, size_t rank) {
    if (rank != 0 && shape == nullptr) throw std::invalid_argument("shape pointer is null");
    if (rank == 0) return {};
    return microllm::Shape(shape, shape + rank);
}

template <typename Function>
ml_status guard(Function&& function) noexcept {
    try {
        last_error.clear();
        function();
        return ML_STATUS_OK;
    } catch (const std::invalid_argument& error) {
        last_error = error.what();
        return ML_STATUS_INVALID_ARGUMENT;
    } catch (const std::out_of_range& error) {
        last_error = error.what();
        return ML_STATUS_OUT_OF_RANGE;
    } catch (const std::exception& error) {
        last_error = error.what();
        return ML_STATUS_RUNTIME_ERROR;
    } catch (...) {
        last_error = "unknown C++ exception";
        return ML_STATUS_UNKNOWN_ERROR;
    }
}

const microllm::Tensor& require_tensor(const ml_tensor* tensor) {
    if (tensor == nullptr) throw std::invalid_argument("tensor is null");
    return tensor->value;
}

ml_event& require_event(ml_event* event) {
    if (event == nullptr) throw std::invalid_argument("event is null");
    return *event;
}

const ml_event& require_event(const ml_event* event) {
    if (event == nullptr) throw std::invalid_argument("event is null");
    return *event;
}

void require_output(ml_tensor** output) {
    if (output == nullptr) throw std::invalid_argument("output tensor pointer is null");
    *output = nullptr;
}

void return_tensor(microllm::Tensor value, ml_tensor** output) {
    auto result = std::make_unique<ml_tensor>();
    result->value = std::move(value);
    *output = result.release();
}

void require_event_output(ml_event** output) {
    if (output == nullptr) throw std::invalid_argument("output event pointer is null");
    *output = nullptr;
}

template <typename Operation>
ml_status binary_operation(const ml_tensor* left, const ml_tensor* right,
                           ml_tensor** output, Operation&& operation) noexcept {
    return guard([&] {
        require_output(output);
        return_tensor(operation(require_tensor(left), require_tensor(right)), output);
    });
}

}  // namespace

extern "C" {

ML_EXPORT uint32_t ml_capi_version(void) { return ML_CAPI_VERSION; }
ML_EXPORT const char* ml_engine_version(void) { return MICROLLM_VERSION; }
ML_EXPORT const char* ml_last_error(void) { return last_error.c_str(); }
ML_EXPORT ml_status ml_hip_device_count(int* count) {
    return guard([&] {
        if (count == nullptr) throw std::invalid_argument("device count output is null");
        *count = microllm::runtime::hip_device_count();
    });
}

ML_EXPORT ml_status ml_tensor_from_f32(const float* values, const int64_t* shape,
                                       size_t rank, ml_device_type device_type,
                                       int device_index, ml_tensor** output) {
    return guard([&] {
        require_output(output);
        auto tensor_shape = shape_from_c(shape, rank);
        const auto elements = microllm::checked_numel(tensor_shape);
        if (elements != 0 && values == nullptr) throw std::invalid_argument("values pointer is null");
        std::vector<float> host_values;
        if (elements != 0) host_values.assign(values, values + elements);
        return_tensor(microllm::Tensor::from_vector(host_values, std::move(tensor_shape))
                          .to(device_from_c(device_type, device_index)),
                      output);
    });
}

ML_EXPORT ml_status ml_tensor_from_i32(const int32_t* values, const int64_t* shape,
                                       size_t rank, ml_device_type device_type,
                                       int device_index, ml_tensor** output) {
    return guard([&] {
        require_output(output);
        auto tensor_shape = shape_from_c(shape, rank);
        const auto elements = microllm::checked_numel(tensor_shape);
        if (elements != 0 && values == nullptr) throw std::invalid_argument("values pointer is null");
        std::vector<int32_t> host_values;
        if (elements != 0) host_values.assign(values, values + elements);
        return_tensor(microllm::Tensor::from_int32_vector(host_values, std::move(tensor_shape))
                          .to(device_from_c(device_type, device_index)),
                      output);
    });
}

ML_EXPORT void ml_tensor_destroy(ml_tensor* tensor) { delete tensor; }

ML_EXPORT ml_status ml_tensor_rank(const ml_tensor* tensor, size_t* rank) {
    return guard([&] {
        if (rank == nullptr) throw std::invalid_argument("rank output is null");
        *rank = static_cast<size_t>(require_tensor(tensor).ndim());
    });
}

ML_EXPORT ml_status ml_tensor_shape(const ml_tensor* tensor, size_t dim, int64_t* size) {
    return guard([&] {
        if (size == nullptr) throw std::invalid_argument("shape output is null");
        *size = require_tensor(tensor).size(static_cast<int64_t>(dim));
    });
}

ML_EXPORT ml_status ml_tensor_numel(const ml_tensor* tensor, int64_t* elements) {
    return guard([&] {
        if (elements == nullptr) throw std::invalid_argument("numel output is null");
        *elements = require_tensor(tensor).numel();
    });
}

ML_EXPORT ml_status ml_tensor_dtype(const ml_tensor* tensor, ml_dtype* dtype) {
    return guard([&] {
        if (dtype == nullptr) throw std::invalid_argument("dtype output is null");
        switch (require_tensor(tensor).dtype()) {
            case microllm::DType::Float32: *dtype = ML_DTYPE_FLOAT32; return;
            case microllm::DType::Int32: *dtype = ML_DTYPE_INT32; return;
            default: throw std::runtime_error("dtype is not exposed by C API version 1");
        }
    });
}

ML_EXPORT ml_status ml_tensor_device(const ml_tensor* tensor, ml_device_type* device_type,
                                     int* device_index) {
    return guard([&] {
        if (device_type == nullptr || device_index == nullptr) {
            throw std::invalid_argument("device outputs are null");
        }
        const auto device = require_tensor(tensor).device();
        *device_type = device.is_cpu() ? ML_DEVICE_CPU : ML_DEVICE_HIP;
        *device_index = device.index();
    });
}

ML_EXPORT ml_status ml_tensor_copy_f32(const ml_tensor* tensor, float* values,
                                       size_t capacity) {
    return guard([&] {
        const auto copied = require_tensor(tensor).to_vector();
        if (capacity < copied.size() || (values == nullptr && !copied.empty())) {
            throw std::invalid_argument("float output buffer is too small or null");
        }
        std::copy(copied.begin(), copied.end(), values);
    });
}

ML_EXPORT ml_status ml_tensor_copy_i32(const ml_tensor* tensor, int32_t* values,
                                       size_t capacity) {
    return guard([&] {
        const auto copied = require_tensor(tensor).to_int32_vector();
        if (capacity < copied.size() || (values == nullptr && !copied.empty())) {
            throw std::invalid_argument("int32 output buffer is too small or null");
        }
        std::copy(copied.begin(), copied.end(), values);
    });
}

ML_EXPORT ml_status ml_tensor_to(const ml_tensor* tensor, ml_device_type device_type,
                                 int device_index, ml_tensor** output) {
    return guard([&] {
        require_output(output);
        return_tensor(require_tensor(tensor).to(device_from_c(device_type, device_index)), output);
    });
}

ML_EXPORT ml_status ml_event_create(ml_device_type device_type, int device_index,
                                    int enable_timing, ml_event** output) {
    return guard([&] {
        require_event_output(output);
        const auto device = device_from_c(device_type, device_index);
        auto event = std::make_unique<ml_event>(device, enable_timing != 0);
        *output = event.release();
    });
}

ML_EXPORT void ml_event_destroy(ml_event* event) { delete event; }

ML_EXPORT ml_status ml_event_record_default_stream(ml_event* event) {
    return guard([&] {
        auto& required = require_event(event);
        if (required.device.is_cpu()) {
            const microllm::runtime::Stream stream(required.device);
            required.value.record(stream);
        } else {
            required.value.record_default_stream();
        }
    });
}

ML_EXPORT ml_status ml_event_ready(const ml_event* event, int* ready) {
    return guard([&] {
        if (ready == nullptr) throw std::invalid_argument("event ready output is null");
        *ready = require_event(event).value.ready() ? 1 : 0;
    });
}

ML_EXPORT ml_status ml_event_synchronize(const ml_event* event) {
    return guard([&] { require_event(event).value.synchronize(); });
}

ML_EXPORT ml_status ml_event_elapsed_ms(const ml_event* start,
                                        const ml_event* finish,
                                        float* milliseconds) {
    return guard([&] {
        if (milliseconds == nullptr) {
            throw std::invalid_argument("elapsed milliseconds output is null");
        }
        const auto& required_start = require_event(start);
        const auto& required_finish = require_event(finish);
        *milliseconds = required_finish.value.elapsed_ms_since(required_start.value);
    });
}

ML_EXPORT ml_status ml_add(const ml_tensor* left, const ml_tensor* right, ml_tensor** output) {
    return binary_operation(left, right, output,
                            [](const auto& first, const auto& second) {
                                return microllm::ops::add(first, second);
                            });
}
ML_EXPORT ml_status ml_multiply(const ml_tensor* left, const ml_tensor* right,
                                ml_tensor** output) {
    return binary_operation(left, right, output,
                            [](const auto& first, const auto& second) {
                                return microllm::ops::multiply(first, second);
                            });
}
ML_EXPORT ml_status ml_matmul(const ml_tensor* left, const ml_tensor* right,
                              ml_tensor** output) {
    return binary_operation(left, right, output,
                            [](const auto& first, const auto& second) {
                                return microllm::ops::matmul(first, second);
                            });
}
ML_EXPORT ml_status ml_softmax(const ml_tensor* input, ml_tensor** output) {
    return guard([&] {
        require_output(output);
        return_tensor(microllm::ops::softmax(require_tensor(input)), output);
    });
}

}  // extern "C"
