#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include <microllm/capi/microllm.h>

#define CHECK(expression)                                                        \
    do {                                                                         \
        if (!(expression)) {                                                     \
            fprintf(stderr, "C API check failed at line %d: %s (%s)\n",         \
                    __LINE__, #expression, ml_last_error());                     \
            return __LINE__;                                                     \
        }                                                                        \
    } while (0)

int main(void) {
    CHECK(ml_capi_version() == ML_CAPI_VERSION);
    CHECK(strlen(ml_engine_version()) > 0);
    const int64_t shape[2] = {2, 2};
    const float left_values[4] = {1, 2, 3, 4};
    const float right_values[4] = {5, 6, 7, 8};
    ml_tensor* left = NULL;
    ml_tensor* right = NULL;
    ml_tensor* sum = NULL;
    ml_tensor* product = NULL;
    CHECK(ml_tensor_from_f32(left_values, shape, 2, ML_DEVICE_CPU, 0, &left) == ML_STATUS_OK);
    CHECK(ml_tensor_from_f32(right_values, shape, 2, ML_DEVICE_CPU, 0, &right) == ML_STATUS_OK);
    CHECK(ml_add(left, right, &sum) == ML_STATUS_OK);
    CHECK(ml_matmul(left, right, &product) == ML_STATUS_OK);
    float output[4] = {0};
    CHECK(ml_tensor_copy_f32(sum, output, 4) == ML_STATUS_OK);
    CHECK(output[0] == 6 && output[1] == 8 && output[2] == 10 && output[3] == 12);
    CHECK(ml_tensor_copy_f32(product, output, 4) == ML_STATUS_OK);
    CHECK(output[0] == 19 && output[1] == 22 && output[2] == 43 && output[3] == 50);
    ml_tensor* invalid = NULL;
    CHECK(ml_add(left, NULL, &invalid) == ML_STATUS_INVALID_ARGUMENT);
    CHECK(invalid == NULL);
    CHECK(strlen(ml_last_error()) > 0);
    ml_event* cpu_event = NULL;
    ml_stream* cpu_stream = NULL;
    int event_ready = 0;
    CHECK(ml_event_create(ML_DEVICE_CPU, 0, 0, &cpu_event) == ML_STATUS_OK);
    CHECK(ml_event_ready(cpu_event, &event_ready) == ML_STATUS_OK);
    CHECK(event_ready == 0);
    CHECK(ml_event_record_default_stream(cpu_event) == ML_STATUS_OK);
    CHECK(ml_event_ready(cpu_event, &event_ready) == ML_STATUS_OK);
    CHECK(event_ready == 1);
    CHECK(ml_event_synchronize(cpu_event) == ML_STATUS_OK);
    float elapsed_ms = -1.0F;
    CHECK(ml_event_elapsed_ms(cpu_event, cpu_event, &elapsed_ms) ==
          ML_STATUS_INVALID_ARGUMENT);
    ml_event_destroy(cpu_event);
    CHECK(ml_stream_create(ML_DEVICE_CPU, 0, 1, &cpu_stream) == ML_STATUS_OK);
    uintptr_t native_stream = 1;
    int stream_owning = 0;
    CHECK(ml_stream_native_handle(cpu_stream, &native_stream) == ML_STATUS_OK);
    CHECK(native_stream == 0);
    CHECK(ml_stream_is_owning(cpu_stream, &stream_owning) == ML_STATUS_OK);
    CHECK(stream_owning == 1);
    CHECK(ml_multiply_out_on_stream(sum, left, right, cpu_stream) == ML_STATUS_OK);
    CHECK(ml_stream_synchronize(cpu_stream) == ML_STATUS_OK);
    CHECK(ml_tensor_copy_f32(sum, output, 4) == ML_STATUS_OK);
    CHECK(output[0] == 5 && output[1] == 12 && output[2] == 21 && output[3] == 32);
    ml_stream_destroy(cpu_stream);
    int hip_devices = 0;
    CHECK(ml_hip_device_count(&hip_devices) == ML_STATUS_OK);
    if (hip_devices > 0) {
        ml_tensor* hip_left = NULL;
        ml_tensor* hip_right = NULL;
        ml_tensor* hip_sum = NULL;
        ml_event* hip_start = NULL;
        ml_event* hip_finish = NULL;
        ml_stream* hip_stream = NULL;
        ml_stream* hip_external = NULL;
        CHECK(ml_tensor_to(left, ML_DEVICE_HIP, 0, &hip_left) == ML_STATUS_OK);
        CHECK(ml_tensor_to(right, ML_DEVICE_HIP, 0, &hip_right) == ML_STATUS_OK);
        CHECK(ml_event_create(ML_DEVICE_HIP, 0, 1, &hip_start) == ML_STATUS_OK);
        CHECK(ml_event_create(ML_DEVICE_HIP, 0, 1, &hip_finish) == ML_STATUS_OK);
        CHECK(ml_stream_create(ML_DEVICE_HIP, 0, 1, &hip_stream) == ML_STATUS_OK);
        CHECK(ml_stream_native_handle(hip_stream, &native_stream) == ML_STATUS_OK);
        CHECK(native_stream != 0);
        CHECK(ml_stream_from_external(ML_DEVICE_HIP, 0, native_stream,
                                      &hip_external) == ML_STATUS_OK);
        CHECK(ml_stream_is_owning(hip_external, &stream_owning) == ML_STATUS_OK);
        CHECK(stream_owning == 0);
        CHECK(ml_event_record(hip_start, hip_external) == ML_STATUS_OK);
        CHECK(ml_add_on_stream(hip_left, hip_right, hip_external, &hip_sum) == ML_STATUS_OK);
        CHECK(ml_event_record(hip_finish, hip_external) == ML_STATUS_OK);
        CHECK(ml_event_synchronize(hip_finish) == ML_STATUS_OK);
        CHECK(ml_event_elapsed_ms(hip_start, hip_finish, &elapsed_ms) == ML_STATUS_OK);
        CHECK(elapsed_ms >= 0.0F);
        CHECK(ml_tensor_copy_f32(hip_sum, output, 4) == ML_STATUS_OK);
        CHECK(output[0] == 6 && output[1] == 8 && output[2] == 10 && output[3] == 12);
        ml_tensor_destroy(hip_left);
        ml_tensor_destroy(hip_right);
        ml_tensor_destroy(hip_sum);
        ml_event_destroy(hip_start);
        ml_event_destroy(hip_finish);
        ml_stream_destroy(hip_external);
        CHECK(ml_stream_synchronize(hip_stream) == ML_STATUS_OK);
        ml_stream_destroy(hip_stream);
    }
    ml_tensor_destroy(left);
    ml_tensor_destroy(right);
    ml_tensor_destroy(sum);
    ml_tensor_destroy(product);
    return 0;
}
