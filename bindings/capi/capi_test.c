#include <assert.h>
#include <math.h>
#include <stdint.h>
#include <string.h>

#include <microllm/capi/microllm.h>

int main(void) {
    assert(ml_capi_version() == ML_CAPI_VERSION);
    assert(strlen(ml_engine_version()) > 0);
    const int64_t shape[2] = {2, 2};
    const float left_values[4] = {1, 2, 3, 4};
    const float right_values[4] = {5, 6, 7, 8};
    ml_tensor* left = NULL;
    ml_tensor* right = NULL;
    ml_tensor* sum = NULL;
    ml_tensor* product = NULL;
    assert(ml_tensor_from_f32(left_values, shape, 2, ML_DEVICE_CPU, 0, &left) == ML_STATUS_OK);
    assert(ml_tensor_from_f32(right_values, shape, 2, ML_DEVICE_CPU, 0, &right) == ML_STATUS_OK);
    assert(ml_add(left, right, &sum) == ML_STATUS_OK);
    assert(ml_matmul(left, right, &product) == ML_STATUS_OK);
    float output[4] = {0};
    assert(ml_tensor_copy_f32(sum, output, 4) == ML_STATUS_OK);
    assert(output[0] == 6 && output[1] == 8 && output[2] == 10 && output[3] == 12);
    assert(ml_tensor_copy_f32(product, output, 4) == ML_STATUS_OK);
    assert(output[0] == 19 && output[1] == 22 && output[2] == 43 && output[3] == 50);
    ml_tensor* invalid = NULL;
    assert(ml_add(left, NULL, &invalid) == ML_STATUS_INVALID_ARGUMENT);
    assert(invalid == NULL);
    assert(strlen(ml_last_error()) > 0);
    int hip_devices = 0;
    assert(ml_hip_device_count(&hip_devices) == ML_STATUS_OK);
    if (hip_devices > 0) {
        ml_tensor* hip_left = NULL;
        ml_tensor* hip_right = NULL;
        ml_tensor* hip_sum = NULL;
        assert(ml_tensor_to(left, ML_DEVICE_HIP, 0, &hip_left) == ML_STATUS_OK);
        assert(ml_tensor_to(right, ML_DEVICE_HIP, 0, &hip_right) == ML_STATUS_OK);
        assert(ml_add(hip_left, hip_right, &hip_sum) == ML_STATUS_OK);
        assert(ml_tensor_copy_f32(hip_sum, output, 4) == ML_STATUS_OK);
        assert(output[0] == 6 && output[1] == 8 && output[2] == 10 && output[3] == 12);
        ml_tensor_destroy(hip_left);
        ml_tensor_destroy(hip_right);
        ml_tensor_destroy(hip_sum);
    }
    ml_tensor_destroy(left);
    ml_tensor_destroy(right);
    ml_tensor_destroy(sum);
    ml_tensor_destroy(product);
    return 0;
}
