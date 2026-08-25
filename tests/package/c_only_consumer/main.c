#include <stdint.h>
#include <stdio.h>

#include <microllm/capi/microllm.h>

int main(void) {
    const int64_t shape[1] = {2};
    const float left_values[2] = {1.0F, 2.0F};
    const float right_values[2] = {3.0F, 4.0F};
    float output_values[2] = {0.0F, 0.0F};
    ml_tensor* left = NULL;
    ml_tensor* right = NULL;
    ml_tensor* output = NULL;

    if (ml_capi_version() != ML_CAPI_VERSION ||
        ml_tensor_from_f32(left_values, shape, 1, ML_DEVICE_CPU, 0, &left) !=
            ML_STATUS_OK ||
        ml_tensor_from_f32(right_values, shape, 1, ML_DEVICE_CPU, 0, &right) !=
            ML_STATUS_OK ||
        ml_add(left, right, &output) != ML_STATUS_OK ||
        ml_tensor_copy_f32(output, output_values, 2) != ML_STATUS_OK) {
        fprintf(stderr, "microLLM C-only package consumer failed: %s\n", ml_last_error());
        ml_tensor_destroy(left);
        ml_tensor_destroy(right);
        ml_tensor_destroy(output);
        return 1;
    }

    ml_tensor_destroy(left);
    ml_tensor_destroy(right);
    ml_tensor_destroy(output);
    if (output_values[0] != 4.0F || output_values[1] != 6.0F) return 1;
    puts("microLLM C-only package consumer: pass");
    return 0;
}
