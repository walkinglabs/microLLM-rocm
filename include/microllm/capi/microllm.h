#ifndef MICROLLM_CAPI_MICROLLM_H
#define MICROLLM_CAPI_MICROLLM_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define ML_CAPI_VERSION 1

typedef enum ml_status {
    ML_STATUS_OK = 0,
    ML_STATUS_INVALID_ARGUMENT = 1,
    ML_STATUS_OUT_OF_RANGE = 2,
    ML_STATUS_RUNTIME_ERROR = 3,
    ML_STATUS_UNKNOWN_ERROR = 255
} ml_status;

typedef enum ml_dtype {
    ML_DTYPE_FLOAT32 = 0,
    ML_DTYPE_INT32 = 1
} ml_dtype;

typedef enum ml_device_type {
    ML_DEVICE_CPU = 0,
    ML_DEVICE_HIP = 1
} ml_device_type;

typedef struct ml_tensor ml_tensor;
typedef struct ml_event ml_event;
typedef struct ml_stream ml_stream;

uint32_t ml_capi_version(void);
const char* ml_engine_version(void);
const char* ml_last_error(void);
ml_status ml_hip_device_count(int* count);

ml_status ml_tensor_from_f32(const float* values, const int64_t* shape, size_t rank,
                             ml_device_type device_type, int device_index,
                             ml_tensor** output);
ml_status ml_tensor_from_i32(const int32_t* values, const int64_t* shape, size_t rank,
                             ml_device_type device_type, int device_index,
                             ml_tensor** output);
void ml_tensor_destroy(ml_tensor* tensor);

ml_status ml_tensor_rank(const ml_tensor* tensor, size_t* rank);
ml_status ml_tensor_shape(const ml_tensor* tensor, size_t dim, int64_t* size);
ml_status ml_tensor_numel(const ml_tensor* tensor, int64_t* elements);
ml_status ml_tensor_dtype(const ml_tensor* tensor, ml_dtype* dtype);
ml_status ml_tensor_device(const ml_tensor* tensor, ml_device_type* device_type,
                           int* device_index);
ml_status ml_tensor_copy_f32(const ml_tensor* tensor, float* values, size_t capacity);
ml_status ml_tensor_copy_i32(const ml_tensor* tensor, int32_t* values, size_t capacity);
ml_status ml_tensor_to(const ml_tensor* tensor, ml_device_type device_type, int device_index,
                       ml_tensor** output);

ml_status ml_event_create(ml_device_type device_type, int device_index,
                          int enable_timing, ml_event** output);
void ml_event_destroy(ml_event* event);
ml_status ml_event_record_default_stream(ml_event* event);
ml_status ml_event_ready(const ml_event* event, int* ready);
ml_status ml_event_synchronize(const ml_event* event);
ml_status ml_event_elapsed_ms(const ml_event* start, const ml_event* finish,
                              float* milliseconds);
ml_status ml_stream_create(ml_device_type device_type, int device_index,
                           int non_blocking, ml_stream** output);
void ml_stream_destroy(ml_stream* stream);
ml_status ml_stream_synchronize(const ml_stream* stream);
ml_status ml_event_record(ml_event* event, ml_stream* stream);
ml_status ml_event_wait(const ml_event* event, ml_stream* stream);

ml_status ml_add(const ml_tensor* left, const ml_tensor* right, ml_tensor** output);
ml_status ml_multiply(const ml_tensor* left, const ml_tensor* right, ml_tensor** output);
ml_status ml_matmul(const ml_tensor* left, const ml_tensor* right, ml_tensor** output);
ml_status ml_softmax(const ml_tensor* input, ml_tensor** output);
ml_status ml_add_on_stream(const ml_tensor* left, const ml_tensor* right,
                           ml_stream* stream, ml_tensor** output);
ml_status ml_multiply_on_stream(const ml_tensor* left, const ml_tensor* right,
                                ml_stream* stream, ml_tensor** output);
ml_status ml_matmul_on_stream(const ml_tensor* left, const ml_tensor* right,
                              ml_stream* stream, ml_tensor** output);
ml_status ml_softmax_on_stream(const ml_tensor* input, ml_stream* stream,
                               ml_tensor** output);
ml_status ml_multiply_out_on_stream(ml_tensor* output, const ml_tensor* left,
                                    const ml_tensor* right, ml_stream* stream);
ml_status ml_matmul_out_on_stream(ml_tensor* output, const ml_tensor* left,
                                  const ml_tensor* right, ml_stream* stream);

#ifdef __cplusplus
}
#endif

#endif
