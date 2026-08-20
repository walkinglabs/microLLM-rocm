#include "kernels.h"

#include <stdexcept>
#include <string>

#include <hip/hip_runtime.h>
#include <hip/hip_bfloat16.h>
#include <hip/hip_fp16.h>
#include <hip/hip_fp8.h>

namespace microllm::ops::hip {
namespace {

constexpr unsigned kBlockSize = 256;

void check_launch(const char* operation) {
    const auto status = hipGetLastError();
    if (status != hipSuccess) {
        throw std::runtime_error(std::string(operation) + ": " + hipGetErrorString(status));
    }
}

hipStream_t as_stream(void* stream) { return reinterpret_cast<hipStream_t>(stream); }

unsigned grid_size(std::int64_t elements) {
    const auto block_size = static_cast<std::int64_t>(kBlockSize);
    return static_cast<unsigned>((elements + block_size - 1) / block_size);
}

__device__ float block_reduce_sum(float value, float* scratch);

__global__ void fill_kernel(float* output, std::int64_t elements, float value) {
    const auto index = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index < elements) output[index] = value;
}

__global__ void adamw_update_kernel(
    float* parameter, const float* gradient, float* first_moment,
    float* second_moment, std::int64_t elements, float learning_rate,
    float beta1, float beta2, float epsilon, float weight_decay,
    float first_correction, float second_correction) {
    const auto index = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= elements) return;
    const auto grad = gradient[index];
    const auto first = beta1 * first_moment[index] + (1.0F - beta1) * grad;
    const auto second = beta2 * second_moment[index] + (1.0F - beta2) * grad * grad;
    first_moment[index] = first;
    second_moment[index] = second;
    parameter[index] *= 1.0F - learning_rate * weight_decay;
    parameter[index] -= learning_rate * (first / first_correction) /
                        (sqrtf(second / second_correction) + epsilon);
}

__global__ void add_kernel(const float* left, const float* right, float* output,
                           std::int64_t elements) {
    const auto index = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index < elements) output[index] = left[index] + right[index];
}

__global__ void add_bias_kernel(const float* input, const float* bias, float* output,
                                std::int64_t elements, std::int64_t width) {
    const auto index = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index < elements) output[index] = input[index] + bias[index % width];
}

__global__ void bias_gradient_kernel(const float* gradient, float* output,
                                     std::int64_t rows, std::int64_t width) {
    const auto column = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (column >= width) return;
    float total = 0.0F;
    for (std::int64_t row = 0; row < rows; ++row) total += gradient[row * width + column];
    output[column] = total;
}

__global__ void multiply_kernel(const float* left, const float* right, float* output,
                                std::int64_t elements) {
    const auto index = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index < elements) output[index] = left[index] * right[index];
}

__global__ void scale_kernel(const float* input, float* output, std::int64_t elements,
                             float factor) {
    const auto index = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index < elements) output[index] = input[index] * factor;
}

template <typename T>
__device__ float load_as_float(const T& value) {
    return static_cast<float>(value);
}

template <typename T>
__device__ T store_from_float(float value) {
    return T(value);
}

__device__ float sigmoid_typed_device(float value) {
    if (value >= 0.0F) return 1.0F / (1.0F + expf(-value));
    const auto exponential = expf(value);
    return exponential / (1.0F + exponential);
}

template <typename T>
__global__ void fill_typed_kernel(T* output, std::int64_t elements, float value) {
    const auto index = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index < elements) output[index] = store_from_float<T>(value);
}

template <typename T>
__global__ void add_typed_kernel(const T* left, const T* right, T* output,
                                 std::int64_t elements) {
    const auto index = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index < elements) {
        output[index] = store_from_float<T>(load_as_float(left[index]) +
                                            load_as_float(right[index]));
    }
}

template <typename T>
__global__ void multiply_typed_kernel(const T* left, const T* right, T* output,
                                      std::int64_t elements) {
    const auto index = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index < elements) {
        output[index] = store_from_float<T>(load_as_float(left[index]) *
                                            load_as_float(right[index]));
    }
}

template <typename T>
__global__ void scale_typed_kernel(const T* input, T* output, std::int64_t elements,
                                   float factor) {
    const auto index = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index < elements) {
        output[index] = store_from_float<T>(load_as_float(input[index]) * factor);
    }
}

template <typename T>
__global__ void matmul_typed_kernel(const T* left, const T* right, T* output,
                                    std::int64_t batches, std::int64_t rows,
                                    std::int64_t inner, std::int64_t columns) {
    const auto index = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const auto elements = batches * rows * columns;
    if (index >= elements) return;
    const auto column = index % columns;
    const auto row_and_batch = index / columns;
    const auto row = row_and_batch % rows;
    const auto batch = row_and_batch / rows;
    const auto left_base = batch * rows * inner;
    const auto right_base = batch * inner * columns;
    float sum = 0.0F;
    for (std::int64_t reduction = 0; reduction < inner; ++reduction) {
        sum += load_as_float(left[left_base + row * inner + reduction]) *
               load_as_float(right[right_base + reduction * columns + column]);
    }
    output[index] = store_from_float<T>(sum);
}

template <typename T>
__global__ void matmul_transposed_typed_kernel(
    const T* left, const T* right, T* output,
    std::int64_t left_rows, std::int64_t left_columns,
    std::int64_t right_rows, std::int64_t right_columns,
    bool transpose_left, bool transpose_right) {
    const auto rows = transpose_left ? left_columns : left_rows;
    const auto inner = transpose_left ? left_rows : left_columns;
    const auto columns = transpose_right ? right_rows : right_columns;
    const auto index = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= rows * columns) return;
    const auto row = index / columns;
    const auto column = index % columns;
    float sum = 0.0F;
    for (std::int64_t reduction = 0; reduction < inner; ++reduction) {
        const auto left_index = transpose_left
                                    ? reduction * left_columns + row
                                    : row * left_columns + reduction;
        const auto right_index = transpose_right
                                     ? column * right_columns + reduction
                                     : reduction * right_columns + column;
        sum += load_as_float(left[left_index]) * load_as_float(right[right_index]);
    }
    output[index] = store_from_float<T>(sum);
}

template <typename T>
__global__ void silu_typed_kernel(const T* input, T* output, std::int64_t elements) {
    const auto index = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index < elements) {
        const auto value = load_as_float(input[index]);
        output[index] = store_from_float<T>(value * sigmoid_typed_device(value));
    }
}

template <typename T>
__global__ void swiglu_typed_kernel(const T* gate, const T* up, T* output,
                                    std::int64_t elements) {
    const auto index = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index < elements) {
        const auto gate_value = load_as_float(gate[index]);
        output[index] = store_from_float<T>(gate_value * sigmoid_typed_device(gate_value) *
                                            load_as_float(up[index]));
    }
}

template <typename Input, typename Output>
__global__ void quantize_fp8_kernel(const Input* input, Output* output,
                                    std::int64_t elements, float inverse_scale) {
    const auto index = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index < elements) output[index] = Output(load_as_float(input[index]) * inverse_scale);
}

template <typename Input, typename Output>
__global__ void dequantize_fp8_kernel(const Input* input, Output* output,
                                      std::int64_t elements, float scale) {
    const auto index = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index < elements) {
        output[index] = store_from_float<Output>(static_cast<float>(input[index]) * scale);
    }
}

template <typename Input, typename Output>
__global__ void cast_kernel(const Input* input, Output* output,
                            std::int64_t elements) {
    const auto index = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index < elements) {
        output[index] = store_from_float<Output>(load_as_float(input[index]));
    }
}

__global__ void matmul_kernel(const float* left, const float* right, float* output,
                              std::int64_t batches, std::int64_t rows,
                              std::int64_t inner, std::int64_t columns) {
    const auto index = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const auto elements = batches * rows * columns;
    if (index >= elements) return;
    const auto column = index % columns;
    const auto row_and_batch = index / columns;
    const auto row = row_and_batch % rows;
    const auto batch = row_and_batch / rows;
    const auto left_base = batch * rows * inner;
    const auto right_base = batch * inner * columns;
    float sum = 0.0F;
    for (std::int64_t reduction = 0; reduction < inner; ++reduction) {
        sum += left[left_base + row * inner + reduction] *
               right[right_base + reduction * columns + column];
    }
    output[index] = sum;
}

__global__ void embedding_kernel(const float* weight, const std::int32_t* indices,
                                 float* output, std::int64_t elements,
                                 std::int64_t vocabulary, std::int64_t width) {
    const auto index = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= elements) return;
    const auto token = index / width;
    const auto column = index % width;
    const auto vocabulary_index = static_cast<std::int64_t>(indices[token]);
    if (vocabulary_index < 0 || vocabulary_index >= vocabulary) {
        output[index] = nanf("");
        return;
    }
    output[index] = weight[vocabulary_index * width + column];
}

__global__ void softmax_kernel(const float* input, float* output, std::int64_t rows,
                               std::int64_t width) {
    const auto row = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (row >= rows) return;
    const auto base = row * width;
    float maximum = -INFINITY;
    for (std::int64_t column = 0; column < width; ++column) {
        maximum = fmaxf(maximum, input[base + column]);
    }
    float denominator = 0.0F;
    for (std::int64_t column = 0; column < width; ++column) {
        const auto value = expf(input[base + column] - maximum);
        output[base + column] = value;
        denominator += value;
    }
    for (std::int64_t column = 0; column < width; ++column) {
        output[base + column] /= denominator;
    }
}

__global__ void rms_norm_kernel(const float* input, const float* weight, float* output,
                                std::int64_t rows, std::int64_t width, float epsilon) {
    const auto row = static_cast<std::int64_t>(blockIdx.x);
    if (row >= rows) return;
    const auto base = row * width;
    float local_square_sum = 0.0F;
    for (std::int64_t column = threadIdx.x; column < width; column += blockDim.x) {
        const auto value = input[base + column];
        local_square_sum += value * value;
    }
    __shared__ float scratch[kBlockSize];
    const auto square_sum = block_reduce_sum(local_square_sum, scratch);
    const auto inverse_rms = rsqrtf(square_sum / static_cast<float>(width) + epsilon);
    for (std::int64_t column = threadIdx.x; column < width; column += blockDim.x) {
        output[base + column] = input[base + column] * inverse_rms * weight[column];
    }
}

template <unsigned BlockSize>
__global__ void add_rms_norm_kernel(const float* left, const float* right,
                                    const float* weight, float* sum,
                                    float* normalized, std::int64_t rows,
                                    std::int64_t width, float epsilon) {
    const auto row = static_cast<std::int64_t>(blockIdx.x);
    if (row >= rows) return;
    const auto base = row * width;
    float local_square_sum = 0.0F;
    for (std::int64_t column = threadIdx.x; column < width; column += blockDim.x) {
        const auto value = left[base + column] + right[base + column];
        sum[base + column] = value;
        local_square_sum += value * value;
    }
    __shared__ float scratch[BlockSize];
    scratch[threadIdx.x] = local_square_sum;
    __syncthreads();
    for (unsigned stride = BlockSize / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) scratch[threadIdx.x] += scratch[threadIdx.x + stride];
        __syncthreads();
    }
    const auto square_sum = scratch[0];
    const auto inverse_rms = rsqrtf(square_sum / static_cast<float>(width) + epsilon);
    for (std::int64_t column = threadIdx.x; column < width; column += blockDim.x) {
        normalized[base + column] = sum[base + column] * inverse_rms * weight[column];
    }
}

__device__ float sigmoid_device(float value) {
    if (value >= 0.0F) return 1.0F / (1.0F + expf(-value));
    const auto exponential = expf(value);
    return exponential / (1.0F + exponential);
}

__global__ void silu_kernel(const float* input, float* output, std::int64_t elements) {
    const auto index = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index < elements) output[index] = input[index] * sigmoid_device(input[index]);
}

__global__ void swiglu_kernel(const float* gate, const float* up, float* output,
                              std::int64_t elements) {
    const auto index = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index < elements) output[index] = gate[index] * sigmoid_device(gate[index]) * up[index];
}

__global__ void rope_kernel(const float* input, float* output, std::int64_t pairs,
                            std::int64_t head_width, std::int64_t sequence_size,
                            std::int64_t sequence_stride, std::int64_t position_offset,
                            float base) {
    const auto pair_index = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (pair_index >= pairs) return;
    const auto even = pair_index * 2;
    const auto position = (even / sequence_stride) % sequence_size;
    const auto pair_in_head = (even % head_width) / 2;
    const auto angle = static_cast<float>(position + position_offset) *
                       powf(base, -2.0F * static_cast<float>(pair_in_head) /
                                      static_cast<float>(head_width));
    float sine = 0.0F;
    float cosine = 0.0F;
    sincosf(angle, &sine, &cosine);
    output[even] = input[even] * cosine - input[even + 1] * sine;
    output[even + 1] = input[even] * sine + input[even + 1] * cosine;
}

__global__ void rope_split_half_kernel(
    const float* input, float* output, std::int64_t pairs,
    std::int64_t head_width, std::int64_t sequence_size,
    std::int64_t sequence_stride, std::int64_t position_offset, float base) {
    const auto pair_index = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (pair_index >= pairs) return;
    const auto half = head_width / 2;
    const auto head = pair_index / half;
    const auto pair = pair_index % half;
    const auto first = head * head_width + pair;
    const auto second = first + half;
    const auto position = (first / sequence_stride) % sequence_size;
    const auto angle = static_cast<float>(position + position_offset) *
                       powf(base, -2.0F * static_cast<float>(pair) /
                                      static_cast<float>(head_width));
    float sine = 0.0F;
    float cosine = 0.0F;
    sincosf(angle, &sine, &cosine);
    output[first] = input[first] * cosine - input[second] * sine;
    output[second] = input[first] * sine + input[second] * cosine;
}

__global__ void rope_split_half_bias_kernel(
    const float* input, const float* bias, float* output, std::int64_t pairs,
    std::int64_t heads, std::int64_t sequence_size, std::int64_t head_width,
    std::int64_t position_offset, float base) {
    const auto pair_index = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (pair_index >= pairs) return;
    const auto half = head_width / 2;
    const auto head_instance = pair_index / half;
    const auto pair = pair_index % half;
    const auto first = head_instance * head_width + pair;
    const auto second = first + half;
    const auto head = (first / (sequence_size * head_width)) % heads;
    const auto position = (first / head_width) % sequence_size;
    const auto angle = static_cast<float>(position + position_offset) *
                       powf(base, -2.0F * static_cast<float>(pair) /
                                      static_cast<float>(head_width));
    float sine = 0.0F;
    float cosine = 0.0F;
    sincosf(angle, &sine, &cosine);
    const auto bias_first = head * head_width + pair;
    const auto bias_second = bias_first + half;
    const auto first_value = input[first] + bias[bias_first];
    const auto second_value = input[second] + bias[bias_second];
    output[first] = first_value * cosine - second_value * sine;
    output[second] = first_value * sine + second_value * cosine;
}

__device__ float block_reduce_max(float value, float* scratch) {
    const auto lane = static_cast<unsigned>(threadIdx.x);
    scratch[lane] = value;
    __syncthreads();
    for (unsigned stride = kBlockSize / 2; stride > 0; stride >>= 1) {
        if (lane < stride) scratch[lane] = fmaxf(scratch[lane], scratch[lane + stride]);
        __syncthreads();
    }
    return scratch[0];
}

__device__ float block_reduce_sum(float value, float* scratch) {
    const auto lane = static_cast<unsigned>(threadIdx.x);
    scratch[lane] = value;
    __syncthreads();
    for (unsigned stride = kBlockSize / 2; stride > 0; stride >>= 1) {
        if (lane < stride) scratch[lane] += scratch[lane + stride];
        __syncthreads();
    }
    return scratch[0];
}

__device__ int block_reduce_sum_int(int value, int* scratch) {
    const auto lane = static_cast<unsigned>(threadIdx.x);
    scratch[lane] = value;
    __syncthreads();
    for (unsigned stride = kBlockSize / 2; stride > 0; stride >>= 1) {
        if (lane < stride) scratch[lane] += scratch[lane + stride];
        __syncthreads();
    }
    return scratch[0];
}

__device__ void block_reduce_argmax(float& best_value, std::int64_t& best_index,
                                    int& invalid, float* value_scratch,
                                    std::int64_t* index_scratch,
                                    int* invalid_scratch) {
    const auto lane = static_cast<unsigned>(threadIdx.x);
    value_scratch[lane] = best_value;
    index_scratch[lane] = best_index;
    invalid_scratch[lane] = invalid;
    __syncthreads();
    for (unsigned stride = kBlockSize / 2; stride > 0; stride >>= 1) {
        if (lane < stride) {
            const auto other_value = value_scratch[lane + stride];
            const auto other_index = index_scratch[lane + stride];
            if (other_value > value_scratch[lane] ||
                (other_value == value_scratch[lane] && other_index < index_scratch[lane])) {
                value_scratch[lane] = other_value;
                index_scratch[lane] = other_index;
            }
            invalid_scratch[lane] += invalid_scratch[lane + stride];
        }
        __syncthreads();
    }
    best_value = value_scratch[0];
    best_index = index_scratch[0];
    invalid = invalid_scratch[0];
}

__global__ void argmax_kernel(const float* input, std::int32_t* output,
                              std::int64_t elements) {
    __shared__ float value_scratch[kBlockSize];
    __shared__ std::int64_t index_scratch[kBlockSize];
    __shared__ int invalid_scratch[kBlockSize];
    float best_value = -INFINITY;
    std::int64_t best_index = 0;
    int invalid = 0;
    for (std::int64_t index = threadIdx.x; index < elements; index += blockDim.x) {
        const auto value = input[index];
        if (!isfinite(value)) {
            invalid = 1;
        } else if (value > best_value || (value == best_value && index < best_index)) {
            best_value = value;
            best_index = index;
        }
    }
    block_reduce_argmax(best_value, best_index, invalid, value_scratch,
                        index_scratch, invalid_scratch);
    if (threadIdx.x == 0) {
        output[0] = invalid != 0 ? -1 : static_cast<std::int32_t>(best_index);
    }
}

__global__ void argmax_partial_kernel(const float* input, float* partials,
                                      std::int64_t elements) {
    __shared__ float value_scratch[kBlockSize];
    __shared__ std::int64_t index_scratch[kBlockSize];
    __shared__ int invalid_scratch[kBlockSize];
    const auto start = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const auto stride = static_cast<std::int64_t>(gridDim.x) * blockDim.x;
    float best_value = -INFINITY;
    std::int64_t best_index = 0;
    int invalid = 0;
    for (auto index = start; index < elements; index += stride) {
        const auto value = input[index];
        if (!isfinite(value)) {
            invalid = 1;
        } else if (value > best_value || (value == best_value && index < best_index)) {
            best_value = value;
            best_index = index;
        }
    }
    block_reduce_argmax(best_value, best_index, invalid, value_scratch,
                        index_scratch, invalid_scratch);
    if (threadIdx.x == 0) {
        partials[blockIdx.x * 3] = best_value;
        partials[blockIdx.x * 3 + 1] = static_cast<float>(best_index);
        partials[blockIdx.x * 3 + 2] = static_cast<float>(invalid);
    }
}

__global__ void argmax_finalize_kernel(const float* partials, std::int32_t* output,
                                       std::int64_t blocks) {
    __shared__ float value_scratch[kBlockSize];
    __shared__ std::int64_t index_scratch[kBlockSize];
    __shared__ int invalid_scratch[kBlockSize];
    float best_value = -INFINITY;
    std::int64_t best_index = 0;
    int invalid = 0;
    for (std::int64_t block = threadIdx.x; block < blocks; block += blockDim.x) {
        const auto value = partials[block * 3];
        const auto index = static_cast<std::int64_t>(partials[block * 3 + 1]);
        invalid += static_cast<int>(partials[block * 3 + 2]);
        if (value > best_value || (value == best_value && index < best_index)) {
            best_value = value;
            best_index = index;
        }
    }
    block_reduce_argmax(best_value, best_index, invalid, value_scratch,
                        index_scratch, invalid_scratch);
    if (threadIdx.x == 0) {
        output[0] = invalid != 0 ? -1 : static_cast<std::int32_t>(best_index);
    }
}

__global__ void cross_entropy_rows_kernel(const float* logits,
                                          const std::int32_t* targets,
                                          float* row_data, std::int64_t rows,
                                          std::int64_t classes) {
    const auto row = static_cast<std::int64_t>(blockIdx.x);
    if (row >= rows) return;
    const auto label = static_cast<std::int64_t>(targets[row]);
    if (label == -100) {
        if (threadIdx.x == 0) {
            row_data[row * 2] = 0.0F;
            row_data[row * 2 + 1] = 0.0F;
        }
        return;
    }
    if (label < 0 || label >= classes) {
        if (threadIdx.x == 0) {
            row_data[row * 2] = nanf("");
            row_data[row * 2 + 1] = -1.0F;
        }
        return;
    }
    __shared__ float scratch[kBlockSize];
    const auto base = row * classes;
    float local_maximum = -INFINITY;
    for (std::int64_t column = threadIdx.x; column < classes; column += blockDim.x) {
        local_maximum = fmaxf(local_maximum, logits[base + column]);
    }
    const auto maximum = block_reduce_max(local_maximum, scratch);
    float local_sum = 0.0F;
    for (std::int64_t column = threadIdx.x; column < classes; column += blockDim.x) {
        local_sum += expf(logits[base + column] - maximum);
    }
    const auto denominator = block_reduce_sum(local_sum, scratch);
    if (threadIdx.x == 0) {
        row_data[row * 2] = logf(denominator) + maximum - logits[base + label];
        row_data[row * 2 + 1] = 1.0F;
    }
}

__global__ void cross_entropy_finalize_kernel(const float* row_data, float* output,
                                              std::int64_t rows) {
    __shared__ float sum_scratch[kBlockSize];
    __shared__ int count_scratch[kBlockSize];
    float local_sum = 0.0F;
    int local_count = 0;
    int local_invalid = 0;
    for (std::int64_t row = threadIdx.x; row < rows; row += blockDim.x) {
        const auto valid = row_data[row * 2 + 1];
        if (valid < 0.0F) local_invalid = 1;
        if (valid > 0.0F) {
            local_sum += row_data[row * 2];
            ++local_count;
        }
    }
    const auto total = block_reduce_sum(local_sum, sum_scratch);
    const auto valid_rows = block_reduce_sum_int(local_count, count_scratch);
    const auto invalid_rows = block_reduce_sum_int(local_invalid, count_scratch);
    if (threadIdx.x == 0) {
        output[0] = invalid_rows != 0 || valid_rows == 0
                        ? nanf("")
                        : total / static_cast<float>(valid_rows);
    }
}

__global__ void reduce_sum_kernel(const float* input, float* output,
                                  std::int64_t elements) {
    if (blockIdx.x != 0 || threadIdx.x != 0) return;
    double total = 0.0;
    for (std::int64_t index = 0; index < elements; ++index) total += input[index];
    output[0] = static_cast<float>(total);
}

__global__ void broadcast_scalar_kernel(const float* scalar, float* output,
                                        std::int64_t elements) {
    const auto index = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index < elements) output[index] = scalar[0];
}

__global__ void embedding_backward_kernel(const float* gradient,
                                          const std::int32_t* indices,
                                          float* weight_gradient,
                                          std::int64_t elements,
                                          std::int64_t vocabulary,
                                          std::int64_t width) {
    const auto index = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= elements) return;
    const auto token = index / width;
    const auto column = index % width;
    const auto vocabulary_index = static_cast<std::int64_t>(indices[token]);
    if (vocabulary_index >= 0 && vocabulary_index < vocabulary) {
        atomicAdd(weight_gradient + vocabulary_index * width + column, gradient[index]);
    }
}

__global__ void softmax_backward_kernel(const float* output, const float* gradient,
                                        float* input_gradient, std::int64_t rows,
                                        std::int64_t width) {
    const auto row = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (row >= rows) return;
    const auto base = row * width;
    float dot = 0.0F;
    for (std::int64_t column = 0; column < width; ++column) {
        dot += gradient[base + column] * output[base + column];
    }
    for (std::int64_t column = 0; column < width; ++column) {
        input_gradient[base + column] =
            output[base + column] * (gradient[base + column] - dot);
    }
}

__global__ void rms_norm_backward_kernel(const float* input, const float* weight,
                                         const float* gradient, float* input_gradient,
                                         float* row_inverse_rms, std::int64_t rows,
                                         std::int64_t width, float epsilon) {
    const auto row = static_cast<std::int64_t>(blockIdx.x);
    if (row >= rows) return;
    const auto base = row * width;
    float local_square_sum = 0.0F;
    float local_weighted_dot = 0.0F;
    for (std::int64_t column = threadIdx.x; column < width; column += blockDim.x) {
        const auto index = base + column;
        local_square_sum += input[index] * input[index];
        local_weighted_dot += gradient[index] * weight[column] * input[index];
    }
    __shared__ float square_scratch[kBlockSize];
    __shared__ float dot_scratch[kBlockSize];
    const auto square_sum = block_reduce_sum(local_square_sum, square_scratch);
    const auto weighted_dot = block_reduce_sum(local_weighted_dot, dot_scratch);
    const auto inverse_rms = rsqrtf(square_sum / static_cast<float>(width) + epsilon);
    const auto correction = inverse_rms * inverse_rms * inverse_rms * weighted_dot /
                            static_cast<float>(width);
    if (threadIdx.x == 0) row_inverse_rms[row] = inverse_rms;
    for (std::int64_t column = threadIdx.x; column < width; column += blockDim.x) {
        const auto index = base + column;
        input_gradient[index] =
            gradient[index] * weight[column] * inverse_rms - input[index] * correction;
    }
}

__global__ void rms_norm_weight_gradient_kernel(
    const float* input, const float* gradient, const float* row_inverse_rms,
    float* weight_gradient, std::int64_t rows, std::int64_t width) {
    const auto column = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (column >= width) return;
    float total = 0.0F;
    for (std::int64_t row = 0; row < rows; ++row) {
        const auto index = row * width + column;
        total += gradient[index] * input[index] * row_inverse_rms[row];
    }
    weight_gradient[column] = total;
}

__global__ void kv_cache_store_kernel(const float* current, float* cache,
                                      std::int64_t heads, std::int64_t capacity,
                                      std::int64_t width, std::int64_t position) {
    const auto index = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= heads * width) return;
    const auto head = index / width;
    const auto column = index % width;
    cache[head * capacity * width + position * width + column] = current[index];
}

__global__ void cached_attention_scores_kernel(
    const float* query, const float* key_cache, float* scores,
    std::int64_t heads, std::int64_t sequence,
    std::int64_t cache_head_stride, std::int64_t width,
    std::int64_t repeats, float scale) {
    const auto index = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= heads * sequence) return;
    const auto head = index / sequence;
    const auto position = index % sequence;
    const auto kv_head = head / repeats;
    float dot = 0.0F;
    for (std::int64_t column = 0; column < width; ++column) {
        dot += query[head * width + column] *
               key_cache[kv_head * cache_head_stride + position * width + column];
    }
    scores[index] = dot * scale;
}

__global__ void cached_attention_context_kernel(
    const float* probabilities, const float* value_cache, float* output,
    std::int64_t heads, std::int64_t sequence,
    std::int64_t cache_head_stride, std::int64_t width,
    std::int64_t repeats) {
    const auto index = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= heads * width) return;
    const auto head = index / width;
    const auto column = index % width;
    const auto kv_head = head / repeats;
    float total = 0.0F;
    for (std::int64_t position = 0; position < sequence; ++position) {
        total += probabilities[head * sequence + position] *
                 value_cache[kv_head * cache_head_stride + position * width + column];
    }
    output[index] = total;
}

__global__ void cached_attention_fused_kernel(
    const float* query, const float* key_cache, const float* value_cache,
    float* output, std::int64_t heads, std::int64_t sequence,
    std::int64_t cache_head_stride, std::int64_t width,
    std::int64_t repeats, float scale) {
    const auto head = static_cast<std::int64_t>(blockIdx.x);
    if (head >= heads) return;
    const auto kv_head = head / repeats;
    extern __shared__ float shared_scores[];
    __shared__ float reduction[kBlockSize];

    float local_maximum = -INFINITY;
    for (std::int64_t position = threadIdx.x; position < sequence;
         position += blockDim.x) {
        float dot = 0.0F;
        for (std::int64_t column = 0; column < width; ++column) {
            dot += query[head * width + column] *
                   key_cache[kv_head * cache_head_stride + position * width + column];
        }
        const auto score = dot * scale;
        shared_scores[position] = score;
        local_maximum = fmaxf(local_maximum, score);
    }
    const auto maximum = block_reduce_max(local_maximum, reduction);
    float local_denominator = 0.0F;
    for (std::int64_t position = threadIdx.x; position < sequence;
         position += blockDim.x) {
        const auto probability = expf(shared_scores[position] - maximum);
        shared_scores[position] = probability;
        local_denominator += probability;
    }
    const auto denominator = block_reduce_sum(local_denominator, reduction);
    for (std::int64_t column = threadIdx.x; column < width; column += blockDim.x) {
        float total = 0.0F;
        for (std::int64_t position = 0; position < sequence; ++position) {
            total += shared_scores[position] / denominator *
                     value_cache[kv_head * cache_head_stride + position * width + column];
        }
        output[head * width + column] = total;
    }
}

__global__ void silu_backward_kernel(const float* input, const float* gradient,
                                     float* input_gradient, std::int64_t elements) {
    const auto index = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= elements) return;
    const auto probability = sigmoid_device(input[index]);
    input_gradient[index] =
        gradient[index] * probability * (1.0F + input[index] * (1.0F - probability));
}

__global__ void swiglu_backward_kernel(const float* gate, const float* up,
                                       const float* gradient, float* gate_gradient,
                                       float* up_gradient, std::int64_t elements) {
    const auto index = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= elements) return;
    const auto probability = sigmoid_device(gate[index]);
    const auto silu_value = gate[index] * probability;
    gate_gradient[index] = gradient[index] * up[index] * probability *
                           (1.0F + gate[index] * (1.0F - probability));
    up_gradient[index] = gradient[index] * silu_value;
}

__global__ void rope_backward_kernel(const float* gradient, float* input_gradient,
                                     std::int64_t pairs, std::int64_t head_width,
                                     std::int64_t sequence_size,
                                     std::int64_t sequence_stride,
                                     std::int64_t position_offset, float base) {
    const auto pair_index = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (pair_index >= pairs) return;
    const auto even = pair_index * 2;
    const auto position = (even / sequence_stride) % sequence_size;
    const auto pair_in_head = (even % head_width) / 2;
    const auto angle = static_cast<float>(position + position_offset) *
                       powf(base, -2.0F * static_cast<float>(pair_in_head) /
                                      static_cast<float>(head_width));
    float sine = 0.0F;
    float cosine = 0.0F;
    sincosf(angle, &sine, &cosine);
    input_gradient[even] = gradient[even] * cosine + gradient[even + 1] * sine;
    input_gradient[even + 1] = -gradient[even] * sine + gradient[even + 1] * cosine;
}

__global__ void rope_split_half_backward_kernel(
    const float* gradient, float* input_gradient, std::int64_t pairs,
    std::int64_t head_width, std::int64_t sequence_size,
    std::int64_t sequence_stride, std::int64_t position_offset, float base) {
    const auto pair_index = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (pair_index >= pairs) return;
    const auto half = head_width / 2;
    const auto head = pair_index / half;
    const auto pair = pair_index % half;
    const auto first = head * head_width + pair;
    const auto second = first + half;
    const auto position = (first / sequence_stride) % sequence_size;
    const auto angle = static_cast<float>(position + position_offset) *
                       powf(base, -2.0F * static_cast<float>(pair) /
                                      static_cast<float>(head_width));
    float sine = 0.0F;
    float cosine = 0.0F;
    sincosf(angle, &sine, &cosine);
    input_gradient[first] = gradient[first] * cosine + gradient[second] * sine;
    input_gradient[second] = -gradient[first] * sine + gradient[second] * cosine;
}

__global__ void cross_entropy_backward_stats_kernel(
    const float* logits, const std::int32_t* targets, float* row_stats,
    std::int64_t rows, std::int64_t classes) {
    const auto row = static_cast<std::int64_t>(blockIdx.x);
    if (row >= rows) return;
    const auto label = static_cast<std::int64_t>(targets[row]);
    if (label == -100) {
        if (threadIdx.x == 0) {
            row_stats[row * 2] = 0.0F;
            row_stats[row * 2 + 1] = 0.0F;
        }
        return;
    }
    if (label < 0 || label >= classes) {
        if (threadIdx.x == 0) {
            row_stats[row * 2] = nanf("");
            row_stats[row * 2 + 1] = nanf("");
        }
        return;
    }
    __shared__ float scratch[kBlockSize];
    const auto base = row * classes;
    float local_maximum = -INFINITY;
    for (std::int64_t column = threadIdx.x; column < classes; column += blockDim.x) {
        local_maximum = fmaxf(local_maximum, logits[base + column]);
    }
    const auto maximum = block_reduce_max(local_maximum, scratch);
    float local_sum = 0.0F;
    for (std::int64_t column = threadIdx.x; column < classes; column += blockDim.x) {
        local_sum += expf(logits[base + column] - maximum);
    }
    const auto denominator = block_reduce_sum(local_sum, scratch);
    if (threadIdx.x == 0) {
        row_stats[row * 2] = maximum;
        row_stats[row * 2 + 1] = 1.0F / denominator;
    }
}

__global__ void cross_entropy_backward_factor_kernel(
    const std::int32_t* targets, const float* loss_gradient, float* factor,
    std::int64_t rows, std::int64_t classes) {
    __shared__ int count_scratch[kBlockSize];
    int local_count = 0;
    int local_invalid = 0;
    for (std::int64_t row = threadIdx.x; row < rows; row += blockDim.x) {
        const auto label = static_cast<std::int64_t>(targets[row]);
        if (label != -100) {
            ++local_count;
            if (label < 0 || label >= classes) local_invalid = 1;
        }
    }
    const auto valid_rows = block_reduce_sum_int(local_count, count_scratch);
    const auto invalid_rows = block_reduce_sum_int(local_invalid, count_scratch);
    if (threadIdx.x == 0) {
        factor[0] = invalid_rows != 0 || valid_rows == 0
                        ? nanf("")
                        : loss_gradient[0] / static_cast<float>(valid_rows);
    }
}

__global__ void cross_entropy_backward_elements_kernel(
    const float* logits, const std::int32_t* targets, const float* row_stats,
    const float* factor, float* logits_gradient, std::int64_t elements,
    std::int64_t classes) {
    const auto index = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= elements) return;
    const auto row = index / classes;
    const auto column = index % classes;
    const auto label = static_cast<std::int64_t>(targets[row]);
    if (label == -100) {
        logits_gradient[index] = 0.0F;
        return;
    }
    const auto maximum = row_stats[row * 2];
    const auto inverse_sum = row_stats[row * 2 + 1];
    auto probability = expf(logits[index] - maximum) * inverse_sum;
    if (column == label) probability -= 1.0F;
    logits_gradient[index] = probability * factor[0];
}

__global__ void causal_softmax_kernel(const float* scores, float* output,
                                      std::int64_t rows, std::int64_t sequence) {
    const auto row = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (row >= rows) return;
    const auto visible = row % sequence;
    const auto base = row * sequence;
    float maximum = -INFINITY;
    for (std::int64_t column = 0; column <= visible; ++column) {
        maximum = fmaxf(maximum, scores[base + column]);
    }
    float denominator = 0.0F;
    for (std::int64_t column = 0; column <= visible; ++column) {
        output[base + column] = expf(scores[base + column] - maximum);
        denominator += output[base + column];
    }
    for (std::int64_t column = 0; column <= visible; ++column) {
        output[base + column] /= denominator;
    }
    for (std::int64_t column = visible + 1; column < sequence; ++column) {
        output[base + column] = 0.0F;
    }
}

__global__ void causal_softmax_backward_kernel(const float* output,
                                               const float* gradient,
                                               float* input_gradient,
                                               std::int64_t rows,
                                               std::int64_t sequence) {
    const auto row = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (row >= rows) return;
    const auto visible = row % sequence;
    const auto base = row * sequence;
    float dot = 0.0F;
    for (std::int64_t column = 0; column <= visible; ++column) {
        dot += gradient[base + column] * output[base + column];
    }
    for (std::int64_t column = 0; column <= visible; ++column) {
        input_gradient[base + column] =
            output[base + column] * (gradient[base + column] - dot);
    }
    for (std::int64_t column = visible + 1; column < sequence; ++column) {
        input_gradient[base + column] = 0.0F;
    }
}

__global__ void repeat_interleave_kernel(const float* input, float* output,
                                         std::int64_t output_elements,
                                         std::int64_t repeated_width,
                                         std::int64_t inner,
                                         std::int64_t repeats) {
    const auto index = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= output_elements) return;
    const auto inner_index = index % inner;
    const auto repeated_coordinate = (index / inner) % repeated_width;
    const auto outer = index / (inner * repeated_width);
    const auto input_width = repeated_width / repeats;
    const auto input_index = outer * input_width * inner +
                             (repeated_coordinate / repeats) * inner + inner_index;
    output[index] = input[input_index];
}

__global__ void repeat_interleave_backward_kernel(const float* gradient,
                                                  float* input_gradient,
                                                  std::int64_t input_elements,
                                                  std::int64_t input_width,
                                                  std::int64_t inner,
                                                  std::int64_t repeats) {
    const auto index = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= input_elements) return;
    const auto inner_index = index % inner;
    const auto input_coordinate = (index / inner) % input_width;
    const auto outer = index / (inner * input_width);
    const auto repeated_width = input_width * repeats;
    float total = 0.0F;
    for (std::int64_t repeat = 0; repeat < repeats; ++repeat) {
        const auto output_coordinate = input_coordinate * repeats + repeat;
        const auto output_index = outer * repeated_width * inner +
                                  output_coordinate * inner + inner_index;
        total += gradient[output_index];
    }
    input_gradient[index] = total;
}

}  // namespace

void launch_fill(float* output, std::int64_t elements, float value, void* stream) {
    if (elements == 0) return;
    hipLaunchKernelGGL(fill_kernel, dim3(grid_size(elements)), dim3(kBlockSize), 0,
                       as_stream(stream), output, elements, value);
    check_launch("fill_kernel");
}

void launch_adamw_update(float* parameter, const float* gradient,
                         float* first_moment, float* second_moment,
                         std::int64_t elements, float learning_rate,
                         float beta1, float beta2, float epsilon,
                         float weight_decay, float first_correction,
                         float second_correction, void* stream) {
    if (elements == 0) return;
    hipLaunchKernelGGL(adamw_update_kernel, dim3(grid_size(elements)),
                       dim3(kBlockSize), 0, as_stream(stream), parameter, gradient,
                       first_moment, second_moment, elements, learning_rate, beta1,
                       beta2, epsilon, weight_decay, first_correction,
                       second_correction);
    check_launch("adamw_update_kernel");
}

void launch_add(const float* left, const float* right, float* output,
                std::int64_t elements, void* stream) {
    if (elements == 0) return;
    hipLaunchKernelGGL(add_kernel, dim3(grid_size(elements)), dim3(kBlockSize), 0,
                       as_stream(stream), left, right, output, elements);
    check_launch("add_kernel");
}

void launch_add_bias(const float* input, const float* bias, float* output,
                     std::int64_t elements, std::int64_t width, void* stream) {
    if (elements == 0) return;
    hipLaunchKernelGGL(add_bias_kernel, dim3(grid_size(elements)), dim3(kBlockSize), 0,
                       as_stream(stream), input, bias, output, elements, width);
    check_launch("add_bias_kernel");
}

void launch_bias_gradient(const float* gradient, float* output,
                          std::int64_t rows, std::int64_t width, void* stream) {
    if (width == 0) return;
    hipLaunchKernelGGL(bias_gradient_kernel, dim3(grid_size(width)), dim3(kBlockSize), 0,
                       as_stream(stream), gradient, output, rows, width);
    check_launch("bias_gradient_kernel");
}

void launch_multiply(const float* left, const float* right, float* output,
                     std::int64_t elements, void* stream) {
    if (elements == 0) return;
    hipLaunchKernelGGL(multiply_kernel, dim3(grid_size(elements)), dim3(kBlockSize), 0,
                       as_stream(stream), left, right, output, elements);
    check_launch("multiply_kernel");
}

void launch_scale(const float* input, float* output, std::int64_t elements, float factor,
                  void* stream) {
    if (elements == 0) return;
    hipLaunchKernelGGL(scale_kernel, dim3(grid_size(elements)), dim3(kBlockSize), 0,
                       as_stream(stream), input, output, elements, factor);
    check_launch("scale_kernel");
}

void launch_fill_typed(void* output, DType dtype, std::int64_t elements, float value,
                       void* stream) {
    if (elements == 0) return;
    const auto grid = dim3(grid_size(elements));
    const auto block = dim3(kBlockSize);
    switch (dtype) {
        case DType::Float32:
            hipLaunchKernelGGL(fill_typed_kernel<float>, grid, block, 0, as_stream(stream),
                               static_cast<float*>(output), elements, value);
            break;
        case DType::Float16:
            hipLaunchKernelGGL(fill_typed_kernel<__half>, grid, block, 0, as_stream(stream),
                               static_cast<__half*>(output), elements, value);
            break;
        case DType::BFloat16:
            hipLaunchKernelGGL(fill_typed_kernel<hip_bfloat16>, grid, block, 0,
                               as_stream(stream), static_cast<hip_bfloat16*>(output),
                               elements, value);
            break;
        default: throw std::invalid_argument("typed fill requires a floating dtype");
    }
    check_launch("fill_typed_kernel");
}

#define MICROLLM_LAUNCH_BINARY_TYPED(kernel_name, operation_name)                       \
    switch (dtype) {                                                                    \
        case DType::Float32:                                                            \
            hipLaunchKernelGGL(kernel_name<float>, grid, block, 0, as_stream(stream),  \
                               static_cast<const float*>(left),                         \
                               static_cast<const float*>(right),                        \
                               static_cast<float*>(output), elements);                  \
            break;                                                                      \
        case DType::Float16:                                                            \
            hipLaunchKernelGGL(kernel_name<__half>, grid, block, 0, as_stream(stream), \
                               static_cast<const __half*>(left),                        \
                               static_cast<const __half*>(right),                       \
                               static_cast<__half*>(output), elements);                 \
            break;                                                                      \
        case DType::BFloat16:                                                           \
            hipLaunchKernelGGL(kernel_name<hip_bfloat16>, grid, block, 0,              \
                               as_stream(stream),                                       \
                               static_cast<const hip_bfloat16*>(left),                  \
                               static_cast<const hip_bfloat16*>(right),                 \
                               static_cast<hip_bfloat16*>(output), elements);           \
            break;                                                                      \
        default: throw std::invalid_argument(operation_name " requires a floating dtype"); \
    }

void launch_add_typed(const void* left, const void* right, void* output, DType dtype,
                      std::int64_t elements, void* stream) {
    if (elements == 0) return;
    const auto grid = dim3(grid_size(elements));
    const auto block = dim3(kBlockSize);
    MICROLLM_LAUNCH_BINARY_TYPED(add_typed_kernel, "typed add")
    check_launch("add_typed_kernel");
}

void launch_multiply_typed(const void* left, const void* right, void* output, DType dtype,
                           std::int64_t elements, void* stream) {
    if (elements == 0) return;
    const auto grid = dim3(grid_size(elements));
    const auto block = dim3(kBlockSize);
    MICROLLM_LAUNCH_BINARY_TYPED(multiply_typed_kernel, "typed multiply")
    check_launch("multiply_typed_kernel");
}

#undef MICROLLM_LAUNCH_BINARY_TYPED

void launch_scale_typed(const void* input, void* output, DType dtype,
                        std::int64_t elements, float factor, void* stream) {
    if (elements == 0) return;
    const auto grid = dim3(grid_size(elements));
    const auto block = dim3(kBlockSize);
#define MICROLLM_SCALE_CASE(cpp_type, enum_value)                                      \
    case enum_value:                                                                   \
        hipLaunchKernelGGL(scale_typed_kernel<cpp_type>, grid, block, 0,              \
                           as_stream(stream), static_cast<const cpp_type*>(input),     \
                           static_cast<cpp_type*>(output), elements, factor);          \
        break
    switch (dtype) {
        MICROLLM_SCALE_CASE(float, DType::Float32);
        MICROLLM_SCALE_CASE(__half, DType::Float16);
        MICROLLM_SCALE_CASE(hip_bfloat16, DType::BFloat16);
        default: throw std::invalid_argument("typed scale requires a floating dtype");
    }
#undef MICROLLM_SCALE_CASE
    check_launch("scale_typed_kernel");
}

void launch_matmul_typed(const void* left, const void* right, void* output, DType dtype,
                         std::int64_t batches, std::int64_t rows, std::int64_t inner,
                         std::int64_t columns, void* stream) {
    const auto elements = batches * rows * columns;
    if (elements == 0) return;
    const auto grid = dim3(grid_size(elements));
    const auto block = dim3(kBlockSize);
#define MICROLLM_MATMUL_CASE(cpp_type, enum_value)                                     \
    case enum_value:                                                                   \
        hipLaunchKernelGGL(matmul_typed_kernel<cpp_type>, grid, block, 0,             \
                           as_stream(stream), static_cast<const cpp_type*>(left),      \
                           static_cast<const cpp_type*>(right),                        \
                           static_cast<cpp_type*>(output), batches, rows, inner,       \
                           columns);                                                   \
        break
    switch (dtype) {
        MICROLLM_MATMUL_CASE(float, DType::Float32);
        MICROLLM_MATMUL_CASE(__half, DType::Float16);
        MICROLLM_MATMUL_CASE(hip_bfloat16, DType::BFloat16);
        default: throw std::invalid_argument("typed matmul requires a floating dtype");
    }
#undef MICROLLM_MATMUL_CASE
    check_launch("matmul_typed_kernel");
}

void launch_matmul_transposed_typed(
    const void* left, const void* right, void* output, DType dtype,
    std::int64_t left_rows, std::int64_t left_columns,
    std::int64_t right_rows, std::int64_t right_columns,
    bool transpose_left, bool transpose_right, void* stream) {
    const auto rows = transpose_left ? left_columns : left_rows;
    const auto columns = transpose_right ? right_rows : right_columns;
    const auto elements = rows * columns;
    if (elements == 0) return;
    const auto grid = dim3(grid_size(elements));
    const auto block = dim3(kBlockSize);
#define MICROLLM_TRANSPOSED_MATMUL_CASE(cpp_type, enum_value)                         \
    case enum_value:                                                                  \
        hipLaunchKernelGGL(matmul_transposed_typed_kernel<cpp_type>, grid, block, 0, \
                           as_stream(stream), static_cast<const cpp_type*>(left),     \
                           static_cast<const cpp_type*>(right),                       \
                           static_cast<cpp_type*>(output), left_rows, left_columns,   \
                           right_rows, right_columns, transpose_left, transpose_right); \
        break
    switch (dtype) {
        MICROLLM_TRANSPOSED_MATMUL_CASE(float, DType::Float32);
        MICROLLM_TRANSPOSED_MATMUL_CASE(__half, DType::Float16);
        MICROLLM_TRANSPOSED_MATMUL_CASE(hip_bfloat16, DType::BFloat16);
        default: throw std::invalid_argument("transpose-aware matmul requires a floating dtype");
    }
#undef MICROLLM_TRANSPOSED_MATMUL_CASE
    check_launch("matmul_transposed_typed_kernel");
}

void launch_silu_typed(const void* input, void* output, DType dtype,
                       std::int64_t elements, void* stream) {
    if (elements == 0) return;
    const auto grid = dim3(grid_size(elements));
    const auto block = dim3(kBlockSize);
#define MICROLLM_SILU_CASE(cpp_type, enum_value)                                      \
    case enum_value:                                                                  \
        hipLaunchKernelGGL(silu_typed_kernel<cpp_type>, grid, block, 0,              \
                           as_stream(stream), static_cast<const cpp_type*>(input),    \
                           static_cast<cpp_type*>(output), elements);                 \
        break
    switch (dtype) {
        MICROLLM_SILU_CASE(float, DType::Float32);
        MICROLLM_SILU_CASE(__half, DType::Float16);
        MICROLLM_SILU_CASE(hip_bfloat16, DType::BFloat16);
        default: throw std::invalid_argument("typed silu requires a floating dtype");
    }
#undef MICROLLM_SILU_CASE
    check_launch("silu_typed_kernel");
}

void launch_swiglu_typed(const void* gate, const void* up, void* output, DType dtype,
                         std::int64_t elements, void* stream) {
    if (elements == 0) return;
    const auto grid = dim3(grid_size(elements));
    const auto block = dim3(kBlockSize);
#define MICROLLM_SWIGLU_CASE(cpp_type, enum_value)                                    \
    case enum_value:                                                                  \
        hipLaunchKernelGGL(swiglu_typed_kernel<cpp_type>, grid, block, 0,            \
                           as_stream(stream), static_cast<const cpp_type*>(gate),     \
                           static_cast<const cpp_type*>(up),                          \
                           static_cast<cpp_type*>(output), elements);                 \
        break
    switch (dtype) {
        MICROLLM_SWIGLU_CASE(float, DType::Float32);
        MICROLLM_SWIGLU_CASE(__half, DType::Float16);
        MICROLLM_SWIGLU_CASE(hip_bfloat16, DType::BFloat16);
        default: throw std::invalid_argument("typed swiglu requires a floating dtype");
    }
#undef MICROLLM_SWIGLU_CASE
    check_launch("swiglu_typed_kernel");
}

template <typename Input>
void launch_quantize_for_input(const Input* input, void* output, DType fp8_dtype,
                               std::int64_t elements, float inverse_scale,
                               hipStream_t stream) {
    const auto grid = dim3(grid_size(elements));
    const auto block = dim3(kBlockSize);
    if (fp8_dtype == DType::Float8E4M3FNUZ) {
        hipLaunchKernelGGL((quantize_fp8_kernel<Input, __hip_fp8_e4m3_fnuz>), grid,
                           block, 0, stream, input,
                           static_cast<__hip_fp8_e4m3_fnuz*>(output), elements,
                           inverse_scale);
    } else if (fp8_dtype == DType::Float8E5M2FNUZ) {
        hipLaunchKernelGGL((quantize_fp8_kernel<Input, __hip_fp8_e5m2_fnuz>), grid,
                           block, 0, stream, input,
                           static_cast<__hip_fp8_e5m2_fnuz*>(output), elements,
                           inverse_scale);
    } else {
        throw std::invalid_argument("quantize output must be an MI300 FNUZ FP8 dtype");
    }
}

void launch_quantize_fp8(const void* input, DType input_dtype, void* output,
                         DType fp8_dtype, std::int64_t elements, float inverse_scale,
                         void* stream) {
    if (elements == 0) return;
    switch (input_dtype) {
        case DType::Float32:
            launch_quantize_for_input(static_cast<const float*>(input), output, fp8_dtype,
                                      elements, inverse_scale, as_stream(stream));
            break;
        case DType::Float16:
            launch_quantize_for_input(static_cast<const __half*>(input), output, fp8_dtype,
                                      elements, inverse_scale, as_stream(stream));
            break;
        case DType::BFloat16:
            launch_quantize_for_input(static_cast<const hip_bfloat16*>(input), output,
                                      fp8_dtype, elements, inverse_scale, as_stream(stream));
            break;
        default: throw std::invalid_argument("quantize input must be FP32, FP16, or BF16");
    }
    check_launch("quantize_fp8_kernel");
}

template <typename Input>
void launch_dequantize_for_input(const Input* input, void* output, DType output_dtype,
                                 std::int64_t elements, float scale,
                                 hipStream_t stream) {
    const auto grid = dim3(grid_size(elements));
    const auto block = dim3(kBlockSize);
#define MICROLLM_DEQUANT_CASE(cpp_type, enum_value)                                  \
    case enum_value:                                                                 \
        hipLaunchKernelGGL((dequantize_fp8_kernel<Input, cpp_type>), grid, block, 0, \
                           stream, input, static_cast<cpp_type*>(output), elements,  \
                           scale);                                                   \
        break
    switch (output_dtype) {
        MICROLLM_DEQUANT_CASE(float, DType::Float32);
        MICROLLM_DEQUANT_CASE(__half, DType::Float16);
        MICROLLM_DEQUANT_CASE(hip_bfloat16, DType::BFloat16);
        default: throw std::invalid_argument("dequantize output must be FP32, FP16, or BF16");
    }
#undef MICROLLM_DEQUANT_CASE
}

template <typename Input>
void launch_cast_for_input(const Input* input, void* output, DType output_dtype,
                           std::int64_t elements, hipStream_t stream) {
    const auto grid = dim3(grid_size(elements));
    const auto block = dim3(kBlockSize);
#define MICROLLM_CAST_OUTPUT_CASE(cpp_type, enum_value)                         \
    case enum_value:                                                            \
        hipLaunchKernelGGL((cast_kernel<Input, cpp_type>), grid, block, 0,     \
                           stream, input, static_cast<cpp_type*>(output),       \
                           elements);                                           \
        break
    switch (output_dtype) {
        MICROLLM_CAST_OUTPUT_CASE(float, DType::Float32);
        MICROLLM_CAST_OUTPUT_CASE(__half, DType::Float16);
        MICROLLM_CAST_OUTPUT_CASE(hip_bfloat16, DType::BFloat16);
        default: throw std::invalid_argument("cast output must be FP32, FP16, or BF16");
    }
#undef MICROLLM_CAST_OUTPUT_CASE
}

void launch_dequantize_fp8(const void* input, DType fp8_dtype, void* output,
                           DType output_dtype, std::int64_t elements, float scale,
                           void* stream) {
    if (elements == 0) return;
    if (fp8_dtype == DType::Float8E4M3FNUZ) {
        launch_dequantize_for_input(static_cast<const __hip_fp8_e4m3_fnuz*>(input),
                                    output, output_dtype, elements, scale,
                                    as_stream(stream));
    } else if (fp8_dtype == DType::Float8E5M2FNUZ) {
        launch_dequantize_for_input(static_cast<const __hip_fp8_e5m2_fnuz*>(input),
                                    output, output_dtype, elements, scale,
                                    as_stream(stream));
    } else {
        throw std::invalid_argument("dequantize input must be an MI300 FNUZ FP8 dtype");
    }
    check_launch("dequantize_fp8_kernel");
}

void launch_cast(const void* input, DType input_dtype, void* output,
                 DType output_dtype, std::int64_t elements, void* stream) {
    if (elements == 0) return;
    switch (input_dtype) {
        case DType::Float32:
            launch_cast_for_input(static_cast<const float*>(input), output,
                                  output_dtype, elements, as_stream(stream));
            break;
        case DType::Float16:
            launch_cast_for_input(static_cast<const __half*>(input), output,
                                  output_dtype, elements, as_stream(stream));
            break;
        case DType::BFloat16:
            launch_cast_for_input(static_cast<const hip_bfloat16*>(input), output,
                                  output_dtype, elements, as_stream(stream));
            break;
        default: throw std::invalid_argument("cast input must be FP32, FP16, or BF16");
    }
    check_launch("cast_kernel");
}

void launch_matmul(const float* left, const float* right, float* output,
                   std::int64_t batches, std::int64_t rows, std::int64_t inner,
                   std::int64_t columns, void* stream) {
    const auto elements = batches * rows * columns;
    if (elements == 0) return;
    hipLaunchKernelGGL(matmul_kernel, dim3(grid_size(elements)), dim3(kBlockSize), 0,
                       as_stream(stream), left, right, output, batches, rows, inner, columns);
    check_launch("matmul_kernel");
}

void launch_embedding(const float* weight, const std::int32_t* indices, float* output,
                      std::int64_t tokens, std::int64_t vocabulary, std::int64_t width,
                      void* stream) {
    const auto elements = tokens * width;
    if (elements == 0) return;
    hipLaunchKernelGGL(embedding_kernel, dim3(grid_size(elements)), dim3(kBlockSize), 0,
                       as_stream(stream), weight, indices, output, elements, vocabulary, width);
    check_launch("embedding_kernel");
}

void launch_softmax(const float* input, float* output, std::int64_t rows,
                    std::int64_t width, void* stream) {
    if (rows == 0) return;
    hipLaunchKernelGGL(softmax_kernel, dim3(grid_size(rows)), dim3(kBlockSize), 0,
                       as_stream(stream), input, output, rows, width);
    check_launch("softmax_kernel");
}

void launch_rms_norm(const float* input, const float* weight, float* output,
                     std::int64_t rows, std::int64_t width, float epsilon, void* stream) {
    if (rows == 0) return;
    hipLaunchKernelGGL(rms_norm_kernel, dim3(static_cast<unsigned>(rows)), dim3(kBlockSize), 0,
                       as_stream(stream), input, weight, output, rows, width, epsilon);
    check_launch("rms_norm_kernel");
}

void launch_add_rms_norm(const float* left, const float* right, const float* weight,
                         float* sum, float* normalized, std::int64_t rows,
                         std::int64_t width, float epsilon, void* stream) {
    if (rows == 0) return;
    if (width >= 1024) {
        hipLaunchKernelGGL((add_rms_norm_kernel<512>),
                           dim3(static_cast<unsigned>(rows)), dim3(512), 0,
                           as_stream(stream), left, right, weight, sum, normalized,
                           rows, width, epsilon);
    } else {
        hipLaunchKernelGGL((add_rms_norm_kernel<kBlockSize>),
                           dim3(static_cast<unsigned>(rows)), dim3(kBlockSize), 0,
                           as_stream(stream), left, right, weight, sum, normalized,
                           rows, width, epsilon);
    }
    check_launch("add_rms_norm_kernel");
}

void launch_silu(const float* input, float* output, std::int64_t elements, void* stream) {
    if (elements == 0) return;
    hipLaunchKernelGGL(silu_kernel, dim3(grid_size(elements)), dim3(kBlockSize), 0,
                       as_stream(stream), input, output, elements);
    check_launch("silu_kernel");
}

void launch_swiglu(const float* gate, const float* up, float* output,
                   std::int64_t elements, void* stream) {
    if (elements == 0) return;
    hipLaunchKernelGGL(swiglu_kernel, dim3(grid_size(elements)), dim3(kBlockSize), 0,
                       as_stream(stream), gate, up, output, elements);
    check_launch("swiglu_kernel");
}

void launch_rope(const float* input, float* output, std::int64_t elements,
                 std::int64_t head_width, std::int64_t sequence_size,
                 std::int64_t sequence_stride, std::int64_t position_offset, float base,
                 void* stream) {
    if (elements == 0) return;
    const auto pairs = elements / 2;
    hipLaunchKernelGGL(rope_kernel, dim3(grid_size(pairs)), dim3(kBlockSize), 0,
                       as_stream(stream), input, output, pairs, head_width, sequence_size,
                       sequence_stride, position_offset, base);
    check_launch("rope_kernel");
}

void launch_rope_split_half(const float* input, float* output,
                            std::int64_t elements, std::int64_t head_width,
                            std::int64_t sequence_size,
                            std::int64_t sequence_stride,
                            std::int64_t position_offset, float base, void* stream) {
    if (elements == 0) return;
    const auto pairs = elements / 2;
    hipLaunchKernelGGL(rope_split_half_kernel, dim3(grid_size(pairs)),
                       dim3(kBlockSize), 0, as_stream(stream), input, output, pairs,
                       head_width, sequence_size, sequence_stride, position_offset, base);
    check_launch("rope_split_half_kernel");
}

void launch_rope_split_half_bias(const float* input, const float* bias, float* output,
                                 std::int64_t elements, std::int64_t heads,
                                 std::int64_t sequence_size, std::int64_t head_width,
                                 std::int64_t position_offset, float base, void* stream) {
    if (elements == 0) return;
    const auto pairs = elements / 2;
    hipLaunchKernelGGL(rope_split_half_bias_kernel, dim3(grid_size(pairs)),
                       dim3(kBlockSize), 0, as_stream(stream), input, bias, output,
                       pairs, heads, sequence_size, head_width, position_offset, base);
    check_launch("rope_split_half_bias_kernel");
}

void launch_cross_entropy(const float* logits, const std::int32_t* targets, float* output,
                          float* row_data, std::int64_t rows,
                          std::int64_t classes, void* stream) {
    hipLaunchKernelGGL(cross_entropy_rows_kernel, dim3(static_cast<unsigned>(rows)),
                       dim3(kBlockSize), 0, as_stream(stream), logits, targets,
                       row_data, rows, classes);
    check_launch("cross_entropy_rows_kernel");
    hipLaunchKernelGGL(cross_entropy_finalize_kernel, dim3(1), dim3(kBlockSize), 0,
                       as_stream(stream), row_data, output, rows);
    check_launch("cross_entropy_finalize_kernel");
}

void launch_reduce_sum(const float* input, float* output, std::int64_t elements,
                       void* stream) {
    hipLaunchKernelGGL(reduce_sum_kernel, dim3(1), dim3(1), 0, as_stream(stream), input,
                       output, elements);
    check_launch("reduce_sum_kernel");
}

void launch_broadcast_scalar(const float* scalar, float* output, std::int64_t elements,
                             void* stream) {
    if (elements == 0) return;
    hipLaunchKernelGGL(broadcast_scalar_kernel, dim3(grid_size(elements)),
                       dim3(kBlockSize), 0, as_stream(stream), scalar, output, elements);
    check_launch("broadcast_scalar_kernel");
}

void launch_embedding_backward(const float* gradient, const std::int32_t* indices,
                               float* weight_gradient, std::int64_t tokens,
                               std::int64_t vocabulary, std::int64_t width, void* stream) {
    const auto elements = tokens * width;
    if (elements == 0) return;
    hipLaunchKernelGGL(embedding_backward_kernel, dim3(grid_size(elements)),
                       dim3(kBlockSize), 0, as_stream(stream), gradient, indices,
                       weight_gradient, elements, vocabulary, width);
    check_launch("embedding_backward_kernel");
}

void launch_softmax_backward(const float* output, const float* gradient,
                             float* input_gradient, std::int64_t rows,
                             std::int64_t width, void* stream) {
    if (rows == 0) return;
    hipLaunchKernelGGL(softmax_backward_kernel, dim3(grid_size(rows)), dim3(kBlockSize),
                       0, as_stream(stream), output, gradient, input_gradient, rows, width);
    check_launch("softmax_backward_kernel");
}

void launch_rms_norm_backward(const float* input, const float* weight,
                              const float* gradient, float* input_gradient,
                              float* weight_gradient, float* row_inverse_rms,
                              std::int64_t rows,
                              std::int64_t width, float epsilon, void* stream) {
    if (rows == 0) return;
    hipLaunchKernelGGL(rms_norm_backward_kernel, dim3(static_cast<unsigned>(rows)), dim3(kBlockSize),
                       0, as_stream(stream), input, weight, gradient, input_gradient,
                       row_inverse_rms, rows, width, epsilon);
    check_launch("rms_norm_backward_kernel");
    hipLaunchKernelGGL(rms_norm_weight_gradient_kernel, dim3(grid_size(width)),
                       dim3(kBlockSize), 0, as_stream(stream), input, gradient,
                       row_inverse_rms, weight_gradient, rows, width);
    check_launch("rms_norm_weight_gradient_kernel");
}

void launch_kv_cache_store(const float* current, float* cache,
                           std::int64_t heads, std::int64_t capacity,
                           std::int64_t width, std::int64_t position, void* stream) {
    const auto elements = heads * width;
    hipLaunchKernelGGL(kv_cache_store_kernel, dim3(grid_size(elements)),
                       dim3(kBlockSize), 0, as_stream(stream), current, cache,
                       heads, capacity, width, position);
    check_launch("kv_cache_store_kernel");
}

void launch_cached_attention_scores(
    const float* query, const float* key_cache, float* scores,
    std::int64_t heads, [[maybe_unused]] std::int64_t kv_heads,
    std::int64_t sequence, std::int64_t cache_head_stride,
    std::int64_t width, std::int64_t repeats, float scale, void* stream) {
    const auto elements = heads * sequence;
    hipLaunchKernelGGL(cached_attention_scores_kernel, dim3(grid_size(elements)),
                       dim3(kBlockSize), 0, as_stream(stream), query, key_cache,
                       scores, heads, sequence, cache_head_stride, width, repeats, scale);
    check_launch("cached_attention_scores_kernel");
}

void launch_cached_attention_context(
    const float* probabilities, const float* value_cache, float* output,
    std::int64_t heads, [[maybe_unused]] std::int64_t kv_heads,
    std::int64_t sequence, std::int64_t cache_head_stride,
    std::int64_t width, std::int64_t repeats, void* stream) {
    const auto elements = heads * width;
    hipLaunchKernelGGL(cached_attention_context_kernel, dim3(grid_size(elements)),
                       dim3(kBlockSize), 0, as_stream(stream), probabilities,
                       value_cache, output, heads, sequence, cache_head_stride,
                       width, repeats);
    check_launch("cached_attention_context_kernel");
}

void launch_cached_attention_fused(
    const float* query, const float* key_cache, const float* value_cache,
    float* output, std::int64_t heads, std::int64_t sequence,
    std::int64_t cache_head_stride, std::int64_t width,
    std::int64_t repeats, float scale, void* stream) {
    hipLaunchKernelGGL(cached_attention_fused_kernel,
                       dim3(static_cast<unsigned>(heads)), dim3(kBlockSize),
                       static_cast<std::size_t>(sequence) * sizeof(float),
                       as_stream(stream), query, key_cache, value_cache, output,
                       heads, sequence, cache_head_stride, width, repeats, scale);
    check_launch("cached_attention_fused_kernel");
}

void launch_argmax(const float* input, std::int32_t* output,
                   std::int64_t elements, void* stream) {
    hipLaunchKernelGGL(argmax_kernel, dim3(1), dim3(kBlockSize), 0,
                       as_stream(stream), input, output, elements);
    check_launch("argmax_kernel");
}

void launch_argmax_two_stage(const float* input, float* partials,
                             std::int32_t* output, std::int64_t elements,
                             std::int64_t blocks, void* stream) {
    hipLaunchKernelGGL(argmax_partial_kernel,
                       dim3(static_cast<unsigned>(blocks)), dim3(kBlockSize), 0,
                       as_stream(stream), input, partials, elements);
    check_launch("argmax_partial_kernel");
    hipLaunchKernelGGL(argmax_finalize_kernel, dim3(1), dim3(kBlockSize), 0,
                       as_stream(stream), partials, output, blocks);
    check_launch("argmax_finalize_kernel");
}

void launch_silu_backward(const float* input, const float* gradient,
                          float* input_gradient, std::int64_t elements, void* stream) {
    if (elements == 0) return;
    hipLaunchKernelGGL(silu_backward_kernel, dim3(grid_size(elements)), dim3(kBlockSize),
                       0, as_stream(stream), input, gradient, input_gradient, elements);
    check_launch("silu_backward_kernel");
}

void launch_swiglu_backward(const float* gate, const float* up, const float* gradient,
                            float* gate_gradient, float* up_gradient,
                            std::int64_t elements, void* stream) {
    if (elements == 0) return;
    hipLaunchKernelGGL(swiglu_backward_kernel, dim3(grid_size(elements)), dim3(kBlockSize),
                       0, as_stream(stream), gate, up, gradient, gate_gradient, up_gradient,
                       elements);
    check_launch("swiglu_backward_kernel");
}

void launch_rope_backward(const float* gradient, float* input_gradient,
                          std::int64_t elements, std::int64_t head_width,
                          std::int64_t sequence_size, std::int64_t sequence_stride,
                          std::int64_t position_offset, float base, void* stream) {
    if (elements == 0) return;
    const auto pairs = elements / 2;
    hipLaunchKernelGGL(rope_backward_kernel, dim3(grid_size(pairs)), dim3(kBlockSize), 0,
                       as_stream(stream), gradient, input_gradient, pairs, head_width,
                       sequence_size, sequence_stride, position_offset, base);
    check_launch("rope_backward_kernel");
}

void launch_rope_split_half_backward(
    const float* gradient, float* input_gradient, std::int64_t elements,
    std::int64_t head_width, std::int64_t sequence_size,
    std::int64_t sequence_stride, std::int64_t position_offset, float base,
    void* stream) {
    if (elements == 0) return;
    const auto pairs = elements / 2;
    hipLaunchKernelGGL(rope_split_half_backward_kernel, dim3(grid_size(pairs)),
                       dim3(kBlockSize), 0, as_stream(stream), gradient,
                       input_gradient, pairs, head_width, sequence_size,
                       sequence_stride, position_offset, base);
    check_launch("rope_split_half_backward_kernel");
}

void launch_cross_entropy_backward(const float* logits, const std::int32_t* targets,
                                   const float* loss_gradient, float* logits_gradient,
                                   float* row_stats, float* factor,
                                   std::int64_t rows, std::int64_t classes, void* stream) {
    hipLaunchKernelGGL(cross_entropy_backward_stats_kernel,
                       dim3(static_cast<unsigned>(rows)), dim3(kBlockSize), 0,
                       as_stream(stream), logits, targets, row_stats, rows, classes);
    check_launch("cross_entropy_backward_stats_kernel");
    hipLaunchKernelGGL(cross_entropy_backward_factor_kernel, dim3(1), dim3(kBlockSize), 0,
                       as_stream(stream), targets, loss_gradient, factor, rows, classes);
    check_launch("cross_entropy_backward_factor_kernel");
    const auto elements = rows * classes;
    hipLaunchKernelGGL(cross_entropy_backward_elements_kernel,
                       dim3(grid_size(elements)), dim3(kBlockSize), 0, as_stream(stream),
                       logits, targets, row_stats, factor, logits_gradient, elements, classes);
    check_launch("cross_entropy_backward_elements_kernel");
}

void launch_causal_softmax(const float* scores, float* output, std::int64_t rows,
                           std::int64_t sequence, void* stream) {
    if (rows == 0) return;
    hipLaunchKernelGGL(causal_softmax_kernel, dim3(grid_size(rows)), dim3(kBlockSize), 0,
                       as_stream(stream), scores, output, rows, sequence);
    check_launch("causal_softmax_kernel");
}

void launch_causal_softmax_backward(const float* output, const float* gradient,
                                    float* input_gradient, std::int64_t rows,
                                    std::int64_t sequence, void* stream) {
    if (rows == 0) return;
    hipLaunchKernelGGL(causal_softmax_backward_kernel, dim3(grid_size(rows)),
                       dim3(kBlockSize), 0, as_stream(stream), output, gradient,
                       input_gradient, rows, sequence);
    check_launch("causal_softmax_backward_kernel");
}

void launch_repeat_interleave(const float* input, float* output,
                              std::int64_t output_elements, std::int64_t repeated_width,
                              std::int64_t inner, std::int64_t repeats, void* stream) {
    if (output_elements == 0) return;
    hipLaunchKernelGGL(repeat_interleave_kernel, dim3(grid_size(output_elements)),
                       dim3(kBlockSize), 0, as_stream(stream), input, output,
                       output_elements, repeated_width, inner, repeats);
    check_launch("repeat_interleave_kernel");
}

void launch_repeat_interleave_backward(const float* gradient, float* input_gradient,
                                       std::int64_t input_elements,
                                       std::int64_t input_width, std::int64_t inner,
                                       std::int64_t repeats, void* stream) {
    if (input_elements == 0) return;
    hipLaunchKernelGGL(repeat_interleave_backward_kernel,
                       dim3(grid_size(input_elements)), dim3(kBlockSize), 0,
                       as_stream(stream), gradient, input_gradient, input_elements,
                       input_width, inner, repeats);
    check_launch("repeat_interleave_backward_kernel");
}

}  // namespace microllm::ops::hip
