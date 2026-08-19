#include "kernels.h"

#include <stdexcept>
#include <string>

#include <hip/hip_runtime.h>

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

__global__ void fill_kernel(float* output, std::int64_t elements, float value) {
    const auto index = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index < elements) output[index] = value;
}

__global__ void add_kernel(const float* left, const float* right, float* output,
                           std::int64_t elements) {
    const auto index = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index < elements) output[index] = left[index] + right[index];
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
    const auto row = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (row >= rows) return;
    const auto base = row * width;
    float square_sum = 0.0F;
    for (std::int64_t column = 0; column < width; ++column) {
        const auto value = input[base + column];
        square_sum += value * value;
    }
    const auto inverse_rms = rsqrtf(square_sum / static_cast<float>(width) + epsilon);
    for (std::int64_t column = 0; column < width; ++column) {
        output[base + column] = input[base + column] * inverse_rms * weight[column];
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

__global__ void cross_entropy_kernel(const float* logits, const std::int32_t* targets,
                                     float* output, std::int64_t rows,
                                     std::int64_t classes) {
    if (blockIdx.x != 0 || threadIdx.x != 0) return;
    double total = 0.0;
    std::int64_t valid_rows = 0;
    for (std::int64_t row = 0; row < rows; ++row) {
        const auto label = static_cast<std::int64_t>(targets[row]);
        if (label == -100) continue;
        if (label < 0 || label >= classes) {
            output[0] = nanf("");
            return;
        }
        ++valid_rows;
        float maximum = -INFINITY;
        for (std::int64_t column = 0; column < classes; ++column) {
            maximum = fmaxf(maximum, logits[row * classes + column]);
        }
        double denominator = 0.0;
        for (std::int64_t column = 0; column < classes; ++column) {
            denominator += exp(static_cast<double>(logits[row * classes + column] - maximum));
        }
        total += log(denominator) +
                 static_cast<double>(maximum - logits[row * classes + label]);
    }
    output[0] = valid_rows == 0 ? nanf("")
                               : static_cast<float>(total / static_cast<double>(valid_rows));
}

}  // namespace

void launch_fill(float* output, std::int64_t elements, float value, void* stream) {
    if (elements == 0) return;
    hipLaunchKernelGGL(fill_kernel, dim3(grid_size(elements)), dim3(kBlockSize), 0,
                       as_stream(stream), output, elements, value);
    check_launch("fill_kernel");
}

void launch_add(const float* left, const float* right, float* output,
                std::int64_t elements, void* stream) {
    if (elements == 0) return;
    hipLaunchKernelGGL(add_kernel, dim3(grid_size(elements)), dim3(kBlockSize), 0,
                       as_stream(stream), left, right, output, elements);
    check_launch("add_kernel");
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
    hipLaunchKernelGGL(rms_norm_kernel, dim3(grid_size(rows)), dim3(kBlockSize), 0,
                       as_stream(stream), input, weight, output, rows, width, epsilon);
    check_launch("rms_norm_kernel");
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

void launch_cross_entropy(const float* logits, const std::int32_t* targets, float* output,
                          std::int64_t rows, std::int64_t classes, void* stream) {
    hipLaunchKernelGGL(cross_entropy_kernel, dim3(1), dim3(1), 0, as_stream(stream),
                       logits, targets, output, rows, classes);
    check_launch("cross_entropy_kernel");
}

}  // namespace microllm::ops::hip
