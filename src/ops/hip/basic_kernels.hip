#include "kernels.h"

#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <type_traits>

#include <hip/hip_runtime.h>
#include <hip/hip_bfloat16.h>
#include <hip/hip_fp16.h>
#include <hip/hip_fp8.h>

namespace microllm::ops::hip {
namespace {

constexpr unsigned kBlockSize = 256;

struct AdamWDescriptorStaging {
    void* pointer = nullptr;
    hipEvent_t ready = nullptr;
    bool recorded = false;
};

void check_launch(const char* operation) {
    const auto status = hipGetLastError();
    if (status != hipSuccess) {
        throw std::runtime_error(std::string(operation) + ": " + hipGetErrorString(status));
    }
}

void check_hip_status(hipError_t status, const char* operation) {
    if (status != hipSuccess) {
        throw std::runtime_error(
            std::string(operation) + ": " + hipGetErrorString(status));
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
    float* second_moment, hip_bfloat16* bf16_mirror,
    std::int64_t elements, float learning_rate,
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
    if (bf16_mirror != nullptr) bf16_mirror[index] = hip_bfloat16(parameter[index]);
}

__device__ float adamw_update_value(float parameter, float gradient,
                                    float& first_moment, float& second_moment,
                                    float learning_rate, float beta1, float beta2,
                                    float epsilon, float weight_decay,
                                    float first_correction, float second_correction) {
    first_moment = beta1 * first_moment + (1.0F - beta1) * gradient;
    second_moment = beta2 * second_moment + (1.0F - beta2) * gradient * gradient;
    parameter *= 1.0F - learning_rate * weight_decay;
    parameter -= learning_rate * (first_moment / first_correction) /
                 (sqrtf(second_moment / second_correction) + epsilon);
    return parameter;
}

__device__ void adamw_update_float4(
    float* parameter, const float* gradient, float* first_moment,
    float* second_moment, hip_bfloat16* bf16_mirror, std::int64_t base,
    float learning_rate, float beta1, float beta2,
    float epsilon, float weight_decay,
    float first_correction, float second_correction) {
    auto values = *reinterpret_cast<float4*>(parameter + base);
    const auto gradients = *reinterpret_cast<const float4*>(gradient + base);
    auto first = *reinterpret_cast<float4*>(first_moment + base);
    auto second = *reinterpret_cast<float4*>(second_moment + base);
    values.x = adamw_update_value(values.x, gradients.x, first.x, second.x,
                                  learning_rate, beta1, beta2, epsilon, weight_decay,
                                  first_correction, second_correction);
    values.y = adamw_update_value(values.y, gradients.y, first.y, second.y,
                                  learning_rate, beta1, beta2, epsilon, weight_decay,
                                  first_correction, second_correction);
    values.z = adamw_update_value(values.z, gradients.z, first.z, second.z,
                                  learning_rate, beta1, beta2, epsilon, weight_decay,
                                  first_correction, second_correction);
    values.w = adamw_update_value(values.w, gradients.w, first.w, second.w,
                                  learning_rate, beta1, beta2, epsilon, weight_decay,
                                  first_correction, second_correction);
    *reinterpret_cast<float4*>(parameter + base) = values;
    *reinterpret_cast<float4*>(first_moment + base) = first;
    *reinterpret_cast<float4*>(second_moment + base) = second;
    if (bf16_mirror != nullptr) {
        bf16_mirror[base] = hip_bfloat16(values.x);
        bf16_mirror[base + 1] = hip_bfloat16(values.y);
        bf16_mirror[base + 2] = hip_bfloat16(values.z);
        bf16_mirror[base + 3] = hip_bfloat16(values.w);
    }
}

__global__ void adamw_update_vectorized_kernel(
    float* parameter, const float* gradient, float* first_moment,
    float* second_moment, hip_bfloat16* bf16_mirror,
    std::int64_t elements, float learning_rate,
    float beta1, float beta2, float epsilon, float weight_decay,
    float first_correction, float second_correction) {
    constexpr std::int64_t width = 4;
    const auto base = (static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x) *
                      width;
    if (base >= elements) return;
    if (base + width <= elements) {
        adamw_update_float4(parameter, gradient, first_moment, second_moment,
                            bf16_mirror, base, learning_rate, beta1, beta2,
                            epsilon, weight_decay, first_correction, second_correction);
        return;
    }
    for (auto index = base; index < elements; ++index) {
        auto first = first_moment[index];
        auto second = second_moment[index];
        const auto value = adamw_update_value(
            parameter[index], gradient[index], first, second, learning_rate,
            beta1, beta2, epsilon, weight_decay, first_correction, second_correction);
        parameter[index] = value;
        first_moment[index] = first;
        second_moment[index] = second;
        if (bf16_mirror != nullptr) bf16_mirror[index] = hip_bfloat16(value);
    }
}

__global__ void adamw_graph_advance_kernel(
    std::int32_t* step, float* corrections, float beta1, float beta2) {
    if (blockIdx.x != 0 || threadIdx.x != 0) return;
    const auto next = *step + 1;
    *step = next;
    corrections[0] = 1.0F - powf(beta1, static_cast<float>(next));
    corrections[1] = 1.0F - powf(beta2, static_cast<float>(next));
}

__global__ void adamw_update_graph_kernel(
    float* parameter, const float* gradient, float* first_moment,
    float* second_moment, hip_bfloat16* bf16_mirror,
    std::int64_t elements, float learning_rate, float beta1, float beta2,
    float epsilon, float weight_decay, const float* corrections) {
    const auto index = static_cast<std::int64_t>(blockIdx.x) * blockDim.x +
                       threadIdx.x;
    if (index >= elements) return;
    auto first = first_moment[index];
    auto second = second_moment[index];
    const auto value = adamw_update_value(
        parameter[index], gradient[index], first, second, learning_rate,
        beta1, beta2, epsilon, weight_decay, corrections[0], corrections[1]);
    parameter[index] = value;
    first_moment[index] = first;
    second_moment[index] = second;
    if (bf16_mirror != nullptr) bf16_mirror[index] = hip_bfloat16(value);
}

__global__ void adamw_update_graph_vectorized_kernel(
    float* parameter, const float* gradient, float* first_moment,
    float* second_moment, hip_bfloat16* bf16_mirror,
    std::int64_t elements, float learning_rate, float beta1, float beta2,
    float epsilon, float weight_decay, const float* corrections) {
    constexpr std::int64_t width = 4;
    const auto base =
        (static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x) *
        width;
    if (base >= elements) return;
    const auto first_correction = corrections[0];
    const auto second_correction = corrections[1];
    if (base + width <= elements) {
        adamw_update_float4(
            parameter, gradient, first_moment, second_moment, bf16_mirror,
            base, learning_rate, beta1, beta2, epsilon, weight_decay,
            first_correction, second_correction);
        return;
    }
    for (auto index = base; index < elements; ++index) {
        auto first = first_moment[index];
        auto second = second_moment[index];
        const auto value = adamw_update_value(
            parameter[index], gradient[index], first, second, learning_rate,
            beta1, beta2, epsilon, weight_decay, first_correction,
            second_correction);
        parameter[index] = value;
        first_moment[index] = first;
        second_moment[index] = second;
        if (bf16_mirror != nullptr) bf16_mirror[index] = hip_bfloat16(value);
    }
}

__device__ float adamw_update_bf16_moment_value(
    float parameter, float gradient, hip_bfloat16& first_moment,
    hip_bfloat16& second_moment, float learning_rate, float beta1,
    float beta2, float epsilon, float weight_decay, float first_correction,
    float second_correction);

__global__ void adamw_update_multi_kernel(
    const AdamWMultiTensorDescriptor* descriptors,
    const std::int32_t* block_to_tensor, float learning_rate,
    float beta1, float beta2, float epsilon, float weight_decay,
    float first_correction, float second_correction,
    const float* device_corrections) {
    if (device_corrections != nullptr) {
        first_correction = device_corrections[0];
        second_correction = device_corrections[1];
    }
    const auto tensor_index = block_to_tensor[blockIdx.x];
    const auto descriptor = descriptors[tensor_index];
    if (descriptor.gradient == nullptr) return;
    constexpr std::int64_t width = 4;
    const auto base =
        (static_cast<std::int64_t>(blockIdx.x) - descriptor.first_block) *
            static_cast<std::int64_t>(blockDim.x) * width +
        static_cast<std::int64_t>(threadIdx.x) * width;
    if (base >= descriptor.elements) return;
    if (descriptor.bf16_moments != 0U) {
        auto* first_moments =
            static_cast<hip_bfloat16*>(descriptor.first_moment);
        auto* second_moments =
            static_cast<hip_bfloat16*>(descriptor.second_moment);
        for (auto index = base;
             index < descriptor.elements && index < base + width; ++index) {
            auto first = first_moments[index];
            auto second = second_moments[index];
            const auto value = adamw_update_bf16_moment_value(
                descriptor.parameter[index], descriptor.gradient[index],
                first, second, learning_rate, beta1, beta2, epsilon,
                weight_decay, first_correction, second_correction);
            descriptor.parameter[index] = value;
            first_moments[index] = first;
            second_moments[index] = second;
            if (descriptor.bf16_mirror != nullptr) {
                static_cast<hip_bfloat16*>(descriptor.bf16_mirror)[index] =
                    hip_bfloat16(value);
            }
        }
        return;
    }
    auto* first_moments = static_cast<float*>(descriptor.first_moment);
    auto* second_moments = static_cast<float*>(descriptor.second_moment);
    for (auto index = base;
         index < descriptor.elements && index < base + width; ++index) {
        auto first = first_moments[index];
        auto second = second_moments[index];
        const auto value = adamw_update_value(
            descriptor.parameter[index], descriptor.gradient[index], first,
            second, learning_rate, beta1, beta2, epsilon, weight_decay,
            first_correction, second_correction);
        descriptor.parameter[index] = value;
        first_moments[index] = first;
        second_moments[index] = second;
        if (descriptor.bf16_mirror != nullptr) {
            static_cast<hip_bfloat16*>(descriptor.bf16_mirror)[index] =
                hip_bfloat16(value);
        }
    }
}

__global__ void adamw_update_bf16_moments_kernel(
    float* parameter, const float* gradient, hip_bfloat16* first_moment,
    hip_bfloat16* second_moment, hip_bfloat16* parameter_mirror,
    std::int64_t elements, float learning_rate, float beta1, float beta2,
    float epsilon, float weight_decay, float first_correction,
    float second_correction) {
    const auto index = static_cast<std::int64_t>(blockIdx.x) * blockDim.x +
                       static_cast<std::int64_t>(threadIdx.x);
    if (index >= elements) return;
    const auto grad = gradient[index];
    const auto first = hip_bfloat16(
        beta1 * static_cast<float>(first_moment[index]) +
        (1.0F - beta1) * grad);
    const auto second = hip_bfloat16(
        beta2 * static_cast<float>(second_moment[index]) +
        (1.0F - beta2) * grad * grad);
    first_moment[index] = first;
    second_moment[index] = second;
    auto value = parameter[index] * (1.0F - learning_rate * weight_decay);
    value -= learning_rate * (static_cast<float>(first) / first_correction) /
             (sqrtf(static_cast<float>(second) / second_correction) + epsilon);
    parameter[index] = value;
    if (parameter_mirror != nullptr) parameter_mirror[index] = hip_bfloat16(value);
}

__device__ float adamw_update_bf16_moment_value(
    float parameter, float gradient, hip_bfloat16& first_moment,
    hip_bfloat16& second_moment, float learning_rate, float beta1,
    float beta2, float epsilon, float weight_decay, float first_correction,
    float second_correction) {
    first_moment = hip_bfloat16(
        beta1 * static_cast<float>(first_moment) +
        (1.0F - beta1) * gradient);
    second_moment = hip_bfloat16(
        beta2 * static_cast<float>(second_moment) +
        (1.0F - beta2) * gradient * gradient);
    parameter *= 1.0F - learning_rate * weight_decay;
    parameter -=
        learning_rate * (static_cast<float>(first_moment) / first_correction) /
        (sqrtf(static_cast<float>(second_moment) / second_correction) + epsilon);
    return parameter;
}

__global__ void adamw_update_bf16_moments_vectorized_kernel(
    float* parameter, const float* gradient, hip_bfloat16* first_moment,
    hip_bfloat16* second_moment, hip_bfloat16* parameter_mirror,
    std::int64_t elements, float learning_rate, float beta1, float beta2,
    float epsilon, float weight_decay, float first_correction,
    float second_correction) {
    constexpr std::int64_t width = 4;
    const auto base =
        (static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x) *
        width;
    if (base >= elements) return;
    if (base + width <= elements) {
        auto values = *reinterpret_cast<float4*>(parameter + base);
        const auto gradients =
            *reinterpret_cast<const float4*>(gradient + base);
        hip_bfloat16 first0 = first_moment[base];
        hip_bfloat16 first1 = first_moment[base + 1];
        hip_bfloat16 first2 = first_moment[base + 2];
        hip_bfloat16 first3 = first_moment[base + 3];
        hip_bfloat16 second0 = second_moment[base];
        hip_bfloat16 second1 = second_moment[base + 1];
        hip_bfloat16 second2 = second_moment[base + 2];
        hip_bfloat16 second3 = second_moment[base + 3];
        values.x = adamw_update_bf16_moment_value(
            values.x, gradients.x, first0, second0, learning_rate, beta1,
            beta2, epsilon, weight_decay, first_correction, second_correction);
        values.y = adamw_update_bf16_moment_value(
            values.y, gradients.y, first1, second1, learning_rate, beta1,
            beta2, epsilon, weight_decay, first_correction, second_correction);
        values.z = adamw_update_bf16_moment_value(
            values.z, gradients.z, first2, second2, learning_rate, beta1,
            beta2, epsilon, weight_decay, first_correction, second_correction);
        values.w = adamw_update_bf16_moment_value(
            values.w, gradients.w, first3, second3, learning_rate, beta1,
            beta2, epsilon, weight_decay, first_correction, second_correction);
        *reinterpret_cast<float4*>(parameter + base) = values;
        first_moment[base] = first0;
        first_moment[base + 1] = first1;
        first_moment[base + 2] = first2;
        first_moment[base + 3] = first3;
        second_moment[base] = second0;
        second_moment[base + 1] = second1;
        second_moment[base + 2] = second2;
        second_moment[base + 3] = second3;
        if (parameter_mirror != nullptr) {
            parameter_mirror[base] = hip_bfloat16(values.x);
            parameter_mirror[base + 1] = hip_bfloat16(values.y);
            parameter_mirror[base + 2] = hip_bfloat16(values.z);
            parameter_mirror[base + 3] = hip_bfloat16(values.w);
        }
        return;
    }
    for (auto index = base; index < elements; ++index) {
        auto first = first_moment[index];
        auto second = second_moment[index];
        const auto value = adamw_update_bf16_moment_value(
            parameter[index], gradient[index], first, second, learning_rate,
            beta1, beta2, epsilon, weight_decay, first_correction,
            second_correction);
        parameter[index] = value;
        first_moment[index] = first;
        second_moment[index] = second;
        if (parameter_mirror != nullptr) {
            parameter_mirror[index] = hip_bfloat16(value);
        }
    }
}

__global__ void adamw_update_bf16_moments_graph_kernel(
    float* parameter, const float* gradient, hip_bfloat16* first_moment,
    hip_bfloat16* second_moment, hip_bfloat16* parameter_mirror,
    std::int64_t elements, float learning_rate, float beta1, float beta2,
    float epsilon, float weight_decay, const float* corrections) {
    const auto index = static_cast<std::int64_t>(blockIdx.x) * blockDim.x +
                       threadIdx.x;
    if (index >= elements) return;
    auto first = first_moment[index];
    auto second = second_moment[index];
    const auto value = adamw_update_bf16_moment_value(
        parameter[index], gradient[index], first, second, learning_rate,
        beta1, beta2, epsilon, weight_decay, corrections[0], corrections[1]);
    parameter[index] = value;
    first_moment[index] = first;
    second_moment[index] = second;
    if (parameter_mirror != nullptr) {
        parameter_mirror[index] = hip_bfloat16(value);
    }
}

__global__ void adamw_update_bf16_moments_graph_vectorized_kernel(
    float* parameter, const float* gradient, hip_bfloat16* first_moment,
    hip_bfloat16* second_moment, hip_bfloat16* parameter_mirror,
    std::int64_t elements, float learning_rate, float beta1, float beta2,
    float epsilon, float weight_decay, const float* corrections) {
    constexpr std::int64_t width = 4;
    const auto base =
        (static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x) *
        width;
    if (base >= elements) return;
    const auto first_correction = corrections[0];
    const auto second_correction = corrections[1];
    for (auto index = base;
         index < elements && index < base + width; ++index) {
        auto first = first_moment[index];
        auto second = second_moment[index];
        const auto value = adamw_update_bf16_moment_value(
            parameter[index], gradient[index], first, second, learning_rate,
            beta1, beta2, epsilon, weight_decay, first_correction,
            second_correction);
        parameter[index] = value;
        first_moment[index] = first;
        second_moment[index] = second;
        if (parameter_mirror != nullptr) {
            parameter_mirror[index] = hip_bfloat16(value);
        }
    }
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

__global__ void bias_gradient_cooperative_kernel(
    const float* gradient, float* output,
    std::int64_t rows, std::int64_t width) {
    constexpr int kColumns = 32;
    constexpr int kRowLanes = 8;
    __shared__ float partial[kRowLanes][kColumns + 1];
    const auto column = static_cast<std::int64_t>(blockIdx.x) * kColumns +
                        static_cast<std::int64_t>(threadIdx.x);
    float total = 0.0F;
    if (column < width) {
        for (std::int64_t row = threadIdx.y; row < rows; row += kRowLanes) {
            total += gradient[row * width + column];
        }
    }
    partial[threadIdx.y][threadIdx.x] = total;
    __syncthreads();
    if (threadIdx.y == 0 && column < width) {
        float combined = 0.0F;
        for (int lane = 0; lane < kRowLanes; ++lane) {
            combined += partial[lane][threadIdx.x];
        }
        output[column] = combined;
    }
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

__global__ void add_bias_bf16_kernel(
    const hip_bfloat16* input, const float* bias, hip_bfloat16* output,
    std::int64_t elements, std::int64_t width) {
    const auto index = static_cast<std::int64_t>(blockIdx.x) * blockDim.x +
                       threadIdx.x;
    if (index < elements) {
        output[index] = store_from_float<hip_bfloat16>(
            load_as_float(input[index]) + bias[index % width]);
    }
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
struct alignas(16) BinaryPacket16 {
    T values[16 / sizeof(T)];
};

struct AddTypedOperation {
    __device__ static float apply(float left, float right) { return left + right; }
};

struct MultiplyTypedOperation {
    __device__ static float apply(float left, float right) { return left * right; }
};

template <typename T, typename Operation>
__global__ void binary_typed_vector16_kernel(
    const T* left, const T* right, T* output, std::int64_t elements) {
    constexpr auto lanes = static_cast<std::int64_t>(16 / sizeof(T));
    const auto packet_index =
        static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const auto base = packet_index * lanes;
    if (base >= elements) return;
    if (base + lanes <= elements) {
        const auto left_packet =
            reinterpret_cast<const BinaryPacket16<T>*>(left)[packet_index];
        const auto right_packet =
            reinterpret_cast<const BinaryPacket16<T>*>(right)[packet_index];
        BinaryPacket16<T> output_packet{};
#pragma unroll
        for (std::int64_t lane = 0; lane < lanes; ++lane) {
            output_packet.values[lane] = store_from_float<T>(Operation::apply(
                load_as_float(left_packet.values[lane]),
                load_as_float(right_packet.values[lane])));
        }
        reinterpret_cast<BinaryPacket16<T>*>(output)[packet_index] = output_packet;
        return;
    }
    for (auto index = base; index < elements; ++index) {
        output[index] = store_from_float<T>(Operation::apply(
            load_as_float(left[index]), load_as_float(right[index])));
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

struct alignas(8) BFloat16x4 {
    hip_bfloat16 x;
    hip_bfloat16 y;
    hip_bfloat16 z;
    hip_bfloat16 w;
};

__device__ hip_bfloat16 swiglu_bfloat16_value(hip_bfloat16 gate,
                                               hip_bfloat16 up) {
    const auto gate_value = static_cast<float>(gate);
    return hip_bfloat16(gate_value * sigmoid_typed_device(gate_value) *
                        static_cast<float>(up));
}

__global__ void swiglu_bfloat16_vectorized_kernel(
    const hip_bfloat16* gate, const hip_bfloat16* up, hip_bfloat16* output,
    std::int64_t elements) {
    constexpr std::int64_t width = 4;
    const auto base = (static_cast<std::int64_t>(blockIdx.x) * blockDim.x +
                       threadIdx.x) * width;
    if (base >= elements) return;
    if (base + width <= elements) {
        const auto gate_values =
            *reinterpret_cast<const BFloat16x4*>(gate + base);
        const auto up_values = *reinterpret_cast<const BFloat16x4*>(up + base);
        BFloat16x4 result;
        result.x = swiglu_bfloat16_value(gate_values.x, up_values.x);
        result.y = swiglu_bfloat16_value(gate_values.y, up_values.y);
        result.z = swiglu_bfloat16_value(gate_values.z, up_values.z);
        result.w = swiglu_bfloat16_value(gate_values.w, up_values.w);
        *reinterpret_cast<BFloat16x4*>(output + base) = result;
        return;
    }
    for (auto index = base; index < elements; ++index) {
        output[index] = swiglu_bfloat16_value(gate[index], up[index]);
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

template <typename Input, typename Output>
__global__ void cast_transpose_2d_kernel(const Input* input, Output* output,
                                         std::int64_t rows,
                                         std::int64_t columns) {
    const auto index = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const auto elements = rows * columns;
    if (index >= elements) return;
    const auto row = index / columns;
    const auto column = index % columns;
    output[column * rows + row] = store_from_float<Output>(load_as_float(input[index]));
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

__global__ void rms_norm_bf16_output_kernel(
    const float* input, const float* weight, hip_bfloat16* output,
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
    const auto inverse_rms =
        rsqrtf(square_sum / static_cast<float>(width) + epsilon);
    for (std::int64_t column = threadIdx.x; column < width; column += blockDim.x) {
        output[base + column] = hip_bfloat16(
            input[base + column] * inverse_rms * weight[column]);
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

template <typename Input, typename Output>
__global__ void rope_split_half_bias_bthd_kernel(
    const Input* input, const float* bias, Output* output, std::int64_t pairs,
    std::int64_t sequence_size, std::int64_t heads, std::int64_t head_width,
    std::int64_t position_offset, float base) {
    const auto pair_index = static_cast<std::int64_t>(blockIdx.x) * blockDim.x +
                            threadIdx.x;
    if (pair_index >= pairs) return;
    const auto half = head_width / 2;
    const auto instance = pair_index / half;
    const auto pair = pair_index % half;
    const auto heads_per_batch = heads * sequence_size;
    const auto batch = instance / heads_per_batch;
    const auto within_batch = instance % heads_per_batch;
    const auto head = within_batch / sequence_size;
    const auto position = within_batch % sequence_size;
    const auto input_row = ((batch * sequence_size + position) * heads + head) *
                           head_width;
    const auto output_row = instance * head_width;
    const auto input_first = input_row + pair;
    const auto input_second = input_first + half;
    const auto output_first = output_row + pair;
    const auto output_second = output_first + half;
    const auto bias_first = head * head_width + pair;
    const auto bias_second = bias_first + half;
    const auto angle = static_cast<float>(position + position_offset) *
                       powf(base, -2.0F * static_cast<float>(pair) /
                                      static_cast<float>(head_width));
    float sine = 0.0F;
    float cosine = 0.0F;
    sincosf(angle, &sine, &cosine);
    const auto first_value =
        load_as_float(input[input_first]) + bias[bias_first];
    const auto second_value =
        load_as_float(input[input_second]) + bias[bias_second];
    output[output_first] = store_from_float<Output>(
        first_value * cosine - second_value * sine);
    output[output_second] = store_from_float<Output>(
        first_value * sine + second_value * cosine);
}

__global__ void rope_positions_kernel(
    const float* input, const std::int32_t* positions, float* output,
    std::int64_t pairs, std::int64_t heads, std::int64_t head_width,
    float base) {
    const auto pair_index = static_cast<std::int64_t>(blockIdx.x) * blockDim.x +
                            threadIdx.x;
    if (pair_index >= pairs) return;
    const auto even = pair_index * 2;
    const auto batch = even / (heads * head_width);
    const auto pair = (even % head_width) / 2;
    const auto position = positions[batch];
    if (position < 0) {
        output[even] = 0.0F;
        output[even + 1] = 0.0F;
        return;
    }
    const auto angle = static_cast<float>(position) *
                       powf(base, -2.0F * static_cast<float>(pair) /
                                      static_cast<float>(head_width));
    float sine = 0.0F;
    float cosine = 0.0F;
    sincosf(angle, &sine, &cosine);
    output[even] = input[even] * cosine - input[even + 1] * sine;
    output[even + 1] = input[even] * sine + input[even + 1] * cosine;
}

__global__ void rope_split_half_positions_kernel(
    const float* input, const std::int32_t* positions, float* output,
    std::int64_t pairs, std::int64_t heads, std::int64_t head_width,
    float base) {
    const auto pair_index = static_cast<std::int64_t>(blockIdx.x) * blockDim.x +
                            threadIdx.x;
    if (pair_index >= pairs) return;
    const auto half = head_width / 2;
    const auto head_instance = pair_index / half;
    const auto pair = pair_index % half;
    const auto batch = head_instance / heads;
    const auto first = head_instance * head_width + pair;
    const auto second = first + half;
    const auto position = positions[batch];
    if (position < 0) {
        output[first] = 0.0F;
        output[second] = 0.0F;
        return;
    }
    const auto angle = static_cast<float>(position) *
                       powf(base, -2.0F * static_cast<float>(pair) /
                                      static_cast<float>(head_width));
    float sine = 0.0F;
    float cosine = 0.0F;
    sincosf(angle, &sine, &cosine);
    output[first] = input[first] * cosine - input[second] * sine;
    output[second] = input[first] * sine + input[second] * cosine;
}

__global__ void rope_split_half_bias_positions_kernel(
    const float* input, const float* bias, const std::int32_t* positions,
    float* output, std::int64_t pairs, std::int64_t heads,
    std::int64_t head_width, float base) {
    const auto pair_index = static_cast<std::int64_t>(blockIdx.x) * blockDim.x +
                            threadIdx.x;
    if (pair_index >= pairs) return;
    const auto half = head_width / 2;
    const auto head_instance = pair_index / half;
    const auto pair = pair_index % half;
    const auto batch = head_instance / heads;
    const auto head = head_instance % heads;
    const auto first = head_instance * head_width + pair;
    const auto second = first + half;
    const auto bias_first = head * head_width + pair;
    const auto bias_second = bias_first + half;
    const auto position = positions[batch];
    if (position < 0) {
        output[first] = 0.0F;
        output[second] = 0.0F;
        return;
    }
    const auto angle = static_cast<float>(position) *
                       powf(base, -2.0F * static_cast<float>(pair) /
                                      static_cast<float>(head_width));
    float sine = 0.0F;
    float cosine = 0.0F;
    sincosf(angle, &sine, &cosine);
    const auto first_value = input[first] + bias[bias_first];
    const auto second_value = input[second] + bias[bias_second];
    output[first] = first_value * cosine - second_value * sine;
    output[second] = first_value * sine + second_value * cosine;
}

__device__ float block_reduce_max(float value, float* scratch) {
    const auto lane = static_cast<unsigned>(threadIdx.x);
    scratch[lane] = value;
    __syncthreads();
    for (unsigned stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (lane < stride) scratch[lane] = fmaxf(scratch[lane], scratch[lane + stride]);
        __syncthreads();
    }
    const auto result = scratch[0];
    // Every lane must read the result before a following reduction reuses the
    // same scratch buffer. Without this barrier, fast lanes can overwrite
    // scratch[0] while slower lanes are still loading the previous result.
    __syncthreads();
    return result;
}

template <typename Input>
__global__ void fp8_tensor_scale_kernel(const Input* input, float* partial_maxima,
                                        std::int64_t elements) {
    __shared__ float scratch[kBlockSize];
    float local_maximum = 0.0F;
    for (std::int64_t index =
             static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         index < elements;
         index += static_cast<std::int64_t>(gridDim.x) * blockDim.x) {
        local_maximum = fmaxf(
            local_maximum, fabsf(static_cast<float>(input[index])));
    }
    const auto maximum = block_reduce_max(local_maximum, scratch);
    if (threadIdx.x == 0) partial_maxima[blockIdx.x] = maximum;
}

__global__ void fp8_finalize_scale_kernel(
    const float* partial_maxima, float* scale, std::int64_t partial_count,
    float minimum_scale, float maximum_fraction, float quantized_maximum) {
    __shared__ float scratch[kBlockSize];
    float local_maximum = 0.0F;
    for (std::int64_t index = threadIdx.x; index < partial_count;
         index += blockDim.x) {
        local_maximum = fmaxf(local_maximum, partial_maxima[index]);
    }
    const auto maximum = block_reduce_max(local_maximum, scratch);
    if (threadIdx.x == 0) {
        scale[0] = fmaxf(maximum * maximum_fraction / quantized_maximum,
                         minimum_scale);
    }
}

template <typename Input, typename Output>
__global__ void quantize_fp8_device_scale_kernel(
    const Input* input, Output* output, std::int64_t elements,
    const float* scale, float quantized_maximum) {
    const auto index = static_cast<std::int64_t>(blockIdx.x) * blockDim.x +
                       threadIdx.x;
    if (index < elements) {
        const auto value = fminf(
            quantized_maximum,
            fmaxf(-quantized_maximum,
                  static_cast<float>(input[index]) / scale[0]));
        output[index] = static_cast<Output>(value);
    }
}

template <typename Input, typename Output>
__global__ void dequantize_fp8_device_scale_kernel(
    const Input* input, Output* output, std::int64_t elements,
    const float* scale) {
    const auto index = static_cast<std::int64_t>(blockIdx.x) * blockDim.x +
                       threadIdx.x;
    if (index < elements) {
        output[index] = static_cast<Output>(
            static_cast<float>(input[index]) * scale[0]);
    }
}

template <typename Input>
__global__ void fp8_row_scale_kernel(const Input* input, float* scales,
                                     std::int64_t columns,
                                     float minimum_scale) {
    __shared__ float scratch[kBlockSize];
    const auto row = static_cast<std::int64_t>(blockIdx.x);
    float local_maximum = 0.0F;
    for (std::int64_t column = threadIdx.x; column < columns;
         column += blockDim.x) {
        local_maximum = fmaxf(
            local_maximum,
            fabsf(static_cast<float>(input[row * columns + column])));
    }
    const auto maximum = block_reduce_max(local_maximum, scratch);
    if (threadIdx.x == 0) {
        scales[row] = fmaxf(maximum / 240.0F, minimum_scale);
    }
}

template <typename Input, typename Output>
__global__ void quantize_fp8_row_scale_kernel(
    const Input* input, Output* output, std::int64_t elements,
    std::int64_t columns, const float* scales) {
    const auto index = static_cast<std::int64_t>(blockIdx.x) * blockDim.x +
                       threadIdx.x;
    if (index < elements) {
        output[index] = static_cast<Output>(
            static_cast<float>(input[index]) / scales[index / columns]);
    }
}

template <typename Input, typename Output>
__global__ void dequantize_fp8_row_scale_kernel(
    const Input* input, Output* output, std::int64_t elements,
    std::int64_t columns, const float* scales) {
    const auto index = static_cast<std::int64_t>(blockIdx.x) * blockDim.x +
                       threadIdx.x;
    if (index < elements) {
        output[index] = static_cast<Output>(
            static_cast<float>(input[index]) * scales[index / columns]);
    }
}

template <typename Input>
__global__ void fp8_column_scale_kernel(const Input* input, float* scales,
                                        std::int64_t rows,
                                        std::int64_t columns,
                                        float minimum_scale) {
    __shared__ float scratch[kBlockSize];
    const auto column = static_cast<std::int64_t>(blockIdx.x);
    float local_maximum = 0.0F;
    for (std::int64_t row = threadIdx.x; row < rows; row += blockDim.x) {
        local_maximum = fmaxf(
            local_maximum,
            fabsf(static_cast<float>(input[row * columns + column])));
    }
    const auto maximum = block_reduce_max(local_maximum, scratch);
    if (threadIdx.x == 0) {
        scales[column] = fmaxf(maximum / 240.0F, minimum_scale);
    }
}

template <typename Input, typename Output>
__global__ void quantize_fp8_column_scale_kernel(
    const Input* input, Output* output, std::int64_t elements,
    std::int64_t columns, const float* scales) {
    const auto index = static_cast<std::int64_t>(blockIdx.x) * blockDim.x +
                       threadIdx.x;
    if (index < elements) {
        output[index] = static_cast<Output>(
            static_cast<float>(input[index]) / scales[index % columns]);
    }
}

template <typename Input, typename Output>
__global__ void dequantize_fp8_column_scale_kernel(
    const Input* input, Output* output, std::int64_t elements,
    std::int64_t columns, const float* scales) {
    const auto index = static_cast<std::int64_t>(blockIdx.x) * blockDim.x +
                       threadIdx.x;
    if (index < elements) {
        output[index] = store_from_float<Output>(
            static_cast<float>(input[index]) * scales[index % columns]);
    }
}

template <typename T>
__global__ void scale_columns_by_first_kernel(
    T* values, std::int64_t elements, std::int64_t columns,
    const float* scales) {
    const auto index = static_cast<std::int64_t>(blockIdx.x) * blockDim.x +
                       threadIdx.x;
    if (index < elements) {
        const auto ratio = scales[index % columns] / scales[0];
        values[index] = store_from_float<T>(load_as_float(values[index]) * ratio);
    }
}

__device__ float block_reduce_sum(float value, float* scratch) {
    const auto lane = static_cast<unsigned>(threadIdx.x);
    scratch[lane] = value;
    __syncthreads();
    for (unsigned stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (lane < stride) scratch[lane] += scratch[lane + stride];
        __syncthreads();
    }
    const auto result = scratch[0];
    __syncthreads();
    return result;
}

__device__ int block_reduce_sum_int(int value, int* scratch) {
    const auto lane = static_cast<unsigned>(threadIdx.x);
    scratch[lane] = value;
    __syncthreads();
    for (unsigned stride = kBlockSize / 2; stride > 0; stride >>= 1) {
        if (lane < stride) scratch[lane] += scratch[lane + stride];
        __syncthreads();
    }
    const auto result = scratch[0];
    __syncthreads();
    return result;
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

__global__ void argmax_last_dim_kernel(const float* input, std::int32_t* output,
                                       std::int64_t rows,
                                       std::int64_t classes) {
    const auto row = static_cast<std::int64_t>(blockIdx.x);
    if (row >= rows) return;
    __shared__ float value_scratch[kBlockSize];
    __shared__ std::int64_t index_scratch[kBlockSize];
    __shared__ int invalid_scratch[kBlockSize];
    const auto base = row * classes;
    float best_value = -INFINITY;
    std::int64_t best_index = 0;
    int invalid = 0;
    for (std::int64_t column = threadIdx.x; column < classes;
         column += blockDim.x) {
        const auto value = input[base + column];
        if (!isfinite(value)) {
            invalid = 1;
        } else if (value > best_value ||
                   (value == best_value && column < best_index)) {
            best_value = value;
            best_index = column;
        }
    }
    block_reduce_argmax(best_value, best_index, invalid, value_scratch,
                        index_scratch, invalid_scratch);
    if (threadIdx.x == 0) {
        output[row] = invalid != 0 ? -1 : static_cast<std::int32_t>(best_index);
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

__global__ void rms_norm_weight_gradient_rows_kernel(
    const float* input, const float* gradient, const float* row_inverse_rms,
    float* weight_gradient, std::int64_t rows, std::int64_t width) {
    const auto column = static_cast<std::int64_t>(blockIdx.x);
    if (column >= width) return;
    float local_total = 0.0F;
    for (std::int64_t row = threadIdx.x; row < rows; row += blockDim.x) {
        const auto index = row * width + column;
        local_total += gradient[index] * input[index] * row_inverse_rms[row];
    }
    __shared__ float reduction[kBlockSize];
    const auto total = block_reduce_sum(local_total, reduction);
    if (threadIdx.x == 0) weight_gradient[column] = total;
}

template <typename Cache>
__global__ void kv_cache_store_kernel(const float* current, Cache* cache,
                                      std::int64_t batches, std::int64_t heads,
                                      std::int64_t capacity,
                                      std::int64_t width, std::int64_t position) {
    const auto index = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= batches * heads * width) return;
    const auto batch_head = index / width;
    const auto batch = batch_head / heads;
    const auto head = batch_head % heads;
    const auto column = index % width;
    cache[((batch * heads + head) * capacity + position) * width + column] =
        Cache(current[index]);
}

template <typename Cache>
__global__ void kv_cache_store_pair_kernel(
    const float* current_key, const float* current_value,
    Cache* key_cache, Cache* value_cache, std::int64_t batches,
    std::int64_t heads,
    std::int64_t capacity, std::int64_t width, std::int64_t position) {
    const auto index = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= batches * heads * width) return;
    const auto batch_head = index / width;
    const auto batch = batch_head / heads;
    const auto head = batch_head % heads;
    const auto column = index % width;
    const auto destination =
        ((batch * heads + head) * capacity + position) * width + column;
    key_cache[destination] = Cache(current_key[index]);
    value_cache[destination] = Cache(current_value[index]);
}

template <typename Cache>
__global__ void kv_cache_store_pair_positions_kernel(
    const float* current_key, const float* current_value,
    const std::int32_t* positions, const std::int32_t* cache_rows,
    Cache* key_cache, Cache* value_cache, std::int64_t active_batches,
    std::int64_t cache_batches, std::int64_t heads,
    std::int64_t capacity, std::int64_t logical_prefix,
    std::int64_t width) {
    const auto index = static_cast<std::int64_t>(blockIdx.x) * blockDim.x +
                       threadIdx.x;
    if (index >= active_batches * heads * width) return;
    const auto active_head = index / width;
    const auto active = active_head / heads;
    const auto head = active_head % heads;
    const auto column = index % width;
    const auto row = static_cast<std::int64_t>(cache_rows[active]);
    const auto position = static_cast<std::int64_t>(positions[active]);
    if (row < 0 || row >= cache_batches || position < 0 ||
        position >= logical_prefix || position >= capacity) return;
    const auto destination =
        ((row * heads + head) * capacity + position) * width + column;
    key_cache[destination] = Cache(current_key[index]);
    value_cache[destination] = Cache(current_value[index]);
}

template <typename Cache>
__global__ void cached_attention_scores_kernel(
    const float* query, const Cache* key_cache, float* scores,
    std::int64_t batches, std::int64_t heads, std::int64_t sequence,
    std::int64_t cache_batch_stride, std::int64_t cache_head_stride,
    std::int64_t width,
    std::int64_t repeats, float scale) {
    const auto index = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= batches * heads * sequence) return;
    const auto batch_head = index / sequence;
    const auto batch = batch_head / heads;
    const auto head = batch_head % heads;
    const auto position = index % sequence;
    const auto kv_head = head / repeats;
    float dot = 0.0F;
    const auto query_base = (batch * heads + head) * width;
    const auto cache_base = batch * cache_batch_stride +
                            kv_head * cache_head_stride + position * width;
    for (std::int64_t column = 0; column < width; ++column) {
        dot += query[query_base + column] *
               static_cast<float>(key_cache[cache_base + column]);
    }
    scores[index] = dot * scale;
}

template <typename Cache>
__global__ void cached_attention_context_kernel(
    const float* probabilities, const Cache* value_cache, float* output,
    std::int64_t batches, std::int64_t heads, std::int64_t sequence,
    std::int64_t cache_batch_stride, std::int64_t cache_head_stride,
    std::int64_t width,
    std::int64_t repeats) {
    const auto index = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= batches * heads * width) return;
    const auto batch_head = index / width;
    const auto batch = batch_head / heads;
    const auto head = batch_head % heads;
    const auto column = index % width;
    const auto kv_head = head / repeats;
    float total = 0.0F;
    for (std::int64_t position = 0; position < sequence; ++position) {
        total += probabilities[(batch * heads + head) * sequence + position] *
                 static_cast<float>(value_cache[batch * cache_batch_stride +
                                                kv_head * cache_head_stride +
                                                position * width + column]);
    }
    output[index] = total;
}

template <typename Cache, unsigned PhysicalThreads>
__global__ void cached_attention_finalize_scores_kernel(
    const float* scores, const Cache* value_cache, float* output,
    std::int64_t batches, std::int64_t heads, std::int64_t sequence,
    std::int64_t cache_batch_stride, std::int64_t cache_head_stride,
    std::int64_t width, std::int64_t repeats) {
    const auto instance = static_cast<std::int64_t>(blockIdx.x);
    const auto batch = instance / heads;
    const auto head = instance % heads;
    if (batch >= batches) return;
    const auto kv_head = head / repeats;
    extern __shared__ float shared_scores[];
    __shared__ float reduction[kBlockSize];

    const auto score_base = (batch * heads + head) * sequence;
    // PhysicalThreads may be smaller than the historical 256-thread block, but
    // each physical lane owns the same logical 256-lane streams.  The shared
    // reduction below therefore performs the exact old tree.  This separates a
    // scheduling experiment from a numerical-order experiment.
    for (unsigned logical_lane = threadIdx.x; logical_lane < kBlockSize;
         logical_lane += PhysicalThreads) {
        float local_maximum = -INFINITY;
        for (std::int64_t position = logical_lane; position < sequence;
             position += kBlockSize) {
            const auto score = scores[score_base + position];
            shared_scores[position] = score;
            local_maximum = fmaxf(local_maximum, score);
        }
        reduction[logical_lane] = local_maximum;
    }
    __syncthreads();
    for (unsigned stride = kBlockSize / 2; stride > 0; stride >>= 1) {
        for (unsigned logical_lane = threadIdx.x; logical_lane < stride;
             logical_lane += PhysicalThreads) {
            reduction[logical_lane] = fmaxf(
                reduction[logical_lane], reduction[logical_lane + stride]);
        }
        __syncthreads();
    }
    const auto maximum = reduction[0];
    __syncthreads();
    for (unsigned logical_lane = threadIdx.x; logical_lane < kBlockSize;
         logical_lane += PhysicalThreads) {
        float local_denominator = 0.0F;
        for (std::int64_t position = logical_lane; position < sequence;
             position += kBlockSize) {
            const auto probability = expf(shared_scores[position] - maximum);
            shared_scores[position] = probability;
            local_denominator += probability;
        }
        reduction[logical_lane] = local_denominator;
    }
    __syncthreads();
    for (unsigned stride = kBlockSize / 2; stride > 0; stride >>= 1) {
        for (unsigned logical_lane = threadIdx.x; logical_lane < stride;
             logical_lane += PhysicalThreads) {
            reduction[logical_lane] += reduction[logical_lane + stride];
        }
        __syncthreads();
    }
    const auto denominator = reduction[0];
    __syncthreads();
    const auto cache_base = batch * cache_batch_stride +
                            kv_head * cache_head_stride;
    for (std::int64_t column = threadIdx.x; column < width;
         column += PhysicalThreads) {
        float total = 0.0F;
        for (std::int64_t position = 0; position < sequence; ++position) {
            total += shared_scores[position] / denominator *
                     static_cast<float>(value_cache[
                         cache_base + position * width + column]);
        }
        output[(batch * heads + head) * width + column] = total;
    }
}

__global__ void cached_attention_exact_softmax_kernel(
    const float* scores, float* probabilities, std::int64_t batches,
    std::int64_t heads, std::int64_t sequence) {
    const auto instance = static_cast<std::int64_t>(blockIdx.x);
    if (instance >= batches * heads) return;
    extern __shared__ float shared_scores[];
    __shared__ float reduction[kBlockSize];
    const auto score_base = instance * sequence;
    float local_maximum = -INFINITY;
    for (std::int64_t position = threadIdx.x; position < sequence;
         position += kBlockSize) {
        const auto score = scores[score_base + position];
        shared_scores[position] = score;
        local_maximum = fmaxf(local_maximum, score);
    }
    const auto maximum = block_reduce_max(local_maximum, reduction);
    float local_denominator = 0.0F;
    for (std::int64_t position = threadIdx.x; position < sequence;
         position += kBlockSize) {
        const auto probability = expf(shared_scores[position] - maximum);
        shared_scores[position] = probability;
        local_denominator += probability;
    }
    const auto denominator = block_reduce_sum(local_denominator, reduction);
    for (std::int64_t position = threadIdx.x; position < sequence;
         position += kBlockSize) {
        probabilities[score_base + position] =
            shared_scores[position] / denominator;
    }
}

template <typename Cache>
__global__ void cached_attention_split_pv_partial_kernel(
    const float* probabilities, const Cache* value_cache, float* partials,
    std::int64_t batches, std::int64_t heads, std::int64_t sequence,
    std::int64_t cache_batch_stride, std::int64_t cache_head_stride,
    std::int64_t width, std::int64_t repeats, std::int64_t splits) {
    const auto block = static_cast<std::int64_t>(blockIdx.x);
    const auto split = block % splits;
    const auto instance = block / splits;
    const auto batch = instance / heads;
    const auto head = instance % heads;
    if (batch >= batches) return;
    const auto kv_head = head / repeats;
    const auto begin = sequence * split / splits;
    const auto end = sequence * (split + 1) / splits;
    const auto score_base = instance * sequence;
    const auto cache_base = batch * cache_batch_stride +
                            kv_head * cache_head_stride;
    const auto partial_base = block * width;
    for (std::int64_t column = threadIdx.x; column < width;
         column += blockDim.x) {
        float total = 0.0F;
        for (auto position = begin; position < end; ++position) {
            total += probabilities[score_base + position] *
                     static_cast<float>(value_cache[
                         cache_base + position * width + column]);
        }
        partials[partial_base + column] = total;
    }
}

__global__ void cached_attention_split_pv_combine_kernel(
    const float* partials, float* output, std::int64_t elements,
    std::int64_t width, std::int64_t splits) {
    const auto index = static_cast<std::int64_t>(blockIdx.x) * blockDim.x +
                       threadIdx.x;
    if (index >= elements) return;
    const auto instance = index / width;
    const auto column = index % width;
    float total = 0.0F;
    for (std::int64_t split = 0; split < splits; ++split) {
        total += partials[(instance * splits + split) * width + column];
    }
    output[index] = total;
}

template <typename Cache, int Repeats>
__global__ void cached_attention_gqa_value_reuse_kernel(
    const float* probabilities, const Cache* value_cache, float* output,
    std::int64_t batches, std::int64_t heads, std::int64_t kv_heads,
    std::int64_t sequence, std::int64_t cache_batch_stride,
    std::int64_t cache_head_stride, std::int64_t width,
    std::int64_t tile_columns) {
    const auto tiles = (width + tile_columns - 1) / tile_columns;
    const auto block = static_cast<std::int64_t>(blockIdx.x);
    const auto tile = block % tiles;
    const auto batch_kv = block / tiles;
    const auto batch = batch_kv / kv_heads;
    const auto kv_head = batch_kv % kv_heads;
    const auto column = tile * tile_columns + threadIdx.x;
    if (batch >= batches || column >= width || threadIdx.x >= tile_columns) {
        return;
    }
    float totals[Repeats] = {};
    const auto cache_base = batch * cache_batch_stride +
                            kv_head * cache_head_stride;
    for (std::int64_t position = 0; position < sequence; ++position) {
        const auto value = static_cast<float>(value_cache[
            cache_base + position * width + column]);
#pragma unroll
        for (int repeat = 0; repeat < Repeats; ++repeat) {
            const auto head = kv_head * Repeats + repeat;
            totals[repeat] += probabilities[
                (batch * heads + head) * sequence + position] * value;
        }
    }
#pragma unroll
    for (int repeat = 0; repeat < Repeats; ++repeat) {
        const auto head = kv_head * Repeats + repeat;
        output[(batch * heads + head) * width + column] = totals[repeat];
    }
}

template <typename Cache>
__global__ void cached_attention_fused_kernel(
    const float* query, const Cache* key_cache, const Cache* value_cache,
    float* output, std::int64_t batches, std::int64_t heads,
    std::int64_t sequence, std::int64_t cache_batch_stride,
    std::int64_t cache_head_stride, std::int64_t width,
    std::int64_t repeats, float scale) {
    const auto instance = static_cast<std::int64_t>(blockIdx.x);
    const auto batch = instance / heads;
    const auto head = instance % heads;
    if (batch >= batches) return;
    const auto kv_head = head / repeats;
    extern __shared__ float shared_scores[];
    __shared__ float reduction[kBlockSize];

    float local_maximum = -INFINITY;
    for (std::int64_t position = threadIdx.x; position < sequence;
         position += blockDim.x) {
        float dot = 0.0F;
        const auto query_base = (batch * heads + head) * width;
        const auto cache_base = batch * cache_batch_stride +
                                kv_head * cache_head_stride + position * width;
        for (std::int64_t column = 0; column < width; ++column) {
            dot += query[query_base + column] *
                   static_cast<float>(key_cache[cache_base + column]);
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
                     static_cast<float>(value_cache[
                         batch * cache_batch_stride + kv_head * cache_head_stride +
                         position * width + column]);
        }
        output[(batch * heads + head) * width + column] = total;
    }
}

template <typename Cache>
__global__ void cached_attention_split_partial_kernel(
    const float* query, const Cache* key_cache, const Cache* value_cache,
    float* partial_stats, float* partial_numerators,
    std::int64_t batches, std::int64_t heads, std::int64_t sequence,
    std::int64_t cache_batch_stride, std::int64_t cache_head_stride,
    std::int64_t width, std::int64_t repeats, float scale,
    std::int64_t splits) {
    const auto partial = static_cast<std::int64_t>(blockIdx.x);
    const auto split = partial % splits;
    const auto instance = partial / splits;
    const auto batch = instance / heads;
    const auto head = instance % heads;
    if (batch >= batches) return;
    const auto begin = sequence * split / splits;
    const auto end = sequence * (split + 1) / splits;
    const auto kv_head = head / repeats;
    const auto query_base = (batch * heads + head) * width;
    const auto cache_base = batch * cache_batch_stride +
                            kv_head * cache_head_stride;
    extern __shared__ float shared_scores[];
    __shared__ float reduction[kBlockSize];

    float local_maximum = -INFINITY;
    for (std::int64_t position = begin + threadIdx.x; position < end;
         position += blockDim.x) {
        float dot = 0.0F;
        for (std::int64_t column = 0; column < width; ++column) {
            dot += query[query_base + column] *
                   static_cast<float>(
                       key_cache[cache_base + position * width + column]);
        }
        const auto score = dot * scale;
        shared_scores[position - begin] = score;
        local_maximum = fmaxf(local_maximum, score);
    }
    const auto maximum = block_reduce_max(local_maximum, reduction);
    float local_denominator = 0.0F;
    for (std::int64_t position = begin + threadIdx.x; position < end;
         position += blockDim.x) {
        const auto exponential =
            expf(shared_scores[position - begin] - maximum);
        shared_scores[position - begin] = exponential;
        local_denominator += exponential;
    }
    const auto denominator = block_reduce_sum(local_denominator, reduction);
    if (threadIdx.x == 0) {
        partial_stats[partial * 2] = maximum;
        partial_stats[partial * 2 + 1] = denominator;
    }
    for (std::int64_t column = threadIdx.x; column < width;
         column += blockDim.x) {
        float numerator = 0.0F;
        for (std::int64_t position = begin; position < end; ++position) {
            numerator += shared_scores[position - begin] *
                         static_cast<float>(value_cache[
                             cache_base + position * width + column]);
        }
        partial_numerators[partial * width + column] = numerator;
    }
}

__global__ void cached_attention_split_combine_kernel(
    const float* partial_stats, const float* partial_numerators,
    float* output, std::int64_t instances, std::int64_t width,
    std::int64_t splits) {
    const auto instance = static_cast<std::int64_t>(blockIdx.x);
    if (instance >= instances) return;
    __shared__ float reduction[kBlockSize];
    const auto split = static_cast<std::int64_t>(threadIdx.x);
    const auto local_maximum = split < splits
        ? partial_stats[(instance * splits + split) * 2]
        : -INFINITY;
    const auto maximum = block_reduce_max(local_maximum, reduction);
    const auto local_denominator = split < splits
        ? partial_stats[(instance * splits + split) * 2 + 1] *
              expf(local_maximum - maximum)
        : 0.0F;
    const auto denominator = block_reduce_sum(local_denominator, reduction);
    for (std::int64_t column = threadIdx.x; column < width;
         column += blockDim.x) {
        float numerator = 0.0F;
        for (std::int64_t partial_index = 0; partial_index < splits;
             ++partial_index) {
            const auto partial = instance * splits + partial_index;
            const auto weight = expf(partial_stats[partial * 2] - maximum);
            numerator += partial_numerators[partial * width + column] * weight;
        }
        output[instance * width + column] = numerator / denominator;
    }
}

template <typename Cache>
__global__ void cached_attention_fused_positions_kernel(
    const float* query, const Cache* key_cache, const Cache* value_cache,
    const std::int32_t* positions, const std::int32_t* cache_rows,
    float* output, std::int64_t active_batches, std::int64_t cache_batches,
    std::int64_t heads, std::int64_t logical_prefix,
    std::int64_t cache_batch_stride, std::int64_t cache_head_stride,
    std::int64_t width, std::int64_t repeats, float scale) {
    const auto instance = static_cast<std::int64_t>(blockIdx.x);
    const auto active = instance / heads;
    const auto head = instance % heads;
    if (active >= active_batches) return;
    const auto row = static_cast<std::int64_t>(cache_rows[active]);
    const auto visible = static_cast<std::int64_t>(positions[active]) + 1;
    if (row < 0 || row >= cache_batches || visible <= 0 ||
        visible > logical_prefix) {
        for (std::int64_t column = threadIdx.x; column < width;
             column += blockDim.x) {
            output[(active * heads + head) * width + column] = 0.0F;
        }
        return;
    }
    const auto kv_head = head / repeats;
    extern __shared__ float shared_scores[];
    __shared__ float reduction[kBlockSize];
    float local_maximum = -INFINITY;
    for (std::int64_t position = threadIdx.x; position < visible;
         position += blockDim.x) {
        float dot = 0.0F;
        const auto query_base = (active * heads + head) * width;
        const auto cache_base = row * cache_batch_stride +
                                kv_head * cache_head_stride + position * width;
        for (std::int64_t column = 0; column < width; ++column) {
            dot += query[query_base + column] *
                   static_cast<float>(key_cache[cache_base + column]);
        }
        const auto score = dot * scale;
        shared_scores[position] = score;
        local_maximum = fmaxf(local_maximum, score);
    }
    const auto maximum = block_reduce_max(local_maximum, reduction);
    float local_denominator = 0.0F;
    for (std::int64_t position = threadIdx.x; position < visible;
         position += blockDim.x) {
        const auto probability = expf(shared_scores[position] - maximum);
        shared_scores[position] = probability;
        local_denominator += probability;
    }
    const auto denominator = block_reduce_sum(local_denominator, reduction);
    for (std::int64_t column = threadIdx.x; column < width;
         column += blockDim.x) {
        float total = 0.0F;
        for (std::int64_t position = 0; position < visible; ++position) {
            total += shared_scores[position] / denominator *
                     static_cast<float>(value_cache[
                         row * cache_batch_stride + kv_head * cache_head_stride +
                         position * width + column]);
        }
        output[(active * heads + head) * width + column] = total;
    }
}

template <typename Cache>
__global__ void cached_attention_scores_positions_kernel(
    const float* query, const Cache* key_cache,
    const std::int32_t* positions, const std::int32_t* cache_rows,
    float* scores, std::int64_t active_batches, std::int64_t cache_batches,
    std::int64_t heads, std::int64_t logical_prefix,
    std::int64_t cache_batch_stride, std::int64_t cache_head_stride,
    std::int64_t width, std::int64_t repeats, float scale) {
    const auto index = static_cast<std::int64_t>(blockIdx.x) * blockDim.x +
                       threadIdx.x;
    if (index >= active_batches * heads * logical_prefix) return;
    const auto active_head = index / logical_prefix;
    const auto active = active_head / heads;
    const auto head = active_head % heads;
    const auto position = index % logical_prefix;
    const auto row = static_cast<std::int64_t>(cache_rows[active]);
    const auto visible = static_cast<std::int64_t>(positions[active]) + 1;
    if (row < 0 || row >= cache_batches || visible <= 0 ||
        visible > logical_prefix || position >= visible) {
        scores[index] = -INFINITY;
        return;
    }
    const auto kv_head = head / repeats;
    float dot = 0.0F;
    const auto query_base = (active * heads + head) * width;
    const auto cache_base = row * cache_batch_stride +
                            kv_head * cache_head_stride + position * width;
    for (std::int64_t column = 0; column < width; ++column) {
        dot += query[query_base + column] *
               static_cast<float>(key_cache[cache_base + column]);
    }
    scores[index] = dot * scale;
}

template <typename Cache>
__global__ void cached_attention_context_positions_kernel(
    const float* probabilities, const Cache* value_cache,
    const std::int32_t* positions, const std::int32_t* cache_rows,
    float* output, std::int64_t active_batches, std::int64_t cache_batches,
    std::int64_t heads, std::int64_t logical_prefix,
    std::int64_t cache_batch_stride, std::int64_t cache_head_stride,
    std::int64_t width, std::int64_t repeats) {
    const auto index = static_cast<std::int64_t>(blockIdx.x) * blockDim.x +
                       threadIdx.x;
    if (index >= active_batches * heads * width) return;
    const auto active_head = index / width;
    const auto active = active_head / heads;
    const auto head = active_head % heads;
    const auto column = index % width;
    const auto row = static_cast<std::int64_t>(cache_rows[active]);
    const auto visible = static_cast<std::int64_t>(positions[active]) + 1;
    if (row < 0 || row >= cache_batches || visible <= 0 ||
        visible > logical_prefix) {
        output[index] = 0.0F;
        return;
    }
    const auto kv_head = head / repeats;
    float total = 0.0F;
    for (std::int64_t position = 0; position < visible; ++position) {
        total += probabilities[(active * heads + head) * logical_prefix +
                               position] *
                 static_cast<float>(value_cache[
                     row * cache_batch_stride + kv_head * cache_head_stride +
                     position * width + column]);
    }
    output[index] = total;
}

__global__ void causal_gqa_attention_kernel(
    const float* query, const float* key, const float* value, float* output,
    float* saved_probabilities,
    std::int64_t batches, std::int64_t heads, std::int64_t kv_heads,
    std::int64_t sequence, std::int64_t width, std::int64_t repeats,
    float scale) {
    const auto row = static_cast<std::int64_t>(blockIdx.x);
    const auto query_position = row % sequence;
    const auto head_and_batch = row / sequence;
    const auto head = head_and_batch % heads;
    const auto batch = head_and_batch / heads;
    if (batch >= batches) return;
    const auto kv_head = head / repeats;
    const auto query_base = ((batch * heads + head) * sequence + query_position) * width;
    const auto kv_base = (batch * kv_heads + kv_head) * sequence * width;
    const auto visible = query_position + 1;
    extern __shared__ float shared[];
    auto* attention_probabilities = shared;
    auto* reduction = shared + sequence;

    float local_maximum = -INFINITY;
    for (std::int64_t position = threadIdx.x; position < visible;
         position += blockDim.x) {
        float dot = 0.0F;
        for (std::int64_t column = 0; column < width; ++column) {
            dot += query[query_base + column] *
                   key[kv_base + position * width + column];
        }
        const auto score = dot * scale;
        attention_probabilities[position] = score;
        local_maximum = fmaxf(local_maximum, score);
    }
    const auto maximum = block_reduce_max(local_maximum, reduction);
    float local_denominator = 0.0F;
    for (std::int64_t position = threadIdx.x; position < visible;
         position += blockDim.x) {
        const auto probability = expf(attention_probabilities[position] - maximum);
        attention_probabilities[position] = probability;
        local_denominator += probability;
    }
    const auto denominator = block_reduce_sum(local_denominator, reduction);
    for (std::int64_t position = threadIdx.x; position < visible;
         position += blockDim.x) {
        attention_probabilities[position] /= denominator;
        if (saved_probabilities != nullptr) {
            saved_probabilities[row * sequence + position] =
                attention_probabilities[position];
        }
    }
    __syncthreads();
    for (std::int64_t column = threadIdx.x; column < width; column += blockDim.x) {
        float total = 0.0F;
        for (std::int64_t position = 0; position < visible; ++position) {
            total += attention_probabilities[position] *
                     value[kv_base + position * width + column];
        }
        output[query_base + column] = total;
    }
}

__global__ void causal_gqa_attention_backward_kernel(
    const float* query, const float* key, const float* value,
    const float* output_gradient, float* query_gradient,
    float* key_gradient, float* value_gradient,
    std::int64_t batches, std::int64_t heads, std::int64_t kv_heads,
    std::int64_t sequence, std::int64_t width, std::int64_t repeats,
    float scale) {
    const auto row = static_cast<std::int64_t>(blockIdx.x);
    const auto query_position = row % sequence;
    const auto head_and_batch = row / sequence;
    const auto head = head_and_batch % heads;
    const auto batch = head_and_batch / heads;
    if (batch >= batches) return;
    const auto kv_head = head / repeats;
    const auto query_base = ((batch * heads + head) * sequence + query_position) * width;
    const auto kv_base = (batch * kv_heads + kv_head) * sequence * width;
    const auto visible = query_position + 1;
    extern __shared__ float shared[];
    auto* backward_probabilities = shared;
    auto* probability_gradients = shared + sequence;
    __shared__ float reduction[kBlockSize];

    float local_maximum = -INFINITY;
    for (std::int64_t position = threadIdx.x; position < visible;
         position += blockDim.x) {
        float dot = 0.0F;
        for (std::int64_t column = 0; column < width; ++column) {
            dot += query[query_base + column] *
                   key[kv_base + position * width + column];
        }
        const auto score = dot * scale;
        backward_probabilities[position] = score;
        local_maximum = fmaxf(local_maximum, score);
    }
    const auto maximum = block_reduce_max(local_maximum, reduction);
    float local_denominator = 0.0F;
    for (std::int64_t position = threadIdx.x; position < visible;
         position += blockDim.x) {
        const auto probability = expf(backward_probabilities[position] - maximum);
        backward_probabilities[position] = probability;
        local_denominator += probability;
    }
    const auto denominator = block_reduce_sum(local_denominator, reduction);
    float local_weighted_gradient = 0.0F;
    for (std::int64_t position = threadIdx.x; position < visible;
         position += blockDim.x) {
        const auto probability = backward_probabilities[position] / denominator;
        backward_probabilities[position] = probability;
        float probability_gradient = 0.0F;
        for (std::int64_t column = 0; column < width; ++column) {
            probability_gradient += output_gradient[query_base + column] *
                                    value[kv_base + position * width + column];
        }
        probability_gradients[position] = probability_gradient;
        local_weighted_gradient += probability * probability_gradient;
    }
    const auto weighted_gradient = block_reduce_sum(local_weighted_gradient, reduction);
    __syncthreads();

    for (std::int64_t column = threadIdx.x; column < width; column += blockDim.x) {
        float total = 0.0F;
        for (std::int64_t position = 0; position < visible; ++position) {
            const auto score_gradient = backward_probabilities[position] *
                (probability_gradients[position] - weighted_gradient);
            total += score_gradient * key[kv_base + position * width + column];
        }
        query_gradient[query_base + column] = total * scale;
    }
    for (std::int64_t index = threadIdx.x; index < visible * width;
         index += blockDim.x) {
        const auto position = index / width;
        const auto column = index % width;
        const auto score_gradient = backward_probabilities[position] *
            (probability_gradients[position] - weighted_gradient);
        atomicAdd(key_gradient + kv_base + position * width + column,
                  score_gradient * scale * query[query_base + column]);
        atomicAdd(value_gradient + kv_base + position * width + column,
                  backward_probabilities[position] *
                      output_gradient[query_base + column]);
    }
}

__global__ void causal_gqa_attention_backward_rows_kernel(
    const float* query, const float* key, const float* value,
    const float* output_gradient, float* query_gradient,
    float* probabilities, float* scaled_score_gradients,
    std::int64_t batches, std::int64_t heads, std::int64_t kv_heads,
    std::int64_t sequence, std::int64_t width, std::int64_t repeats,
    float scale) {
    const auto row = static_cast<std::int64_t>(blockIdx.x);
    const auto query_position = row % sequence;
    const auto head_and_batch = row / sequence;
    const auto head = head_and_batch % heads;
    const auto batch = head_and_batch / heads;
    if (batch >= batches) return;
    const auto kv_head = head / repeats;
    const auto query_base = ((batch * heads + head) * sequence + query_position) * width;
    const auto kv_base = (batch * kv_heads + kv_head) * sequence * width;
    const auto visible = query_position + 1;
    const auto matrix_base = row * sequence;
    extern __shared__ float shared[];
    auto* row_probabilities = shared;
    auto* probability_gradients = shared + sequence;
    __shared__ float reduction[kBlockSize];

    float local_maximum = -INFINITY;
    for (std::int64_t position = threadIdx.x; position < visible;
         position += blockDim.x) {
        float dot = 0.0F;
        for (std::int64_t column = 0; column < width; ++column) {
            dot += query[query_base + column] *
                   key[kv_base + position * width + column];
        }
        const auto score = dot * scale;
        row_probabilities[position] = score;
        local_maximum = fmaxf(local_maximum, score);
    }
    const auto maximum = block_reduce_max(local_maximum, reduction);
    float local_denominator = 0.0F;
    for (std::int64_t position = threadIdx.x; position < visible;
         position += blockDim.x) {
        const auto probability = expf(row_probabilities[position] - maximum);
        row_probabilities[position] = probability;
        local_denominator += probability;
    }
    const auto denominator = block_reduce_sum(local_denominator, reduction);
    float local_weighted_gradient = 0.0F;
    for (std::int64_t position = threadIdx.x; position < visible;
         position += blockDim.x) {
        const auto probability = row_probabilities[position] / denominator;
        row_probabilities[position] = probability;
        float probability_gradient = 0.0F;
        for (std::int64_t column = 0; column < width; ++column) {
            probability_gradient += output_gradient[query_base + column] *
                                    value[kv_base + position * width + column];
        }
        probability_gradients[position] = probability_gradient;
        local_weighted_gradient += probability * probability_gradient;
    }
    const auto weighted_gradient = block_reduce_sum(local_weighted_gradient, reduction);
    __syncthreads();

    for (std::int64_t position = threadIdx.x; position < visible;
         position += blockDim.x) {
        const auto probability = row_probabilities[position];
        const auto score_gradient = probability *
            (probability_gradients[position] - weighted_gradient);
        probabilities[matrix_base + position] = probability;
        scaled_score_gradients[matrix_base + position] = score_gradient * scale;
    }
    for (std::int64_t column = threadIdx.x; column < width; column += blockDim.x) {
        float total = 0.0F;
        for (std::int64_t position = 0; position < visible; ++position) {
            const auto score_gradient = row_probabilities[position] *
                (probability_gradients[position] - weighted_gradient);
            total += score_gradient * key[kv_base + position * width + column];
        }
        query_gradient[query_base + column] = total * scale;
    }
}

__global__ void causal_gqa_attention_backward_saved_rows_kernel(
    const float* key, const float* value, const float* output_gradient,
    const float* probabilities, float* query_gradient,
    float* scaled_score_gradients,
    std::int64_t batches, std::int64_t heads, std::int64_t kv_heads,
    std::int64_t sequence, std::int64_t width, std::int64_t repeats,
    float scale) {
    const auto row = static_cast<std::int64_t>(blockIdx.x);
    const auto query_position = row % sequence;
    const auto head_and_batch = row / sequence;
    const auto head = head_and_batch % heads;
    const auto batch = head_and_batch / heads;
    if (batch >= batches) return;
    const auto kv_head = head / repeats;
    const auto query_base = ((batch * heads + head) * sequence + query_position) * width;
    const auto kv_base = (batch * kv_heads + kv_head) * sequence * width;
    const auto visible = query_position + 1;
    const auto matrix_base = row * sequence;
    extern __shared__ float probability_gradients[];
    __shared__ float reduction[kBlockSize];

    float local_weighted_gradient = 0.0F;
    for (std::int64_t position = threadIdx.x; position < visible;
         position += blockDim.x) {
        float probability_gradient = 0.0F;
        for (std::int64_t column = 0; column < width; ++column) {
            probability_gradient += output_gradient[query_base + column] *
                                    value[kv_base + position * width + column];
        }
        probability_gradients[position] = probability_gradient;
        local_weighted_gradient += probabilities[matrix_base + position] *
                                   probability_gradient;
    }
    const auto weighted_gradient = block_reduce_sum(local_weighted_gradient, reduction);
    __syncthreads();

    for (std::int64_t position = threadIdx.x; position < visible;
         position += blockDim.x) {
        const auto score_gradient = probabilities[matrix_base + position] *
            (probability_gradients[position] - weighted_gradient);
        scaled_score_gradients[matrix_base + position] = score_gradient * scale;
    }
    for (std::int64_t column = threadIdx.x; column < width; column += blockDim.x) {
        float total = 0.0F;
        for (std::int64_t position = 0; position < visible; ++position) {
            const auto score_gradient = probabilities[matrix_base + position] *
                (probability_gradients[position] - weighted_gradient);
            total += score_gradient * key[kv_base + position * width + column];
        }
        query_gradient[query_base + column] = total * scale;
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

__global__ void swiglu_backward_scalar_seed_kernel(
    const float* gate, const float* up, const float* scalar_gradient,
    float* gate_gradient, float* up_gradient, std::int64_t elements) {
    const auto index = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= elements) return;
    const auto probability = sigmoid_device(gate[index]);
    const auto silu_value = gate[index] * probability;
    const auto gradient = scalar_gradient[0];
    gate_gradient[index] = gradient * up[index] * probability *
                           (1.0F + gate[index] * (1.0F - probability));
    up_gradient[index] = gradient * silu_value;
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

__global__ void rope_split_half_bias_bthd_backward_kernel(
    const float* gradient, float* input_gradient, std::int64_t pairs,
    std::int64_t sequence_size, std::int64_t heads, std::int64_t head_width,
    std::int64_t position_offset, float base) {
    const auto pair_index = static_cast<std::int64_t>(blockIdx.x) * blockDim.x +
                            threadIdx.x;
    if (pair_index >= pairs) return;
    const auto half = head_width / 2;
    const auto instance = pair_index / half;
    const auto pair = pair_index % half;
    const auto heads_per_batch = heads * sequence_size;
    const auto batch = instance / heads_per_batch;
    const auto within_batch = instance % heads_per_batch;
    const auto head = within_batch / sequence_size;
    const auto position = within_batch % sequence_size;
    const auto gradient_row = instance * head_width;
    const auto input_row = ((batch * sequence_size + position) * heads + head) *
                           head_width;
    const auto gradient_first = gradient_row + pair;
    const auto gradient_second = gradient_first + half;
    const auto input_first = input_row + pair;
    const auto input_second = input_first + half;
    const auto angle = static_cast<float>(position + position_offset) *
                       powf(base, -2.0F * static_cast<float>(pair) /
                                      static_cast<float>(head_width));
    float sine = 0.0F;
    float cosine = 0.0F;
    sincosf(angle, &sine, &cosine);
    input_gradient[input_first] =
        gradient[gradient_first] * cosine + gradient[gradient_second] * sine;
    input_gradient[input_second] =
        -gradient[gradient_first] * sine + gradient[gradient_second] * cosine;
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

__global__ void causal_softmax_rows_kernel(const float* scores, float* output,
                                           std::int64_t rows,
                                           std::int64_t sequence) {
    const auto row = static_cast<std::int64_t>(blockIdx.x);
    if (row >= rows) return;
    const auto visible = row % sequence;
    const auto base = row * sequence;
    __shared__ float reduction[kBlockSize];
    float local_maximum = -INFINITY;
    for (std::int64_t column = threadIdx.x; column <= visible;
         column += blockDim.x) {
        local_maximum = fmaxf(local_maximum, scores[base + column]);
    }
    const auto maximum = block_reduce_max(local_maximum, reduction);
    float local_denominator = 0.0F;
    for (std::int64_t column = threadIdx.x; column <= visible;
         column += blockDim.x) {
        const auto probability = expf(scores[base + column] - maximum);
        output[base + column] = probability;
        local_denominator += probability;
    }
    const auto denominator = block_reduce_sum(local_denominator, reduction);
    for (std::int64_t column = threadIdx.x; column < sequence;
         column += blockDim.x) {
        output[base + column] = column <= visible
                                    ? output[base + column] / denominator
                                    : 0.0F;
    }
}

__global__ void causal_softmax_rows_register_kernel(
    const float* scores, float* output, std::int64_t rows,
    std::int64_t sequence) {
    constexpr unsigned kValuesPerThread = 8;
    const auto row = static_cast<std::int64_t>(blockIdx.x);
    if (row >= rows) return;
    const auto visible = row % sequence;
    const auto base = row * sequence;
    __shared__ float reduction[kBlockSize];
    float local_maximum = -INFINITY;
    for (std::int64_t column = threadIdx.x; column <= visible;
         column += blockDim.x) {
        local_maximum = fmaxf(local_maximum, scores[base + column]);
    }
    const auto maximum = block_reduce_max(local_maximum, reduction);
    float local_probabilities[kValuesPerThread];
    unsigned local_count = 0;
    float local_denominator = 0.0F;
    for (std::int64_t column = threadIdx.x; column <= visible;
         column += blockDim.x) {
        const auto probability = expf(scores[base + column] - maximum);
        local_probabilities[local_count++] = probability;
        local_denominator += probability;
    }
    const auto denominator = block_reduce_sum(local_denominator, reduction);
    local_count = 0;
    for (std::int64_t column = threadIdx.x; column <= visible;
         column += blockDim.x) {
        output[base + column] = local_probabilities[local_count++] / denominator;
    }
    for (std::int64_t column = visible + 1 + threadIdx.x; column < sequence;
         column += blockDim.x) {
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

__global__ void causal_softmax_backward_rows_kernel(
    const float* output, const float* gradient, float* input_gradient,
    std::int64_t rows, std::int64_t sequence) {
    const auto row = static_cast<std::int64_t>(blockIdx.x);
    if (row >= rows) return;
    const auto visible = row % sequence;
    const auto base = row * sequence;
    __shared__ float reduction[kBlockSize];
    float local_dot = 0.0F;
    for (std::int64_t column = threadIdx.x; column <= visible;
         column += blockDim.x) {
        local_dot += gradient[base + column] * output[base + column];
    }
    const auto dot = block_reduce_sum(local_dot, reduction);
    for (std::int64_t column = threadIdx.x; column < sequence;
         column += blockDim.x) {
        input_gradient[base + column] = column <= visible
            ? output[base + column] * (gradient[base + column] - dot)
            : 0.0F;
    }
}

template <typename Input>
__global__ void repeat_interleave_typed_kernel(
    const Input* input, float* output, std::int64_t output_elements,
    std::int64_t repeated_width, std::int64_t inner,
    std::int64_t repeats) {
    const auto index = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= output_elements) return;
    const auto inner_index = index % inner;
    const auto repeated_coordinate = (index / inner) % repeated_width;
    const auto outer = index / (inner * repeated_width);
    const auto input_width = repeated_width / repeats;
    const auto input_index = outer * input_width * inner +
                             (repeated_coordinate / repeats) * inner + inner_index;
    output[index] = load_as_float(input[input_index]);
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

__global__ void repeat_gqa_kv_bthd_kernel(
    const float* key, const float* value, float* expanded_key,
    float* expanded_value, std::int64_t output_elements,
    std::int64_t heads, std::int64_t kv_heads,
    std::int64_t sequence, std::int64_t width, std::int64_t repeats) {
    const auto index = static_cast<std::int64_t>(blockIdx.x) * blockDim.x +
                       threadIdx.x;
    if (index >= output_elements) return;
    const auto column = index % width;
    const auto position = (index / width) % sequence;
    const auto head = (index / (width * sequence)) % heads;
    const auto batch = index / (width * sequence * heads);
    const auto kv_head = head / repeats;
    const auto source_key =
        ((batch * kv_heads + kv_head) * sequence + position) * width + column;
    const auto source_value =
        ((batch * sequence + position) * kv_heads + kv_head) * width + column;
    const auto destination_value =
        ((batch * sequence + position) * heads + head) * width + column;
    expanded_key[index] = key[source_key];
    expanded_value[destination_value] = value[source_value];
}

__global__ void repeat_gqa_kv_bthd_backward_kernel(
    const float* key_gradient, const float* value_gradient,
    float* reduced_key_gradient, float* reduced_value_gradient,
    std::int64_t output_elements, std::int64_t heads,
    std::int64_t kv_heads, std::int64_t sequence,
    std::int64_t width, std::int64_t repeats) {
    const auto index = static_cast<std::int64_t>(blockIdx.x) * blockDim.x +
                       threadIdx.x;
    if (index >= output_elements) return;
    const auto column = index % width;
    const auto position = (index / width) % sequence;
    const auto kv_head = (index / (width * sequence)) % kv_heads;
    const auto batch = index / (width * sequence * kv_heads);
    float key_total = 0.0F;
    float value_total = 0.0F;
    for (std::int64_t repeat = 0; repeat < repeats; ++repeat) {
        const auto head = kv_head * repeats + repeat;
        const auto source_key =
            ((batch * heads + head) * sequence + position) * width + column;
        const auto source_value =
            ((batch * sequence + position) * heads + head) * width + column;
        key_total += key_gradient[source_key];
        value_total += value_gradient[source_value];
    }
    const auto destination_value =
        ((batch * sequence + position) * kv_heads + kv_head) * width + column;
    reduced_key_gradient[index] = key_total;
    reduced_value_gradient[destination_value] = value_total;
}

}  // namespace

void* create_adamw_descriptor_staging(std::size_t bytes) {
    if (bytes == 0) {
        throw std::invalid_argument("AdamW descriptor staging cannot be empty");
    }
    auto* staging = new AdamWDescriptorStaging();
    try {
        check_hip_status(
            hipHostMalloc(&staging->pointer, bytes, hipHostMallocDefault),
            "hipHostMalloc(AdamW descriptors)");
        check_hip_status(
            hipEventCreateWithFlags(&staging->ready, hipEventDisableTiming),
            "hipEventCreate(AdamW descriptors)");
    } catch (...) {
        if (staging->pointer != nullptr) (void)hipHostFree(staging->pointer);
        delete staging;
        throw;
    }
    return staging;
}

void destroy_adamw_descriptor_staging(void* handle) noexcept {
    auto* staging = static_cast<AdamWDescriptorStaging*>(handle);
    if (staging == nullptr) return;
    if (staging->recorded && staging->ready != nullptr) {
        (void)hipEventSynchronize(staging->ready);
    }
    if (staging->ready != nullptr) (void)hipEventDestroy(staging->ready);
    if (staging->pointer != nullptr) (void)hipHostFree(staging->pointer);
    delete staging;
}

void* acquire_adamw_descriptor_staging(void* handle) {
    auto* staging = static_cast<AdamWDescriptorStaging*>(handle);
    if (staging == nullptr) {
        throw std::invalid_argument("AdamW descriptor staging is undefined");
    }
    if (staging->recorded) {
        check_hip_status(hipEventSynchronize(staging->ready),
                         "hipEventSynchronize(AdamW descriptors)");
        staging->recorded = false;
    }
    return staging->pointer;
}

void mark_adamw_descriptor_staging_in_use(void* handle, void* stream) {
    auto* staging = static_cast<AdamWDescriptorStaging*>(handle);
    if (staging == nullptr) {
        throw std::invalid_argument("AdamW descriptor staging is undefined");
    }
    check_hip_status(hipEventRecord(staging->ready, as_stream(stream)),
                     "hipEventRecord(AdamW descriptors)");
    staging->recorded = true;
}

void launch_fill(float* output, std::int64_t elements, float value, void* stream) {
    if (elements == 0) return;
    hipLaunchKernelGGL(fill_kernel, dim3(grid_size(elements)), dim3(kBlockSize), 0,
                       as_stream(stream), output, elements, value);
    check_launch("fill_kernel");
}

void launch_adamw_update(float* parameter, const float* gradient,
                         float* first_moment, float* second_moment,
                         void* bf16_mirror,
                         std::int64_t elements, float learning_rate,
                         float beta1, float beta2, float epsilon,
                         float weight_decay, float first_correction,
                         float second_correction, void* stream) {
    if (elements == 0) return;
    hipLaunchKernelGGL(adamw_update_kernel, dim3(grid_size(elements)),
                       dim3(kBlockSize), 0, as_stream(stream), parameter, gradient,
                       first_moment, second_moment,
                       static_cast<hip_bfloat16*>(bf16_mirror), elements, learning_rate, beta1,
                       beta2, epsilon, weight_decay, first_correction,
                       second_correction);
    check_launch("adamw_update_kernel");
}

void launch_adamw_update_vectorized(float* parameter, const float* gradient,
                                    float* first_moment, float* second_moment,
                                    void* bf16_mirror, std::int64_t elements,
                                    float learning_rate, float beta1, float beta2,
                                    float epsilon, float weight_decay,
                                    float first_correction, float second_correction,
                                    void* stream) {
    if (elements == 0) return;
    constexpr std::int64_t vector_width = 4;
    const auto vectors = (elements + vector_width - 1) / vector_width;
    hipLaunchKernelGGL(adamw_update_vectorized_kernel, dim3(grid_size(vectors)),
                       dim3(kBlockSize), 0, as_stream(stream), parameter, gradient,
                       first_moment, second_moment,
                       static_cast<hip_bfloat16*>(bf16_mirror), elements,
                       learning_rate, beta1, beta2, epsilon, weight_decay,
                       first_correction, second_correction);
    check_launch("adamw_update_vectorized_kernel");
}

void launch_adamw_update_multi(
    const AdamWMultiTensorDescriptor* descriptors,
    const std::int32_t* block_to_tensor, std::int64_t blocks,
    float learning_rate, float beta1, float beta2, float epsilon,
    float weight_decay, float first_correction, float second_correction,
    void* stream) {
    if (blocks == 0) return;
    hipLaunchKernelGGL(
        adamw_update_multi_kernel, dim3(static_cast<unsigned>(blocks)),
        dim3(kBlockSize), 0, as_stream(stream), descriptors, block_to_tensor,
        learning_rate, beta1, beta2, epsilon, weight_decay,
        first_correction, second_correction, nullptr);
    check_launch("adamw_update_multi_kernel");
}

void launch_adamw_update_multi_graph(
    const AdamWMultiTensorDescriptor* descriptors,
    const std::int32_t* block_to_tensor, std::int64_t blocks,
    float learning_rate, float beta1, float beta2, float epsilon,
    float weight_decay, const float* corrections, void* stream) {
    if (blocks == 0) return;
    hipLaunchKernelGGL(
        adamw_update_multi_kernel, dim3(static_cast<unsigned>(blocks)),
        dim3(kBlockSize), 0, as_stream(stream), descriptors, block_to_tensor,
        learning_rate, beta1, beta2, epsilon, weight_decay, 0.0F, 0.0F,
        corrections);
    check_launch("adamw_update_multi_graph_kernel");
}

void launch_adamw_update_bf16_moments(
    float* parameter, const float* gradient, void* first_moment_bf16,
    void* second_moment_bf16, void* parameter_bf16_mirror,
    std::int64_t elements, float learning_rate, float beta1, float beta2,
    float epsilon, float weight_decay, float first_correction,
    float second_correction, void* stream) {
    if (elements == 0) return;
    const auto vector_aligned =
        (reinterpret_cast<std::uintptr_t>(parameter) & 15U) == 0U &&
        (reinterpret_cast<std::uintptr_t>(gradient) & 15U) == 0U;
    if (vector_aligned) {
        const auto vectors = (elements + 3) / 4;
        hipLaunchKernelGGL(
            adamw_update_bf16_moments_vectorized_kernel,
            dim3(grid_size(vectors)), dim3(kBlockSize), 0, as_stream(stream),
            parameter, gradient,
            static_cast<hip_bfloat16*>(first_moment_bf16),
            static_cast<hip_bfloat16*>(second_moment_bf16),
            static_cast<hip_bfloat16*>(parameter_bf16_mirror), elements,
            learning_rate, beta1, beta2, epsilon, weight_decay,
            first_correction, second_correction);
        check_launch("adamw_update_bf16_moments_vectorized_kernel");
        return;
    }
    hipLaunchKernelGGL(
        adamw_update_bf16_moments_kernel, dim3(grid_size(elements)),
        dim3(kBlockSize), 0, as_stream(stream), parameter, gradient,
        static_cast<hip_bfloat16*>(first_moment_bf16),
        static_cast<hip_bfloat16*>(second_moment_bf16),
        static_cast<hip_bfloat16*>(parameter_bf16_mirror), elements,
        learning_rate, beta1, beta2, epsilon, weight_decay,
        first_correction, second_correction);
    check_launch("adamw_update_bf16_moments_kernel");
}

void launch_adamw_graph_advance(std::int32_t* step, float* corrections,
                                float beta1, float beta2, void* stream) {
    hipLaunchKernelGGL(adamw_graph_advance_kernel, dim3(1), dim3(1), 0,
                       as_stream(stream), step, corrections, beta1, beta2);
    check_launch("adamw_graph_advance_kernel");
}

void launch_adamw_update_graph(
    float* parameter, const float* gradient, float* first_moment,
    float* second_moment, void* bf16_mirror, std::int64_t elements,
    float learning_rate, float beta1, float beta2, float epsilon,
    float weight_decay, const float* corrections, bool vectorized,
    void* stream) {
    if (elements == 0) return;
    if (vectorized) {
        const auto vectors = (elements + 3) / 4;
        hipLaunchKernelGGL(
            adamw_update_graph_vectorized_kernel, dim3(grid_size(vectors)),
            dim3(kBlockSize), 0, as_stream(stream), parameter, gradient,
            first_moment, second_moment,
            static_cast<hip_bfloat16*>(bf16_mirror), elements, learning_rate,
            beta1, beta2, epsilon, weight_decay, corrections);
        check_launch("adamw_update_graph_vectorized_kernel");
        return;
    }
    hipLaunchKernelGGL(
        adamw_update_graph_kernel, dim3(grid_size(elements)), dim3(kBlockSize),
        0, as_stream(stream), parameter, gradient, first_moment, second_moment,
        static_cast<hip_bfloat16*>(bf16_mirror), elements, learning_rate,
        beta1, beta2, epsilon, weight_decay, corrections);
    check_launch("adamw_update_graph_kernel");
}

void launch_adamw_update_bf16_moments_graph(
    float* parameter, const float* gradient, void* first_moment_bf16,
    void* second_moment_bf16, void* parameter_bf16_mirror,
    std::int64_t elements, float learning_rate, float beta1, float beta2,
    float epsilon, float weight_decay, const float* corrections,
    void* stream) {
    if (elements == 0) return;
    const auto vector_aligned =
        (reinterpret_cast<std::uintptr_t>(parameter) & 15U) == 0U &&
        (reinterpret_cast<std::uintptr_t>(gradient) & 15U) == 0U;
    if (vector_aligned) {
        const auto vectors = (elements + 3) / 4;
        hipLaunchKernelGGL(
            adamw_update_bf16_moments_graph_vectorized_kernel,
            dim3(grid_size(vectors)), dim3(kBlockSize), 0, as_stream(stream),
            parameter, gradient,
            static_cast<hip_bfloat16*>(first_moment_bf16),
            static_cast<hip_bfloat16*>(second_moment_bf16),
            static_cast<hip_bfloat16*>(parameter_bf16_mirror), elements,
            learning_rate, beta1, beta2, epsilon, weight_decay, corrections);
        check_launch("adamw_update_bf16_moments_graph_vectorized_kernel");
        return;
    }
    hipLaunchKernelGGL(
        adamw_update_bf16_moments_graph_kernel, dim3(grid_size(elements)),
        dim3(kBlockSize), 0, as_stream(stream), parameter, gradient,
        static_cast<hip_bfloat16*>(first_moment_bf16),
        static_cast<hip_bfloat16*>(second_moment_bf16),
        static_cast<hip_bfloat16*>(parameter_bf16_mirror), elements,
        learning_rate, beta1, beta2, epsilon, weight_decay, corrections);
    check_launch("adamw_update_bf16_moments_graph_kernel");
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

void launch_add_bias_bf16(const void* input_bf16, const float* bias,
                          void* output_bf16, std::int64_t elements,
                          std::int64_t width, void* stream) {
    if (elements == 0) return;
    hipLaunchKernelGGL(
        add_bias_bf16_kernel, dim3(grid_size(elements)), dim3(kBlockSize), 0,
        as_stream(stream), static_cast<const hip_bfloat16*>(input_bf16),
        bias, static_cast<hip_bfloat16*>(output_bf16), elements, width);
    check_launch("add_bias_bf16_kernel");
}

void launch_bias_gradient(const float* gradient, float* output,
                          std::int64_t rows, std::int64_t width, void* stream) {
    if (width == 0) return;
    hipLaunchKernelGGL(bias_gradient_kernel, dim3(grid_size(width)), dim3(kBlockSize), 0,
                       as_stream(stream), gradient, output, rows, width);
    check_launch("bias_gradient_kernel");
}

void launch_bias_gradient_cooperative(const float* gradient, float* output,
                                      std::int64_t rows, std::int64_t width,
                                      void* stream) {
    if (width == 0) return;
    constexpr unsigned int columns = 32;
    constexpr unsigned int row_lanes = 8;
    const auto blocks = static_cast<unsigned int>((width + columns - 1) / columns);
    hipLaunchKernelGGL(bias_gradient_cooperative_kernel, dim3(blocks),
                       dim3(columns, row_lanes), 0, as_stream(stream),
                       gradient, output, rows, width);
    check_launch("bias_gradient_cooperative_kernel");
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

bool aligned_to_16(const void* pointer) {
    return reinterpret_cast<std::uintptr_t>(pointer) % 16U == 0U;
}

template <typename T, typename Operation>
void launch_binary_typed_auto(const void* left, const void* right, void* output,
                              std::int64_t elements, void* stream) {
    constexpr std::int64_t kVectorMinimumElements = 1LL << 22;
    constexpr bool kLowPrecision = !std::is_same_v<T, float>;
    if (kLowPrecision && elements >= kVectorMinimumElements &&
        aligned_to_16(left) && aligned_to_16(right) && aligned_to_16(output)) {
        constexpr auto lanes = static_cast<std::int64_t>(16 / sizeof(T));
        const auto packets = (elements + lanes - 1) / lanes;
        hipLaunchKernelGGL((binary_typed_vector16_kernel<T, Operation>),
                           dim3(grid_size(packets)), dim3(kBlockSize), 0,
                           as_stream(stream), static_cast<const T*>(left),
                           static_cast<const T*>(right), static_cast<T*>(output),
                           elements);
        return;
    }
    if constexpr (std::is_same_v<Operation, AddTypedOperation>) {
        hipLaunchKernelGGL(add_typed_kernel<T>, dim3(grid_size(elements)),
                           dim3(kBlockSize), 0, as_stream(stream),
                           static_cast<const T*>(left),
                           static_cast<const T*>(right), static_cast<T*>(output),
                           elements);
    } else {
        hipLaunchKernelGGL(multiply_typed_kernel<T>, dim3(grid_size(elements)),
                           dim3(kBlockSize), 0, as_stream(stream),
                           static_cast<const T*>(left),
                           static_cast<const T*>(right), static_cast<T*>(output),
                           elements);
    }
}

#define MICROLLM_LAUNCH_BINARY_TYPED_AUTO(operation_type, operation_name)               \
    switch (dtype) {                                                                    \
        case DType::Float32:                                                            \
            launch_binary_typed_auto<float, operation_type>(                           \
                left, right, output, elements, stream);                                 \
            break;                                                                      \
        case DType::Float16:                                                            \
            launch_binary_typed_auto<__half, operation_type>(                          \
                left, right, output, elements, stream);                                 \
            break;                                                                      \
        case DType::BFloat16:                                                           \
            launch_binary_typed_auto<hip_bfloat16, operation_type>(                    \
                left, right, output, elements, stream);                                 \
            break;                                                                      \
        default: throw std::invalid_argument(operation_name " requires a floating dtype"); \
    }

void launch_add_typed(const void* left, const void* right, void* output, DType dtype,
                      std::int64_t elements, void* stream) {
    if (elements == 0) return;
    MICROLLM_LAUNCH_BINARY_TYPED_AUTO(AddTypedOperation, "typed add")
    check_launch("binary_typed_add_kernel");
}

void launch_multiply_typed(const void* left, const void* right, void* output, DType dtype,
                           std::int64_t elements, void* stream) {
    if (elements == 0) return;
    MICROLLM_LAUNCH_BINARY_TYPED_AUTO(MultiplyTypedOperation, "typed multiply")
    check_launch("binary_typed_multiply_kernel");
}

#undef MICROLLM_LAUNCH_BINARY_TYPED_AUTO

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
                         std::int64_t elements, void* stream, bool vectorized) {
    if (elements == 0) return;
    if (vectorized) {
        if (dtype != DType::BFloat16) {
            throw std::invalid_argument(
                "vectorized swiglu requires bfloat16 tensors");
        }
        constexpr std::int64_t width = 4;
        const auto vectors = (elements + width - 1) / width;
        hipLaunchKernelGGL(
            swiglu_bfloat16_vectorized_kernel, dim3(grid_size(vectors)),
            dim3(kBlockSize), 0, as_stream(stream),
            static_cast<const hip_bfloat16*>(gate),
            static_cast<const hip_bfloat16*>(up),
            static_cast<hip_bfloat16*>(output), elements);
        check_launch("swiglu_bfloat16_vectorized_kernel");
        return;
    }
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
void launch_dynamic_quantize_for_input(
    const Input* input, void* output, DType fp8_dtype, float* scale,
    float* partial_maxima, std::int64_t partial_count,
    std::int64_t elements, float minimum_scale, float maximum_fraction,
    hipStream_t stream) {
    hipLaunchKernelGGL(fp8_tensor_scale_kernel<Input>,
                       dim3(static_cast<unsigned>(partial_count)),
                       dim3(kBlockSize), 0, stream, input, partial_maxima,
                       elements);
    check_launch("fp8_tensor_scale_kernel");
    const auto quantized_maximum =
        fp8_dtype == DType::Float8E4M3FNUZ ? 240.0F : 57344.0F;
    hipLaunchKernelGGL(fp8_finalize_scale_kernel, dim3(1), dim3(kBlockSize),
                       0, stream, partial_maxima, scale, partial_count,
                       minimum_scale, maximum_fraction, quantized_maximum);
    check_launch("fp8_finalize_scale_kernel");
    const auto grid = dim3(grid_size(elements));
    const auto block = dim3(kBlockSize);
    if (fp8_dtype == DType::Float8E4M3FNUZ) {
        hipLaunchKernelGGL(
            (quantize_fp8_device_scale_kernel<Input, __hip_fp8_e4m3_fnuz>),
            grid, block, 0, stream, input,
            static_cast<__hip_fp8_e4m3_fnuz*>(output), elements, scale,
            quantized_maximum);
    } else if (fp8_dtype == DType::Float8E5M2FNUZ) {
        hipLaunchKernelGGL(
            (quantize_fp8_device_scale_kernel<Input, __hip_fp8_e5m2_fnuz>),
            grid, block, 0, stream, input,
            static_cast<__hip_fp8_e5m2_fnuz*>(output), elements, scale,
            quantized_maximum);
    } else {
        throw std::invalid_argument(
            "dynamic quantize output must be an MI300 FNUZ FP8 dtype");
    }
    check_launch("quantize_fp8_device_scale_kernel");
}

void launch_quantize_fp8_dynamic(
    const void* input, DType input_dtype, void* output, DType fp8_dtype,
    float* scale, float* partial_maxima, std::int64_t partial_count,
    std::int64_t elements, float minimum_scale, float maximum_fraction,
    void* stream) {
    if (elements == 0) return;
    if (partial_count <= 0 || partial_count > 1024) {
        throw std::invalid_argument("dynamic FP8 partial count is invalid");
    }
    switch (input_dtype) {
        case DType::Float32:
            launch_dynamic_quantize_for_input(
                static_cast<const float*>(input), output, fp8_dtype, scale,
                partial_maxima, partial_count,
                elements, minimum_scale, maximum_fraction, as_stream(stream));
            break;
        case DType::Float16:
            launch_dynamic_quantize_for_input(
                static_cast<const __half*>(input), output, fp8_dtype, scale,
                partial_maxima, partial_count,
                elements, minimum_scale, maximum_fraction, as_stream(stream));
            break;
        case DType::BFloat16:
            launch_dynamic_quantize_for_input(
                static_cast<const hip_bfloat16*>(input), output, fp8_dtype,
                scale, partial_maxima, partial_count, elements, minimum_scale,
                maximum_fraction, as_stream(stream));
            break;
        default:
            throw std::invalid_argument(
                "dynamic quantize input must be FP32, FP16, or BF16");
    }
}

template <typename Input>
void launch_row_quantize_for_input(
    const Input* input, void* output, DType fp8_dtype, float* scales,
    std::int64_t rows, std::int64_t columns, float minimum_scale,
    hipStream_t stream) {
    hipLaunchKernelGGL(fp8_row_scale_kernel<Input>,
                       dim3(static_cast<unsigned>(rows)),
                       dim3(kBlockSize), 0, stream, input, scales,
                       columns, minimum_scale);
    check_launch("fp8_row_scale_kernel");
    const auto elements = rows * columns;
    const auto grid = dim3(grid_size(elements));
    if (fp8_dtype == DType::Float8E4M3FNUZ) {
        hipLaunchKernelGGL(
            (quantize_fp8_row_scale_kernel<Input, __hip_fp8_e4m3_fnuz>),
            grid, dim3(kBlockSize), 0, stream, input,
            static_cast<__hip_fp8_e4m3_fnuz*>(output), elements, columns,
            scales);
    } else if (fp8_dtype == DType::Float8E5M2FNUZ) {
        hipLaunchKernelGGL(
            (quantize_fp8_row_scale_kernel<Input, __hip_fp8_e5m2_fnuz>),
            grid, dim3(kBlockSize), 0, stream, input,
            static_cast<__hip_fp8_e5m2_fnuz*>(output), elements, columns,
            scales);
    } else {
        throw std::invalid_argument("row quantize requires FNUZ FP8 output");
    }
    check_launch("quantize_fp8_row_scale_kernel");
}

void launch_quantize_fp8_rows_dynamic(
    const void* input, DType input_dtype, void* output, DType fp8_dtype,
    float* scales, std::int64_t rows, std::int64_t columns,
    float minimum_scale, void* stream) {
    switch (input_dtype) {
        case DType::Float32:
            launch_row_quantize_for_input(static_cast<const float*>(input),
                output, fp8_dtype, scales, rows, columns, minimum_scale,
                as_stream(stream)); break;
        case DType::Float16:
            launch_row_quantize_for_input(static_cast<const __half*>(input),
                output, fp8_dtype, scales, rows, columns, minimum_scale,
                as_stream(stream)); break;
        case DType::BFloat16:
            launch_row_quantize_for_input(static_cast<const hip_bfloat16*>(input),
                output, fp8_dtype, scales, rows, columns, minimum_scale,
                as_stream(stream)); break;
        default: throw std::invalid_argument("row quantize input dtype is invalid");
    }
}

template <typename Input>
void launch_column_quantize_for_input(
    const Input* input, void* output, DType fp8_dtype, float* scales,
    std::int64_t rows, std::int64_t columns, float minimum_scale,
    hipStream_t stream) {
    hipLaunchKernelGGL(fp8_column_scale_kernel<Input>,
                       dim3(static_cast<unsigned>(columns)),
                       dim3(kBlockSize), 0, stream, input, scales, rows,
                       columns, minimum_scale);
    check_launch("fp8_column_scale_kernel");
    const auto elements = rows * columns;
    const auto grid = dim3(grid_size(elements));
    if (fp8_dtype == DType::Float8E4M3FNUZ) {
        hipLaunchKernelGGL(
            (quantize_fp8_column_scale_kernel<Input, __hip_fp8_e4m3_fnuz>),
            grid, dim3(kBlockSize), 0, stream, input,
            static_cast<__hip_fp8_e4m3_fnuz*>(output), elements, columns,
            scales);
    } else if (fp8_dtype == DType::Float8E5M2FNUZ) {
        hipLaunchKernelGGL(
            (quantize_fp8_column_scale_kernel<Input, __hip_fp8_e5m2_fnuz>),
            grid, dim3(kBlockSize), 0, stream, input,
            static_cast<__hip_fp8_e5m2_fnuz*>(output), elements, columns,
            scales);
    } else {
        throw std::invalid_argument("column quantize requires FNUZ FP8 output");
    }
    check_launch("quantize_fp8_column_scale_kernel");
}

void launch_quantize_fp8_columns_dynamic(
    const void* input, DType input_dtype, void* output, DType fp8_dtype,
    float* scales, std::int64_t rows, std::int64_t columns,
    float minimum_scale, void* stream) {
    switch (input_dtype) {
        case DType::Float32:
            launch_column_quantize_for_input(static_cast<const float*>(input),
                output, fp8_dtype, scales, rows, columns, minimum_scale,
                as_stream(stream)); break;
        case DType::Float16:
            launch_column_quantize_for_input(static_cast<const __half*>(input),
                output, fp8_dtype, scales, rows, columns, minimum_scale,
                as_stream(stream)); break;
        case DType::BFloat16:
            launch_column_quantize_for_input(
                static_cast<const hip_bfloat16*>(input), output, fp8_dtype,
                scales, rows, columns, minimum_scale, as_stream(stream)); break;
        default:
            throw std::invalid_argument("column quantize input dtype is invalid");
    }
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

template <typename Input>
void launch_cast_transpose_for_input(const Input* input, void* output,
                                     DType output_dtype, std::int64_t rows,
                                     std::int64_t columns, hipStream_t stream) {
    const auto grid = dim3(grid_size(rows * columns));
    const auto block = dim3(kBlockSize);
#define MICROLLM_CAST_TRANSPOSE_OUTPUT_CASE(cpp_type, enum_value)                       \
    case enum_value:                                                                    \
        hipLaunchKernelGGL((cast_transpose_2d_kernel<Input, cpp_type>), grid, block, 0, \
                           stream, input, static_cast<cpp_type*>(output), rows, columns); \
        break
    switch (output_dtype) {
        MICROLLM_CAST_TRANSPOSE_OUTPUT_CASE(float, DType::Float32);
        MICROLLM_CAST_TRANSPOSE_OUTPUT_CASE(__half, DType::Float16);
        MICROLLM_CAST_TRANSPOSE_OUTPUT_CASE(hip_bfloat16, DType::BFloat16);
        default: throw std::invalid_argument(
            "cast transpose output must be FP32, FP16, or BF16");
    }
#undef MICROLLM_CAST_TRANSPOSE_OUTPUT_CASE
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

template <typename Input>
void launch_device_scale_dequantize_for_input(
    const Input* input, void* output, DType output_dtype,
    std::int64_t elements, const float* scale, hipStream_t stream) {
    const auto grid = dim3(grid_size(elements));
    const auto block = dim3(kBlockSize);
#define MICROLLM_DEVICE_SCALE_DEQUANT_CASE(cpp_type, enum_value)                   \
    case enum_value:                                                              \
        hipLaunchKernelGGL(                                                       \
            (dequantize_fp8_device_scale_kernel<Input, cpp_type>), grid, block,  \
            0, stream, input, static_cast<cpp_type*>(output), elements, scale);  \
        break
    switch (output_dtype) {
        MICROLLM_DEVICE_SCALE_DEQUANT_CASE(float, DType::Float32);
        MICROLLM_DEVICE_SCALE_DEQUANT_CASE(__half, DType::Float16);
        MICROLLM_DEVICE_SCALE_DEQUANT_CASE(hip_bfloat16, DType::BFloat16);
        default:
            throw std::invalid_argument(
                "device-scale dequantize output must be a floating dtype");
    }
#undef MICROLLM_DEVICE_SCALE_DEQUANT_CASE
    check_launch("dequantize_fp8_device_scale_kernel");
}

void launch_dequantize_fp8_device_scale(
    const void* input, DType fp8_dtype, void* output, DType output_dtype,
    std::int64_t elements, const float* scale, void* stream) {
    if (elements == 0) return;
    if (fp8_dtype == DType::Float8E4M3FNUZ) {
        launch_device_scale_dequantize_for_input(
            static_cast<const __hip_fp8_e4m3_fnuz*>(input), output,
            output_dtype, elements, scale, as_stream(stream));
    } else if (fp8_dtype == DType::Float8E5M2FNUZ) {
        launch_device_scale_dequantize_for_input(
            static_cast<const __hip_fp8_e5m2_fnuz*>(input), output,
            output_dtype, elements, scale, as_stream(stream));
    } else {
        throw std::invalid_argument(
            "device-scale dequantize input must be an MI300 FNUZ dtype");
    }
}

template <typename Input, typename Output>
void launch_row_dequantize_typed(
    const Input* input, Output* output, std::int64_t rows,
    std::int64_t columns, const float* scales, hipStream_t stream) {
    const auto elements = rows * columns;
    hipLaunchKernelGGL((dequantize_fp8_row_scale_kernel<Input, Output>),
                       dim3(grid_size(elements)), dim3(kBlockSize), 0, stream,
                       input, output, elements, columns, scales);
    check_launch("dequantize_fp8_row_scale_kernel");
}

template <typename Input>
void launch_row_dequantize_for_input(
    const Input* input, void* output, DType output_dtype,
    std::int64_t rows, std::int64_t columns, const float* scales,
    hipStream_t stream) {
    switch (output_dtype) {
        case DType::Float32: launch_row_dequantize_typed(
            input, static_cast<float*>(output), rows, columns, scales, stream); break;
        case DType::Float16: launch_row_dequantize_typed(
            input, static_cast<__half*>(output), rows, columns, scales, stream); break;
        case DType::BFloat16: launch_row_dequantize_typed(
            input, static_cast<hip_bfloat16*>(output), rows, columns, scales, stream); break;
        default: throw std::invalid_argument("row dequantize output dtype is invalid");
    }
}

void launch_dequantize_fp8_row_scales(
    const void* input, DType fp8_dtype, void* output, DType output_dtype,
    std::int64_t rows, std::int64_t columns, const float* scales,
    void* stream) {
    if (fp8_dtype == DType::Float8E4M3FNUZ) {
        launch_row_dequantize_for_input(
            static_cast<const __hip_fp8_e4m3_fnuz*>(input), output,
            output_dtype, rows, columns, scales, as_stream(stream));
    } else if (fp8_dtype == DType::Float8E5M2FNUZ) {
        launch_row_dequantize_for_input(
            static_cast<const __hip_fp8_e5m2_fnuz*>(input), output,
            output_dtype, rows, columns, scales, as_stream(stream));
    } else {
        throw std::invalid_argument("row dequantize input dtype is invalid");
    }
}

template <typename Input, typename Output>
void launch_column_dequantize_typed(
    const Input* input, Output* output, std::int64_t rows,
    std::int64_t columns, const float* scales, hipStream_t stream) {
    const auto elements = rows * columns;
    hipLaunchKernelGGL((dequantize_fp8_column_scale_kernel<Input, Output>),
                       dim3(grid_size(elements)), dim3(kBlockSize), 0, stream,
                       input, output, elements, columns, scales);
    check_launch("dequantize_fp8_column_scale_kernel");
}

template <typename Input>
void launch_column_dequantize_for_input(
    const Input* input, void* output, DType output_dtype,
    std::int64_t rows, std::int64_t columns, const float* scales,
    hipStream_t stream) {
    switch (output_dtype) {
        case DType::Float32: launch_column_dequantize_typed(
            input, static_cast<float*>(output), rows, columns, scales, stream); break;
        case DType::Float16: launch_column_dequantize_typed(
            input, static_cast<__half*>(output), rows, columns, scales, stream); break;
        case DType::BFloat16: launch_column_dequantize_typed(
            input, static_cast<hip_bfloat16*>(output), rows, columns, scales, stream); break;
        default: throw std::invalid_argument("column dequantize output dtype is invalid");
    }
}

void launch_dequantize_fp8_column_scales(
    const void* input, DType fp8_dtype, void* output, DType output_dtype,
    std::int64_t rows, std::int64_t columns, const float* scales,
    void* stream) {
    if (fp8_dtype == DType::Float8E4M3FNUZ) {
        launch_column_dequantize_for_input(
            static_cast<const __hip_fp8_e4m3_fnuz*>(input), output,
            output_dtype, rows, columns, scales, as_stream(stream));
    } else if (fp8_dtype == DType::Float8E5M2FNUZ) {
        launch_column_dequantize_for_input(
            static_cast<const __hip_fp8_e5m2_fnuz*>(input), output,
            output_dtype, rows, columns, scales, as_stream(stream));
    } else {
        throw std::invalid_argument("column dequantize input dtype is invalid");
    }
}

template <typename T>
void launch_scale_columns_typed(T* values, std::int64_t rows,
                                std::int64_t columns, const float* scales,
                                hipStream_t stream) {
    const auto elements = rows * columns;
    hipLaunchKernelGGL(scale_columns_by_first_kernel<T>,
                       dim3(grid_size(elements)), dim3(kBlockSize), 0, stream,
                       values, elements, columns, scales);
    check_launch("scale_columns_by_first_kernel");
}

void launch_scale_columns_by_first(
    void* values, DType dtype, std::int64_t rows, std::int64_t columns,
    const float* scales, void* stream) {
    switch (dtype) {
        case DType::Float32: launch_scale_columns_typed(
            static_cast<float*>(values), rows, columns, scales,
            as_stream(stream)); break;
        case DType::Float16: launch_scale_columns_typed(
            static_cast<__half*>(values), rows, columns, scales,
            as_stream(stream)); break;
        case DType::BFloat16: launch_scale_columns_typed(
            static_cast<hip_bfloat16*>(values), rows, columns, scales,
            as_stream(stream)); break;
        default: throw std::invalid_argument("column output scale dtype is invalid");
    }
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

void launch_cast_transpose_2d(const void* input, DType input_dtype,
                              void* output, DType output_dtype,
                              std::int64_t rows, std::int64_t columns,
                              void* stream) {
    if (rows == 0 || columns == 0) return;
    switch (input_dtype) {
        case DType::Float32:
            launch_cast_transpose_for_input(static_cast<const float*>(input), output,
                                            output_dtype, rows, columns, as_stream(stream));
            break;
        case DType::Float16:
            launch_cast_transpose_for_input(static_cast<const __half*>(input), output,
                                            output_dtype, rows, columns, as_stream(stream));
            break;
        case DType::BFloat16:
            launch_cast_transpose_for_input(static_cast<const hip_bfloat16*>(input), output,
                                            output_dtype, rows, columns, as_stream(stream));
            break;
        default: throw std::invalid_argument(
            "cast transpose input must be FP32, FP16, or BF16");
    }
    check_launch("cast_transpose_2d_kernel");
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

void launch_rms_norm_bf16_output(
    const float* input, const float* weight, void* output_bf16,
    std::int64_t rows, std::int64_t width, float epsilon, void* stream) {
    if (rows == 0) return;
    hipLaunchKernelGGL(
        rms_norm_bf16_output_kernel, dim3(static_cast<unsigned>(rows)),
        dim3(kBlockSize), 0, as_stream(stream), input, weight,
        static_cast<hip_bfloat16*>(output_bf16), rows, width, epsilon);
    check_launch("rms_norm_bf16_output_kernel");
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

void launch_rope_split_half_bias_bthd(
    const void* input, DType input_dtype, const float* bias, float* output,
    std::int64_t batches, std::int64_t sequence_size, std::int64_t heads,
    std::int64_t head_width, std::int64_t position_offset, float base,
    void* stream) {
    const auto pairs = batches * sequence_size * heads * head_width / 2;
    if (pairs == 0) return;
    if (input_dtype == DType::Float32) {
        hipLaunchKernelGGL((rope_split_half_bias_bthd_kernel<float, float>),
                           dim3(grid_size(pairs)), dim3(kBlockSize), 0,
                           as_stream(stream), static_cast<const float*>(input),
                           bias, output, pairs, sequence_size, heads,
                           head_width, position_offset, base);
    } else if (input_dtype == DType::BFloat16) {
        hipLaunchKernelGGL((rope_split_half_bias_bthd_kernel<hip_bfloat16, float>),
                           dim3(grid_size(pairs)), dim3(kBlockSize), 0,
                           as_stream(stream),
                           static_cast<const hip_bfloat16*>(input),
                           bias, output, pairs, sequence_size, heads,
                           head_width, position_offset, base);
    } else {
        throw std::invalid_argument(
            "rope_split_half_bias_bthd input must be FP32 or BF16");
    }
    check_launch("rope_split_half_bias_bthd_kernel");
}

void launch_rope_split_half_bias_bthd_bf16(
    const void* input_bf16, const float* bias, void* output_bf16,
    std::int64_t batches, std::int64_t sequence_size, std::int64_t heads,
    std::int64_t head_width, std::int64_t position_offset, float base,
    void* stream) {
    const auto pairs = batches * sequence_size * heads * head_width / 2;
    if (pairs == 0) return;
    hipLaunchKernelGGL(
        (rope_split_half_bias_bthd_kernel<hip_bfloat16, hip_bfloat16>),
        dim3(grid_size(pairs)), dim3(kBlockSize), 0, as_stream(stream),
        static_cast<const hip_bfloat16*>(input_bf16), bias,
        static_cast<hip_bfloat16*>(output_bf16), pairs, sequence_size,
        heads, head_width, position_offset, base);
    check_launch("rope_split_half_bias_bthd_bf16_kernel");
}

void launch_rope_positions(const float* input, const std::int32_t* positions,
                           float* output, std::int64_t batches,
                           std::int64_t heads, std::int64_t head_width,
                           float base, void* stream) {
    const auto pairs = batches * heads * head_width / 2;
    hipLaunchKernelGGL(rope_positions_kernel, dim3(grid_size(pairs)),
                       dim3(kBlockSize), 0, as_stream(stream), input, positions,
                       output, pairs, heads, head_width, base);
    check_launch("rope_positions_kernel");
}

void launch_rope_split_half_positions(
    const float* input, const std::int32_t* positions, float* output,
    std::int64_t batches, std::int64_t heads, std::int64_t head_width,
    float base, void* stream) {
    const auto pairs = batches * heads * head_width / 2;
    hipLaunchKernelGGL(rope_split_half_positions_kernel,
                       dim3(grid_size(pairs)), dim3(kBlockSize), 0,
                       as_stream(stream), input, positions, output, pairs,
                       heads, head_width, base);
    check_launch("rope_split_half_positions_kernel");
}

void launch_rope_split_half_bias_positions(
    const float* input, const float* bias, const std::int32_t* positions,
    float* output, std::int64_t batches, std::int64_t heads,
    std::int64_t head_width, float base, void* stream) {
    const auto pairs = batches * heads * head_width / 2;
    hipLaunchKernelGGL(rope_split_half_bias_positions_kernel,
                       dim3(grid_size(pairs)), dim3(kBlockSize), 0,
                       as_stream(stream), input, bias, positions, output, pairs,
                       heads, head_width, base);
    check_launch("rope_split_half_bias_positions_kernel");
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
    if (rows >= 256) {
        hipLaunchKernelGGL(rms_norm_weight_gradient_rows_kernel,
                           dim3(static_cast<unsigned>(width)), dim3(kBlockSize), 0,
                           as_stream(stream), input, gradient, row_inverse_rms,
                           weight_gradient, rows, width);
        check_launch("rms_norm_weight_gradient_rows_kernel");
    } else {
        hipLaunchKernelGGL(rms_norm_weight_gradient_kernel, dim3(grid_size(width)),
                           dim3(kBlockSize), 0, as_stream(stream), input, gradient,
                           row_inverse_rms, weight_gradient, rows, width);
        check_launch("rms_norm_weight_gradient_kernel");
    }
}

void launch_kv_cache_store(const float* current, void* cache, DType cache_dtype,
                           std::int64_t batches, std::int64_t heads,
                           std::int64_t capacity,
                           std::int64_t width, std::int64_t position, void* stream) {
    const auto elements = batches * heads * width;
    if (cache_dtype == DType::Float32) {
        hipLaunchKernelGGL(kv_cache_store_kernel<float>, dim3(grid_size(elements)),
                           dim3(kBlockSize), 0, as_stream(stream), current,
                           static_cast<float*>(cache), batches, heads, capacity,
                           width, position);
    } else if (cache_dtype == DType::BFloat16) {
        hipLaunchKernelGGL(kv_cache_store_kernel<hip_bfloat16>,
                           dim3(grid_size(elements)), dim3(kBlockSize), 0,
                           as_stream(stream), current,
                           static_cast<hip_bfloat16*>(cache), batches, heads,
                           capacity, width, position);
    } else {
        throw std::invalid_argument("KV cache store supports FP32 and BF16");
    }
    check_launch("kv_cache_store_kernel");
}

void launch_kv_cache_store_pair(const float* current_key, const float* current_value,
                                void* key_cache, void* value_cache,
                                DType cache_dtype,
                                std::int64_t batches, std::int64_t heads,
                                std::int64_t capacity,
                                std::int64_t width, std::int64_t position, void* stream) {
    const auto elements = batches * heads * width;
    if (cache_dtype == DType::Float32) {
        hipLaunchKernelGGL(kv_cache_store_pair_kernel<float>,
                           dim3(grid_size(elements)), dim3(kBlockSize), 0,
                           as_stream(stream), current_key, current_value,
                           static_cast<float*>(key_cache),
                           static_cast<float*>(value_cache), batches, heads,
                           capacity, width, position);
    } else if (cache_dtype == DType::BFloat16) {
        hipLaunchKernelGGL(kv_cache_store_pair_kernel<hip_bfloat16>,
                           dim3(grid_size(elements)), dim3(kBlockSize), 0,
                           as_stream(stream), current_key, current_value,
                           static_cast<hip_bfloat16*>(key_cache),
                           static_cast<hip_bfloat16*>(value_cache), batches,
                           heads, capacity, width, position);
    } else {
        throw std::invalid_argument("paired KV cache store supports FP32 and BF16");
    }
    check_launch("kv_cache_store_pair_kernel");
}

void launch_kv_cache_store_pair_positions(
    const float* current_key, const float* current_value,
    const std::int32_t* positions, const std::int32_t* cache_rows,
    void* key_cache, void* value_cache, DType cache_dtype,
    std::int64_t active_batches, std::int64_t cache_batches,
    std::int64_t heads, std::int64_t capacity,
    std::int64_t logical_prefix, std::int64_t width, void* stream) {
    const auto elements = active_batches * heads * width;
    if (cache_dtype == DType::Float32) {
        hipLaunchKernelGGL(kv_cache_store_pair_positions_kernel<float>,
                           dim3(grid_size(elements)), dim3(kBlockSize), 0,
                           as_stream(stream), current_key, current_value,
                           positions, cache_rows, static_cast<float*>(key_cache),
                           static_cast<float*>(value_cache), active_batches,
                           cache_batches, heads, capacity, logical_prefix, width);
    } else if (cache_dtype == DType::BFloat16) {
        hipLaunchKernelGGL(
            kv_cache_store_pair_positions_kernel<hip_bfloat16>,
            dim3(grid_size(elements)), dim3(kBlockSize), 0,
            as_stream(stream), current_key, current_value, positions,
            cache_rows, static_cast<hip_bfloat16*>(key_cache),
            static_cast<hip_bfloat16*>(value_cache), active_batches,
            cache_batches, heads, capacity, logical_prefix, width);
    } else {
        throw std::invalid_argument(
            "positioned paired KV store supports FP32 and BF16");
    }
    check_launch("kv_cache_store_pair_positions_kernel");
}

void launch_cached_attention_scores(
    const float* query, const void* key_cache, DType cache_dtype, float* scores,
    std::int64_t batches, std::int64_t heads,
    [[maybe_unused]] std::int64_t kv_heads, std::int64_t sequence,
    std::int64_t cache_batch_stride, std::int64_t cache_head_stride,
    std::int64_t width, std::int64_t repeats, float scale, void* stream) {
    const auto elements = batches * heads * sequence;
    if (cache_dtype == DType::Float32) {
        hipLaunchKernelGGL(cached_attention_scores_kernel<float>,
                           dim3(grid_size(elements)), dim3(kBlockSize), 0,
                           as_stream(stream), query,
                           static_cast<const float*>(key_cache), scores, batches,
                           heads, sequence, cache_batch_stride, cache_head_stride,
                           width, repeats, scale);
    } else if (cache_dtype == DType::BFloat16) {
        hipLaunchKernelGGL(cached_attention_scores_kernel<hip_bfloat16>,
                           dim3(grid_size(elements)), dim3(kBlockSize), 0,
                           as_stream(stream), query,
                           static_cast<const hip_bfloat16*>(key_cache), scores,
                           batches, heads, sequence, cache_batch_stride,
                           cache_head_stride, width, repeats, scale);
    } else {
        throw std::invalid_argument("cached attention supports FP32 and BF16 cache");
    }
    check_launch("cached_attention_scores_kernel");
}

void launch_cached_attention_context(
    const float* probabilities, const void* value_cache, DType cache_dtype,
    float* output,
    std::int64_t batches, std::int64_t heads,
    [[maybe_unused]] std::int64_t kv_heads, std::int64_t sequence,
    std::int64_t cache_batch_stride, std::int64_t cache_head_stride,
    std::int64_t width, std::int64_t repeats, void* stream) {
    const auto elements = batches * heads * width;
    if (cache_dtype == DType::Float32) {
        hipLaunchKernelGGL(cached_attention_context_kernel<float>,
                           dim3(grid_size(elements)), dim3(kBlockSize), 0,
                           as_stream(stream), probabilities,
                           static_cast<const float*>(value_cache), output,
                           batches, heads, sequence, cache_batch_stride,
                           cache_head_stride, width, repeats);
    } else if (cache_dtype == DType::BFloat16) {
        hipLaunchKernelGGL(cached_attention_context_kernel<hip_bfloat16>,
                           dim3(grid_size(elements)), dim3(kBlockSize), 0,
                           as_stream(stream), probabilities,
                           static_cast<const hip_bfloat16*>(value_cache), output,
                           batches, heads, sequence, cache_batch_stride,
                           cache_head_stride, width, repeats);
    } else {
        throw std::invalid_argument("cached attention supports FP32 and BF16 cache");
    }
    check_launch("cached_attention_context_kernel");
}

void launch_cached_attention_finalize_scores(
    const float* scores, const void* value_cache, DType cache_dtype,
    float* output, std::int64_t batches, std::int64_t heads,
    std::int64_t sequence, std::int64_t cache_batch_stride,
    std::int64_t cache_head_stride, std::int64_t width,
    std::int64_t repeats, std::int64_t finalize_threads, void* stream) {
    const auto grid = dim3(static_cast<unsigned>(batches * heads));
    const auto shared_bytes = static_cast<std::size_t>(sequence) * sizeof(float);
    if (cache_dtype != DType::Float32 && cache_dtype != DType::BFloat16) {
        throw std::invalid_argument(
            "materialized-score cached attention supports FP32 and BF16 cache");
    }
    const auto launch = [&]<typename Cache, unsigned Threads>(const Cache* cache) {
        hipLaunchKernelGGL(
            (cached_attention_finalize_scores_kernel<Cache, Threads>), grid,
            dim3(Threads), shared_bytes, as_stream(stream), scores, cache,
            output, batches, heads, sequence, cache_batch_stride,
            cache_head_stride, width, repeats);
    };
    const auto dispatch = [&]<typename Cache>(const Cache* cache) {
        switch (finalize_threads) {
            case 64: launch.template operator()<Cache, 64>(cache); break;
            case 128: launch.template operator()<Cache, 128>(cache); break;
            case 256: launch.template operator()<Cache, 256>(cache); break;
            default:
                throw std::invalid_argument(
                    "materialized-score finalize threads must be 64, 128, or 256");
        }
    };
    if (cache_dtype == DType::Float32) {
        dispatch(static_cast<const float*>(value_cache));
    } else {
        dispatch(static_cast<const hip_bfloat16*>(value_cache));
    }
    check_launch("cached_attention_finalize_scores_kernel");
}

void launch_cached_attention_split_pv(
    const float* scores, const void* value_cache, DType cache_dtype,
    float* probabilities, float* partials, float* output,
    std::int64_t batches, std::int64_t heads, std::int64_t sequence,
    std::int64_t cache_batch_stride, std::int64_t cache_head_stride,
    std::int64_t width, std::int64_t repeats, std::int64_t splits,
    void* stream) {
    const auto softmax_grid = dim3(static_cast<unsigned>(batches * heads));
    const auto shared_bytes = static_cast<std::size_t>(sequence) * sizeof(float);
    hipLaunchKernelGGL(
        cached_attention_exact_softmax_kernel, softmax_grid,
        dim3(kBlockSize), shared_bytes, as_stream(stream), scores,
        probabilities, batches, heads, sequence);
    check_launch("cached_attention_exact_softmax_kernel");
    const auto partial_grid = dim3(
        static_cast<unsigned>(batches * heads * splits));
    const auto threads = dim3(static_cast<unsigned>(std::min<std::int64_t>(
        width, static_cast<std::int64_t>(kBlockSize))));
    if (cache_dtype == DType::Float32) {
        hipLaunchKernelGGL(
            cached_attention_split_pv_partial_kernel<float>, partial_grid,
            threads, 0, as_stream(stream), probabilities,
            static_cast<const float*>(value_cache), partials, batches, heads,
            sequence, cache_batch_stride, cache_head_stride, width, repeats,
            splits);
    } else if (cache_dtype == DType::BFloat16) {
        hipLaunchKernelGGL(
            cached_attention_split_pv_partial_kernel<hip_bfloat16>,
            partial_grid, threads, 0, as_stream(stream), probabilities,
            static_cast<const hip_bfloat16*>(value_cache), partials, batches,
            heads, sequence, cache_batch_stride, cache_head_stride, width,
            repeats, splits);
    } else {
        throw std::invalid_argument(
            "split-PV cached attention supports FP32 and BF16 cache");
    }
    check_launch("cached_attention_split_pv_partial_kernel");
    const auto elements = batches * heads * width;
    hipLaunchKernelGGL(
        cached_attention_split_pv_combine_kernel, dim3(grid_size(elements)),
        dim3(kBlockSize), 0, as_stream(stream), partials, output, elements,
        width, splits);
    check_launch("cached_attention_split_pv_combine_kernel");
}

void launch_cached_attention_gqa_value_reuse(
    const float* scores, const void* value_cache, DType cache_dtype,
    float* probabilities, float* output, std::int64_t batches,
    std::int64_t heads, std::int64_t kv_heads, std::int64_t sequence,
    std::int64_t cache_batch_stride, std::int64_t cache_head_stride,
    std::int64_t width, std::int64_t repeats, std::int64_t tile_columns,
    void* stream) {
    const auto softmax_grid = dim3(static_cast<unsigned>(batches * heads));
    const auto shared_bytes = static_cast<std::size_t>(sequence) * sizeof(float);
    hipLaunchKernelGGL(
        cached_attention_exact_softmax_kernel, softmax_grid,
        dim3(kBlockSize), shared_bytes, as_stream(stream), scores,
        probabilities, batches, heads, sequence);
    check_launch("cached_attention_exact_softmax_kernel");
    const auto tiles = (width + tile_columns - 1) / tile_columns;
    const auto grid = dim3(static_cast<unsigned>(batches * kv_heads * tiles));
    constexpr unsigned threads = 64;
    if (cache_dtype != DType::Float32 && cache_dtype != DType::BFloat16) {
        throw std::invalid_argument(
            "GQA value-reuse cached attention supports FP32 and BF16 cache");
    }
    const auto launch = [&]<typename Cache, int RepeatCount>(const Cache* cache) {
        hipLaunchKernelGGL(
            (cached_attention_gqa_value_reuse_kernel<Cache, RepeatCount>),
            grid, dim3(threads), 0, as_stream(stream), probabilities, cache,
            output, batches, heads, kv_heads, sequence, cache_batch_stride,
            cache_head_stride, width, tile_columns);
    };
    const auto dispatch = [&]<typename Cache>(const Cache* cache) {
        switch (repeats) {
            case 1: launch.template operator()<Cache, 1>(cache); break;
            case 2: launch.template operator()<Cache, 2>(cache); break;
            case 3: launch.template operator()<Cache, 3>(cache); break;
            case 4: launch.template operator()<Cache, 4>(cache); break;
            case 5: launch.template operator()<Cache, 5>(cache); break;
            case 6: launch.template operator()<Cache, 6>(cache); break;
            case 7: launch.template operator()<Cache, 7>(cache); break;
            case 8: launch.template operator()<Cache, 8>(cache); break;
            default:
                throw std::invalid_argument(
                    "GQA value-reuse repeats must be 1..8");
        }
    };
    if (cache_dtype == DType::Float32) {
        dispatch(static_cast<const float*>(value_cache));
    } else {
        dispatch(static_cast<const hip_bfloat16*>(value_cache));
    }
    check_launch("cached_attention_gqa_value_reuse_kernel");
}

void launch_cached_attention_fused(
    const float* query, const void* key_cache, const void* value_cache,
    DType cache_dtype, float* output, std::int64_t batches, std::int64_t heads,
    std::int64_t sequence, std::int64_t cache_batch_stride,
    std::int64_t cache_head_stride, std::int64_t width,
    std::int64_t repeats, float scale, void* stream) {
    const auto grid = dim3(static_cast<unsigned>(batches * heads));
    const auto shared_bytes = static_cast<std::size_t>(sequence) * sizeof(float);
    if (cache_dtype == DType::Float32) {
        hipLaunchKernelGGL(cached_attention_fused_kernel<float>, grid,
                           dim3(kBlockSize), shared_bytes, as_stream(stream), query,
                           static_cast<const float*>(key_cache),
                           static_cast<const float*>(value_cache), output, batches,
                           heads, sequence, cache_batch_stride, cache_head_stride,
                           width, repeats, scale);
    } else if (cache_dtype == DType::BFloat16) {
        hipLaunchKernelGGL(cached_attention_fused_kernel<hip_bfloat16>, grid,
                           dim3(kBlockSize), shared_bytes, as_stream(stream), query,
                           static_cast<const hip_bfloat16*>(key_cache),
                           static_cast<const hip_bfloat16*>(value_cache), output,
                           batches, heads, sequence, cache_batch_stride,
                           cache_head_stride, width, repeats, scale);
    } else {
        throw std::invalid_argument("cached attention supports FP32 and BF16 cache");
    }
    check_launch("cached_attention_fused_kernel");
}

void launch_cached_attention_split_sequence(
    const float* query, const void* key_cache, const void* value_cache,
    DType cache_dtype, float* partial_stats, float* partial_numerators,
    float* output, std::int64_t batches, std::int64_t heads,
    std::int64_t sequence, std::int64_t cache_batch_stride,
    std::int64_t cache_head_stride, std::int64_t width,
    std::int64_t repeats, float scale, std::int64_t splits, void* stream) {
    const auto instances = batches * heads;
    const auto partials = instances * splits;
    const auto maximum_segment = (sequence + splits - 1) / splits;
    const auto shared_bytes =
        static_cast<std::size_t>(maximum_segment) * sizeof(float);
    if (cache_dtype == DType::Float32) {
        hipLaunchKernelGGL(
            cached_attention_split_partial_kernel<float>,
            dim3(static_cast<unsigned>(partials)), dim3(kBlockSize),
            shared_bytes, as_stream(stream), query,
            static_cast<const float*>(key_cache),
            static_cast<const float*>(value_cache), partial_stats,
            partial_numerators, batches, heads, sequence,
            cache_batch_stride, cache_head_stride, width, repeats, scale,
            splits);
    } else if (cache_dtype == DType::BFloat16) {
        hipLaunchKernelGGL(
            cached_attention_split_partial_kernel<hip_bfloat16>,
            dim3(static_cast<unsigned>(partials)), dim3(kBlockSize),
            shared_bytes, as_stream(stream), query,
            static_cast<const hip_bfloat16*>(key_cache),
            static_cast<const hip_bfloat16*>(value_cache), partial_stats,
            partial_numerators, batches, heads, sequence,
            cache_batch_stride, cache_head_stride, width, repeats, scale,
            splits);
    } else {
        throw std::invalid_argument(
            "split cached attention supports FP32 and BF16 cache");
    }
    check_launch("cached_attention_split_partial_kernel");
    hipLaunchKernelGGL(
        cached_attention_split_combine_kernel,
        dim3(static_cast<unsigned>(instances)), dim3(kBlockSize), 0,
        as_stream(stream), partial_stats, partial_numerators, output,
        instances, width, splits);
    check_launch("cached_attention_split_combine_kernel");
}

void launch_cached_attention_fused_positions(
    const float* query, const void* key_cache, const void* value_cache,
    const std::int32_t* positions, const std::int32_t* cache_rows,
    DType cache_dtype, float* output, std::int64_t active_batches,
    std::int64_t cache_batches, std::int64_t heads,
    std::int64_t logical_prefix, std::int64_t cache_batch_stride,
    std::int64_t cache_head_stride, std::int64_t width,
    std::int64_t repeats, float scale, void* stream) {
    const auto grid = dim3(static_cast<unsigned>(active_batches * heads));
    const auto shared_bytes =
        static_cast<std::size_t>(logical_prefix) * sizeof(float);
    if (cache_dtype == DType::Float32) {
        hipLaunchKernelGGL(
            cached_attention_fused_positions_kernel<float>, grid,
            dim3(kBlockSize), shared_bytes, as_stream(stream), query,
            static_cast<const float*>(key_cache),
            static_cast<const float*>(value_cache), positions, cache_rows,
            output, active_batches, cache_batches, heads, logical_prefix,
            cache_batch_stride, cache_head_stride, width, repeats, scale);
    } else if (cache_dtype == DType::BFloat16) {
        hipLaunchKernelGGL(
            cached_attention_fused_positions_kernel<hip_bfloat16>, grid,
            dim3(kBlockSize), shared_bytes, as_stream(stream), query,
            static_cast<const hip_bfloat16*>(key_cache),
            static_cast<const hip_bfloat16*>(value_cache), positions,
            cache_rows, output, active_batches, cache_batches, heads,
            logical_prefix, cache_batch_stride, cache_head_stride, width,
            repeats, scale);
    } else {
        throw std::invalid_argument(
            "positioned cached attention supports FP32 and BF16 cache");
    }
    check_launch("cached_attention_fused_positions_kernel");
}

void launch_cached_attention_scores_positions(
    const float* query, const void* key_cache, const std::int32_t* positions,
    const std::int32_t* cache_rows, DType cache_dtype, float* scores,
    std::int64_t active_batches, std::int64_t cache_batches,
    std::int64_t heads, std::int64_t logical_prefix,
    std::int64_t cache_batch_stride, std::int64_t cache_head_stride,
    std::int64_t width, std::int64_t repeats, float scale, void* stream) {
    const auto elements = active_batches * heads * logical_prefix;
    if (cache_dtype == DType::Float32) {
        hipLaunchKernelGGL(
            cached_attention_scores_positions_kernel<float>,
            dim3(grid_size(elements)), dim3(kBlockSize), 0,
            as_stream(stream), query, static_cast<const float*>(key_cache),
            positions, cache_rows, scores, active_batches, cache_batches,
            heads, logical_prefix, cache_batch_stride, cache_head_stride,
            width, repeats, scale);
    } else if (cache_dtype == DType::BFloat16) {
        hipLaunchKernelGGL(
            cached_attention_scores_positions_kernel<hip_bfloat16>,
            dim3(grid_size(elements)), dim3(kBlockSize), 0,
            as_stream(stream), query,
            static_cast<const hip_bfloat16*>(key_cache), positions,
            cache_rows, scores, active_batches, cache_batches, heads,
            logical_prefix, cache_batch_stride, cache_head_stride, width,
            repeats, scale);
    } else {
        throw std::invalid_argument(
            "positioned cached attention supports FP32 and BF16 cache");
    }
    check_launch("cached_attention_scores_positions_kernel");
}

void launch_cached_attention_context_positions(
    const float* probabilities, const void* value_cache,
    const std::int32_t* positions, const std::int32_t* cache_rows,
    DType cache_dtype, float* output, std::int64_t active_batches,
    std::int64_t cache_batches, std::int64_t heads,
    std::int64_t logical_prefix, std::int64_t cache_batch_stride,
    std::int64_t cache_head_stride, std::int64_t width,
    std::int64_t repeats, void* stream) {
    const auto elements = active_batches * heads * width;
    if (cache_dtype == DType::Float32) {
        hipLaunchKernelGGL(
            cached_attention_context_positions_kernel<float>,
            dim3(grid_size(elements)), dim3(kBlockSize), 0,
            as_stream(stream), probabilities,
            static_cast<const float*>(value_cache), positions, cache_rows,
            output, active_batches, cache_batches, heads, logical_prefix,
            cache_batch_stride, cache_head_stride, width, repeats);
    } else if (cache_dtype == DType::BFloat16) {
        hipLaunchKernelGGL(
            cached_attention_context_positions_kernel<hip_bfloat16>,
            dim3(grid_size(elements)), dim3(kBlockSize), 0,
            as_stream(stream), probabilities,
            static_cast<const hip_bfloat16*>(value_cache), positions,
            cache_rows, output, active_batches, cache_batches, heads,
            logical_prefix, cache_batch_stride, cache_head_stride, width,
            repeats);
    } else {
        throw std::invalid_argument(
            "positioned cached attention supports FP32 and BF16 cache");
    }
    check_launch("cached_attention_context_positions_kernel");
}

void launch_argmax(const float* input, std::int32_t* output,
                   std::int64_t elements, void* stream) {
    hipLaunchKernelGGL(argmax_kernel, dim3(1), dim3(kBlockSize), 0,
                       as_stream(stream), input, output, elements);
    check_launch("argmax_kernel");
}

void launch_argmax_last_dim(const float* input, std::int32_t* output,
                            std::int64_t rows, std::int64_t classes,
                            void* stream) {
    hipLaunchKernelGGL(argmax_last_dim_kernel,
                       dim3(static_cast<unsigned>(rows)), dim3(kBlockSize), 0,
                       as_stream(stream), input, output, rows, classes);
    check_launch("argmax_last_dim_kernel");
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

void launch_swiglu_backward_scalar_seed(
    const float* gate, const float* up, const float* scalar_gradient,
    float* gate_gradient, float* up_gradient,
    std::int64_t elements, void* stream) {
    if (elements == 0) return;
    hipLaunchKernelGGL(
        swiglu_backward_scalar_seed_kernel, dim3(grid_size(elements)),
        dim3(kBlockSize), 0, as_stream(stream), gate, up, scalar_gradient,
        gate_gradient, up_gradient, elements);
    check_launch("swiglu_backward_scalar_seed_kernel");
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

void launch_rope_split_half_bias_bthd_backward(
    const float* gradient, float* input_gradient,
    std::int64_t batches, std::int64_t sequence_size, std::int64_t heads,
    std::int64_t head_width, std::int64_t position_offset, float base,
    void* stream) {
    const auto pairs = batches * sequence_size * heads * head_width / 2;
    if (pairs == 0) return;
    hipLaunchKernelGGL(rope_split_half_bias_bthd_backward_kernel,
                       dim3(grid_size(pairs)), dim3(kBlockSize), 0,
                       as_stream(stream), gradient, input_gradient, pairs,
                       sequence_size, heads, head_width, position_offset, base);
    check_launch("rope_split_half_bias_bthd_backward_kernel");
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
                           std::int64_t sequence, void* stream,
                           bool use_128_threads) {
    if (rows == 0) return;
    if (sequence >= 256) {
        if (sequence <= static_cast<std::int64_t>(kBlockSize) * 8) {
            const auto threads = use_128_threads && sequence <= 1024
                                     ? 128U : kBlockSize;
            hipLaunchKernelGGL(causal_softmax_rows_register_kernel,
                               dim3(static_cast<unsigned>(rows)), dim3(threads), 0,
                               as_stream(stream), scores, output, rows, sequence);
            check_launch("causal_softmax_rows_register_kernel");
            return;
        }
        hipLaunchKernelGGL(causal_softmax_rows_kernel,
                           dim3(static_cast<unsigned>(rows)), dim3(kBlockSize), 0,
                           as_stream(stream), scores, output, rows, sequence);
        check_launch("causal_softmax_rows_kernel");
        return;
    }
    hipLaunchKernelGGL(causal_softmax_kernel, dim3(grid_size(rows)), dim3(kBlockSize), 0,
                       as_stream(stream), scores, output, rows, sequence);
    check_launch("causal_softmax_kernel");
}

void launch_causal_softmax_backward(const float* output, const float* gradient,
                                    float* input_gradient, std::int64_t rows,
                                    std::int64_t sequence, void* stream) {
    if (rows == 0) return;
    if (sequence >= 256) {
        hipLaunchKernelGGL(causal_softmax_backward_rows_kernel,
                           dim3(static_cast<unsigned>(rows)), dim3(kBlockSize), 0,
                           as_stream(stream), output, gradient, input_gradient,
                           rows, sequence);
        check_launch("causal_softmax_backward_rows_kernel");
        return;
    }
    hipLaunchKernelGGL(causal_softmax_backward_kernel, dim3(grid_size(rows)),
                       dim3(kBlockSize), 0, as_stream(stream), output, gradient,
                       input_gradient, rows, sequence);
    check_launch("causal_softmax_backward_kernel");
}

void launch_causal_gqa_attention(
    const float* query, const float* key, const float* value, float* output,
    float* saved_probabilities,
    std::int64_t batches, std::int64_t heads, std::int64_t kv_heads,
    std::int64_t sequence, std::int64_t width, std::int64_t repeats,
    float scale, void* stream) {
    const auto rows = batches * heads * sequence;
    const auto shared_bytes = static_cast<std::size_t>(
        sequence + kBlockSize) * sizeof(float);
    hipLaunchKernelGGL(causal_gqa_attention_kernel,
                       dim3(static_cast<unsigned>(rows)), dim3(kBlockSize), shared_bytes,
                       as_stream(stream), query, key, value, output,
                       saved_probabilities, batches, heads, kv_heads, sequence,
                       width, repeats, scale);
    check_launch("causal_gqa_attention_kernel");
}

void launch_causal_gqa_attention_backward(
    const float* query, const float* key, const float* value,
    const float* output_gradient, float* query_gradient,
    float* key_gradient, float* value_gradient,
    std::int64_t batches, std::int64_t heads, std::int64_t kv_heads,
    std::int64_t sequence, std::int64_t width, std::int64_t repeats,
    float scale, void* stream) {
    const auto rows = batches * heads * sequence;
    const auto shared_bytes = static_cast<std::size_t>(sequence) * 2U * sizeof(float);
    hipLaunchKernelGGL(causal_gqa_attention_backward_kernel,
                       dim3(static_cast<unsigned>(rows)), dim3(kBlockSize), shared_bytes,
                       as_stream(stream), query, key, value, output_gradient,
                       query_gradient, key_gradient, value_gradient, batches, heads,
                       kv_heads, sequence, width, repeats, scale);
    check_launch("causal_gqa_attention_backward_kernel");
}

void launch_causal_gqa_attention_backward_rows(
    const float* query, const float* key, const float* value,
    const float* output_gradient, float* query_gradient,
    float* probabilities, float* scaled_score_gradients,
    std::int64_t batches, std::int64_t heads, std::int64_t kv_heads,
    std::int64_t sequence, std::int64_t width, std::int64_t repeats,
    float scale, void* stream) {
    const auto rows = batches * heads * sequence;
    const auto shared_bytes = static_cast<std::size_t>(sequence) * 2U * sizeof(float);
    hipLaunchKernelGGL(causal_gqa_attention_backward_rows_kernel,
                       dim3(static_cast<unsigned>(rows)), dim3(kBlockSize), shared_bytes,
                       as_stream(stream), query, key, value, output_gradient,
                       query_gradient, probabilities, scaled_score_gradients,
                       batches, heads, kv_heads, sequence, width, repeats, scale);
    check_launch("causal_gqa_attention_backward_rows_kernel");
}

void launch_causal_gqa_attention_backward_saved_rows(
    const float* key, const float* value, const float* output_gradient,
    const float* probabilities, float* query_gradient,
    float* scaled_score_gradients,
    std::int64_t batches, std::int64_t heads, std::int64_t kv_heads,
    std::int64_t sequence, std::int64_t width, std::int64_t repeats,
    float scale, void* stream) {
    const auto rows = batches * heads * sequence;
    const auto shared_bytes = static_cast<std::size_t>(sequence) * sizeof(float);
    hipLaunchKernelGGL(causal_gqa_attention_backward_saved_rows_kernel,
                       dim3(static_cast<unsigned>(rows)), dim3(kBlockSize), shared_bytes,
                       as_stream(stream), key, value, output_gradient, probabilities,
                       query_gradient, scaled_score_gradients, batches, heads,
                       kv_heads, sequence, width, repeats, scale);
    check_launch("causal_gqa_attention_backward_saved_rows_kernel");
}

void launch_repeat_interleave(const float* input, float* output,
                              std::int64_t output_elements, std::int64_t repeated_width,
                              std::int64_t inner, std::int64_t repeats, void* stream) {
    if (output_elements == 0) return;
    hipLaunchKernelGGL((repeat_interleave_typed_kernel<float>),
                       dim3(grid_size(output_elements)), dim3(kBlockSize), 0,
                       as_stream(stream), input, output, output_elements,
                       repeated_width, inner, repeats);
    check_launch("repeat_interleave_kernel");
}

void launch_repeat_interleave_bf16_to_float(
    const void* input, float* output, std::int64_t output_elements,
    std::int64_t repeated_width, std::int64_t inner,
    std::int64_t repeats, void* stream) {
    if (output_elements == 0) return;
    hipLaunchKernelGGL((repeat_interleave_typed_kernel<hip_bfloat16>),
                       dim3(grid_size(output_elements)), dim3(kBlockSize), 0,
                       as_stream(stream),
                       static_cast<const hip_bfloat16*>(input), output,
                       output_elements, repeated_width, inner, repeats);
    check_launch("repeat_interleave_bf16_to_float_kernel");
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

void launch_repeat_gqa_kv_bthd(
    const float* key, const float* value, float* expanded_key,
    float* expanded_value, std::int64_t batches, std::int64_t kv_heads,
    std::int64_t sequence, std::int64_t width, std::int64_t repeats,
    void* stream) {
    const auto heads = kv_heads * repeats;
    const auto output_elements = batches * heads * sequence * width;
    if (output_elements == 0) return;
    hipLaunchKernelGGL(
        repeat_gqa_kv_bthd_kernel, dim3(grid_size(output_elements)),
        dim3(kBlockSize), 0, as_stream(stream), key, value, expanded_key,
        expanded_value, output_elements, heads, kv_heads, sequence, width,
        repeats);
    check_launch("repeat_gqa_kv_bthd_kernel");
}

void launch_repeat_gqa_kv_bthd_backward(
    const float* key_gradient, const float* value_gradient,
    float* reduced_key_gradient, float* reduced_value_gradient,
    std::int64_t batches, std::int64_t kv_heads,
    std::int64_t sequence, std::int64_t width, std::int64_t repeats,
    void* stream) {
    const auto heads = kv_heads * repeats;
    const auto output_elements = batches * kv_heads * sequence * width;
    if (output_elements == 0) return;
    hipLaunchKernelGGL(
        repeat_gqa_kv_bthd_backward_kernel,
        dim3(grid_size(output_elements)), dim3(kBlockSize), 0,
        as_stream(stream), key_gradient, value_gradient,
        reduced_key_gradient, reduced_value_gradient, output_elements,
        heads, kv_heads, sequence, width, repeats);
    check_launch("repeat_gqa_kv_bthd_backward_kernel");
}

}  // namespace microllm::ops::hip
