#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include <hip/hip_runtime.h>
#include <rocwmma/rocwmma.hpp>

#include <microllm/ops/ops.h>
#include <microllm/runtime/runtime.h>

#ifndef MICROLLM_ROCWMMA_VERSION
#define MICROLLM_ROCWMMA_VERSION "unknown"
#endif

namespace {

constexpr std::uint32_t kTile = 32;
constexpr std::uint32_t kTileK = 16;
constexpr std::uint32_t kWaveSize = 64;
constexpr std::uint32_t kScalarThreads = 256;
using BFloat16 = rocwmma::bfloat16_t;

void check_hip(hipError_t status, const char* operation) {
    if (status != hipSuccess) {
        throw std::runtime_error(std::string(operation) + ": " +
                                 hipGetErrorString(status));
    }
}

struct Options {
    std::int64_t sequence = 512;
    std::int64_t width = 128;
    std::int64_t heads = 14;
    std::int64_t kv_heads = 2;
    std::int64_t worker_threads = 256;
    int warmup = 5;
    int repetitions = 20;
};

std::int64_t integer(const char* value, const char* name) {
    char* end = nullptr;
    const auto result = std::strtoll(value, &end, 10);
    if (end == value || *end != '\0') {
        throw std::invalid_argument(std::string("invalid ") + name);
    }
    return result;
}

Options parse(int argc, char** argv) {
    Options result;
    for (int index = 1; index < argc; index += 2) {
        if (index + 1 >= argc) {
            throw std::invalid_argument("option is missing a value");
        }
        const std::string_view name(argv[index]);
        if (name == "--sequence") {
            result.sequence = integer(argv[index + 1], "sequence");
        } else if (name == "--width") {
            result.width = integer(argv[index + 1], "width");
        } else if (name == "--heads") {
            result.heads = integer(argv[index + 1], "heads");
        } else if (name == "--kv-heads") {
            result.kv_heads = integer(argv[index + 1], "kv-heads");
        } else if (name == "--worker-threads") {
            result.worker_threads = integer(argv[index + 1], "worker-threads");
        } else if (name == "--warmup") {
            result.warmup = static_cast<int>(integer(argv[index + 1], "warmup"));
        } else if (name == "--repetitions") {
            result.repetitions =
                static_cast<int>(integer(argv[index + 1], "repetitions"));
        } else {
            throw std::invalid_argument("unknown option: " + std::string(name));
        }
    }
    if (result.sequence <= 0 || result.sequence > 4096 ||
        result.sequence % kTile != 0 ||
        (result.width != 64 && result.width != 128) ||
        result.heads <= 0 || result.heads > 256 ||
        result.kv_heads <= 0 || result.kv_heads > result.heads ||
        result.heads % result.kv_heads != 0 ||
        (result.worker_threads != 64 && result.worker_threads != 128 &&
         result.worker_threads != 256 && result.worker_threads != 512) ||
        result.worker_threads < result.width * 2 ||
        result.warmup < 0 || result.repetitions <= 0) {
        throw std::invalid_argument(
            "online Attention requires T divisible by 32 and width 64 or 128");
    }
    return result;
}

class DeviceBuffer {
public:
    explicit DeviceBuffer(std::size_t bytes) {
        check_hip(hipMalloc(&value_, bytes), "hipMalloc");
    }
    ~DeviceBuffer() {
        if (value_ != nullptr) (void)hipFree(value_);
    }
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
    template <typename T>
    T* as() noexcept {
        return static_cast<T*>(value_);
    }
private:
    void* value_ = nullptr;
};

template <std::uint32_t Width, std::uint32_t WorkerThreads>
__global__ __launch_bounds__(WorkerThreads)
void rocwmma_online_attention_kernel(
    std::uint32_t sequence, std::uint32_t kv_heads,
    const BFloat16* query, const BFloat16* key,
    const BFloat16* value, float scale, float* output) {
    using FragA = rocwmma::fragment<rocwmma::matrix_a, kTile, kTile, kTileK,
                                    BFloat16, rocwmma::row_major>;
    using FragB = rocwmma::fragment<rocwmma::matrix_b, kTile, kTile, kTileK,
                                    BFloat16, rocwmma::col_major>;
    using FragAcc = rocwmma::fragment<rocwmma::accumulator, kTile, kTile,
                                      kTileK, float>;
    extern __shared__ float shared[];
    float* weights = shared;
    float* row_maximum = weights + kTile * kTile;
    float* row_sum = row_maximum + kTile;
    float* row_alpha = row_sum + kTile;
    float* accumulator_output = row_alpha + kTile;
    BFloat16* weights_bf16 = reinterpret_cast<BFloat16*>(
        accumulator_output + kTile * Width);
    const auto thread = threadIdx.x;
    const auto query_head = blockIdx.y;
    const auto query_heads = gridDim.y;
    const auto group_size = query_heads / kv_heads;
    const auto kv_head = query_head / group_size;
    const auto query_head_offset = query_head * sequence * Width;
    const auto kv_head_offset = kv_head * sequence * Width;
    const auto query_base = blockIdx.x * kTile;

    for (std::uint32_t index = thread; index < kTile * Width;
         index += WorkerThreads) {
        accumulator_output[index] = 0.0F;
    }
    if (thread < kTile) {
        row_maximum[thread] = -std::numeric_limits<float>::infinity();
        row_sum[thread] = 0.0F;
        row_alpha[thread] = 0.0F;
    }
    __syncthreads();

    for (std::uint32_t key_base = 0; key_base <= query_base;
         key_base += kTile) {
        if (thread < kWaveSize) {
            FragA fragment_query;
            FragB fragment_key;
            FragAcc score;
            rocwmma::fill_fragment(score, 0.0F);
            for (std::uint32_t offset = 0; offset < Width; offset += kTileK) {
                rocwmma::load_matrix_sync(
                    fragment_query,
                    query + query_head_offset + query_base * Width + offset,
                    Width);
                rocwmma::load_matrix_sync(
                    fragment_key,
                    key + kv_head_offset + key_base * Width + offset, Width);
                rocwmma::mma_sync(score, fragment_query, fragment_key, score);
            }
            rocwmma::store_matrix_sync(weights, score, kTile,
                                       rocwmma::mem_row_major);
        }
        __syncthreads();

        if (thread < kTile) {
            const auto query_index = query_base + thread;
            float tile_maximum = -std::numeric_limits<float>::infinity();
            for (std::uint32_t column = 0; column < kTile; ++column) {
                const auto key_index = key_base + column;
                if (key_index <= query_index && key_index < sequence) {
                    tile_maximum = fmaxf(
                        tile_maximum, weights[thread * kTile + column] * scale);
                }
            }
            const auto old_maximum = row_maximum[thread];
            const auto new_maximum = fmaxf(old_maximum, tile_maximum);
            const auto alpha = row_sum[thread] == 0.0F
                                   ? 0.0F
                                   : expf(old_maximum - new_maximum);
            float tile_sum = 0.0F;
            for (std::uint32_t column = 0; column < kTile; ++column) {
                const auto key_index = key_base + column;
                const auto weight =
                    key_index <= query_index && key_index < sequence
                        ? expf(weights[thread * kTile + column] * scale -
                               new_maximum)
                        : 0.0F;
                weights[thread * kTile + column] = weight;
                tile_sum += weight;
            }
            row_maximum[thread] = new_maximum;
            row_alpha[thread] = alpha;
            row_sum[thread] = row_sum[thread] * alpha + tile_sum;
        }
        __syncthreads();

        for (std::uint32_t index = thread; index < kTile * kTile;
             index += WorkerThreads) {
            weights_bf16[index] = BFloat16(weights[index]);
        }
        for (std::uint32_t index = thread; index < kTile * Width;
             index += WorkerThreads) {
            accumulator_output[index] *= row_alpha[index / Width];
        }
        __syncthreads();

        using WeightFragment = rocwmma::fragment<
            rocwmma::matrix_a, kTile, kTile, kTileK,
            BFloat16, rocwmma::row_major>;
        using ValueFragment = rocwmma::fragment<
            rocwmma::matrix_b, kTile, kTile, kTileK,
            BFloat16, rocwmma::row_major>;
        using OutputFragment = rocwmma::fragment<
            rocwmma::accumulator, kTile, kTile, kTileK, float>;
        const auto output_tile = thread / kWaveSize;
        if (output_tile < Width / kTile) {
            const auto output_column = output_tile * kTile;
            WeightFragment weight_fragment;
            ValueFragment value_fragment;
            OutputFragment output_fragment;
            rocwmma::load_matrix_sync(
                output_fragment, accumulator_output + output_column,
                Width, rocwmma::mem_row_major);
            for (std::uint32_t offset = 0; offset < kTile;
                 offset += kTileK) {
                rocwmma::load_matrix_sync(
                    weight_fragment, weights_bf16 + offset, kTile);
                rocwmma::load_matrix_sync(
                    value_fragment,
                    value + kv_head_offset +
                        (key_base + offset) * Width + output_column,
                    Width);
                rocwmma::mma_sync(
                    output_fragment, weight_fragment, value_fragment,
                    output_fragment);
            }
            rocwmma::store_matrix_sync(
                accumulator_output + output_column, output_fragment,
                Width, rocwmma::mem_row_major);
        }
        __syncthreads();
    }

    for (std::uint32_t index = thread; index < kTile * Width;
         index += WorkerThreads) {
        const auto row = index / Width;
        output[query_head_offset +
               (query_base + row) * Width + index % Width] =
            accumulator_output[index] / row_sum[row];
    }
}

template <std::uint32_t Width, std::uint32_t WorkerThreads>
void launch_online_attention(
    dim3 grid, std::size_t shared_bytes, std::uint32_t sequence,
    std::uint32_t kv_heads,
    const BFloat16* query, const BFloat16* key, const BFloat16* value,
    float scale, float* output) {
    rocwmma_online_attention_kernel<Width, WorkerThreads>
        <<<grid, WorkerThreads, shared_bytes>>>(
            sequence, kv_heads, query, key, value, scale, output);
}

template <std::uint32_t Width>
__global__ __launch_bounds__(kScalarThreads)
void scalar_fused_attention_kernel(
    std::uint32_t sequence, std::uint32_t kv_heads,
    const BFloat16* query, const BFloat16* key,
    const BFloat16* value, float scale, float* output) {
    extern __shared__ float shared[];
    float* scores = shared;
    float* reduction = scores + sequence;
    const auto query_index = blockIdx.x;
    const auto query_head = blockIdx.y;
    const auto query_heads = gridDim.y;
    const auto kv_head = query_head / (query_heads / kv_heads);
    const auto query_head_offset = query_head * sequence * Width;
    const auto kv_head_offset = kv_head * sequence * Width;
    float local_maximum = -std::numeric_limits<float>::infinity();
    for (std::uint32_t key_index = threadIdx.x; key_index <= query_index;
         key_index += blockDim.x) {
        float score = 0.0F;
        for (std::uint32_t column = 0; column < Width; ++column) {
            score += static_cast<float>(
                         query[query_head_offset + query_index * Width + column]) *
                     static_cast<float>(
                         key[kv_head_offset + key_index * Width + column]);
        }
        score *= scale;
        scores[key_index] = score;
        local_maximum = fmaxf(local_maximum, score);
    }
    reduction[threadIdx.x] = local_maximum;
    __syncthreads();
    for (std::uint32_t stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (threadIdx.x < stride) {
            reduction[threadIdx.x] =
                fmaxf(reduction[threadIdx.x], reduction[threadIdx.x + stride]);
        }
        __syncthreads();
    }
    const auto maximum = reduction[0];
    float local_sum = 0.0F;
    for (std::uint32_t key_index = threadIdx.x; key_index <= query_index;
         key_index += blockDim.x) {
        const auto weight = expf(scores[key_index] - maximum);
        scores[key_index] = weight;
        local_sum += weight;
    }
    reduction[threadIdx.x] = local_sum;
    __syncthreads();
    for (std::uint32_t stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (threadIdx.x < stride) {
            reduction[threadIdx.x] += reduction[threadIdx.x + stride];
        }
        __syncthreads();
    }
    const auto denominator = reduction[0];
    for (std::uint32_t column = threadIdx.x; column < Width;
         column += blockDim.x) {
        float sum = 0.0F;
        for (std::uint32_t key_index = 0; key_index <= query_index;
             ++key_index) {
            sum += scores[key_index] *
                   static_cast<float>(
                       value[kv_head_offset + key_index * Width + column]);
        }
        output[query_head_offset + query_index * Width + column] =
            sum / denominator;
    }
}

double percentile(std::vector<float> values, double fraction) {
    std::sort(values.begin(), values.end());
    const auto position = fraction * static_cast<double>(values.size() - 1);
    const auto lower = static_cast<std::size_t>(std::floor(position));
    const auto upper = static_cast<std::size_t>(std::ceil(position));
    const auto weight = position - static_cast<double>(lower);
    return static_cast<double>(values[lower]) * (1.0 - weight) +
           static_cast<double>(values[upper]) * weight;
}

template <typename Launch>
std::vector<float> time_kernel(const Options& options, Launch launch) {
    for (int iteration = 0; iteration < options.warmup; ++iteration) launch();
    check_hip(hipDeviceSynchronize(), "hipDeviceSynchronize(warmup)");
    hipEvent_t start = nullptr;
    hipEvent_t finish = nullptr;
    check_hip(hipEventCreate(&start), "hipEventCreate(start)");
    check_hip(hipEventCreate(&finish), "hipEventCreate(finish)");
    std::vector<float> times;
    for (int iteration = 0; iteration < options.repetitions; ++iteration) {
        check_hip(hipEventRecord(start), "hipEventRecord(start)");
        launch();
        check_hip(hipEventRecord(finish), "hipEventRecord(finish)");
        check_hip(hipEventSynchronize(finish), "hipEventSynchronize(finish)");
        float elapsed = 0.0F;
        check_hip(hipEventElapsedTime(&elapsed, start, finish),
                  "hipEventElapsedTime");
        times.push_back(elapsed);
    }
    (void)hipEventDestroy(start);
    (void)hipEventDestroy(finish);
    return times;
}

struct Error {
    float maximum = 0.0F;
    double rms = 0.0;
    bool finite = true;
};

Error compare(const std::vector<float>& actual,
              const std::vector<float>& expected) {
    Error result;
    double squared = 0.0;
    for (std::size_t index = 0; index < actual.size(); ++index) {
        result.finite = result.finite && std::isfinite(actual[index]);
        const auto difference = std::abs(actual[index] - expected[index]);
        result.maximum = std::max(result.maximum, difference);
        squared += static_cast<double>(difference) * difference;
    }
    result.rms = std::sqrt(squared / static_cast<double>(actual.size()));
    return result;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const auto options = parse(argc, argv);
        int device_count = 0;
        check_hip(hipGetDeviceCount(&device_count), "hipGetDeviceCount");
        if (device_count <= 0) {
            throw std::runtime_error("online Attention benchmark needs a HIP device");
        }
        hipDeviceProp_t properties{};
        check_hip(hipGetDeviceProperties(&properties, 0),
                  "hipGetDeviceProperties");
        if (properties.warpSize != static_cast<int>(kWaveSize)) {
            throw std::runtime_error("online Attention prototype requires wave64");
        }
        const auto query_elements = static_cast<std::size_t>(
            options.heads * options.sequence * options.width);
        const auto kv_elements = static_cast<std::size_t>(
            options.kv_heads * options.sequence * options.width);
        std::vector<BFloat16> query(query_elements);
        std::vector<BFloat16> key(kv_elements);
        std::vector<BFloat16> value(kv_elements);
        std::vector<float> query_fp32(query_elements);
        std::vector<float> key_fp32(kv_elements);
        std::vector<float> value_bthd_fp32(kv_elements);
        for (std::size_t index = 0; index < query.size(); ++index) {
            query[index] = BFloat16(
                static_cast<float>(static_cast<int>(index % 29) - 14) / 32.0F);
            query_fp32[index] = static_cast<float>(query[index]);
        }
        for (std::size_t index = 0; index < key.size(); ++index) {
            key[index] = BFloat16(
                static_cast<float>(static_cast<int>(index % 31) - 15) / 31.0F);
            value[index] = BFloat16(
                static_cast<float>(static_cast<int>(index % 37) - 18) / 37.0F);
            key_fp32[index] = static_cast<float>(key[index]);
        }
        for (std::int64_t kv_head = 0; kv_head < options.kv_heads; ++kv_head) {
            for (std::int64_t token = 0; token < options.sequence; ++token) {
                for (std::int64_t inner = 0; inner < options.width; ++inner) {
                    const auto source = static_cast<std::size_t>(
                        (kv_head * options.sequence + token) * options.width + inner);
                    const auto destination = static_cast<std::size_t>(
                        (token * options.kv_heads + kv_head) * options.width + inner);
                    value_bthd_fp32[destination] = static_cast<float>(value[source]);
                }
            }
        }
        const auto scale = 1.0F / std::sqrt(static_cast<float>(options.width));
        std::vector<float> reference(query_elements, 0.0F);
        std::vector<float> scores(static_cast<std::size_t>(options.sequence));
        for (std::int64_t head = 0; head < options.heads; ++head) {
          const auto kv_head = head / (options.heads / options.kv_heads);
          const auto query_head_offset = head * options.sequence * options.width;
          const auto kv_head_offset = kv_head * options.sequence * options.width;
          for (std::int64_t row = 0; row < options.sequence; ++row) {
            float maximum = -std::numeric_limits<float>::infinity();
            for (std::int64_t column = 0; column <= row; ++column) {
                float score = 0.0F;
                for (std::int64_t inner = 0; inner < options.width; ++inner) {
                    score += static_cast<float>(query[static_cast<std::size_t>(
                                 query_head_offset + row * options.width + inner)]) *
                             static_cast<float>(key[static_cast<std::size_t>(
                                 kv_head_offset + column * options.width + inner)]);
                }
                scores[static_cast<std::size_t>(column)] = score * scale;
                maximum = std::max(maximum, score * scale);
            }
            float denominator = 0.0F;
            for (std::int64_t column = 0; column <= row; ++column) {
                scores[static_cast<std::size_t>(column)] =
                    std::exp(scores[static_cast<std::size_t>(column)] - maximum);
                denominator += scores[static_cast<std::size_t>(column)];
            }
            for (std::int64_t inner = 0; inner < options.width; ++inner) {
                float sum = 0.0F;
                for (std::int64_t column = 0; column <= row; ++column) {
                    sum += scores[static_cast<std::size_t>(column)] *
                           static_cast<float>(value[static_cast<std::size_t>(
                               kv_head_offset + column * options.width + inner)]);
                }
                reference[static_cast<std::size_t>(
                    query_head_offset + row * options.width + inner)] =
                        sum / denominator;
            }
          }
        }

        DeviceBuffer query_device(query_elements * sizeof(BFloat16));
        DeviceBuffer key_device(kv_elements * sizeof(BFloat16));
        DeviceBuffer value_device(kv_elements * sizeof(BFloat16));
        DeviceBuffer online_output_device(query_elements * sizeof(float));
        DeviceBuffer scalar_output_device(query_elements * sizeof(float));
        check_hip(hipMemcpy(query_device.as<BFloat16>(), query.data(),
                            query_elements * sizeof(BFloat16), hipMemcpyHostToDevice),
                  "hipMemcpy(query)");
        check_hip(hipMemcpy(key_device.as<BFloat16>(), key.data(),
                            kv_elements * sizeof(BFloat16), hipMemcpyHostToDevice),
                  "hipMemcpy(key)");
        check_hip(hipMemcpy(value_device.as<BFloat16>(), value.data(),
                            kv_elements * sizeof(BFloat16), hipMemcpyHostToDevice),
                  "hipMemcpy(value)");
        const auto device = microllm::Device::hip(0);
        const auto current_query = microllm::Tensor::from_vector(
            query_fp32, {1, options.heads, options.sequence, options.width}).to(device);
        const auto current_key = microllm::Tensor::from_vector(
            key_fp32, {1, options.kv_heads, options.sequence, options.width}).to(device);
        const auto current_value = microllm::Tensor::from_vector(
            value_bthd_fp32,
            {1, options.sequence, options.kv_heads, options.width}).to(device);
        const auto width_size = static_cast<std::size_t>(options.width);
        const auto sequence_size = static_cast<std::size_t>(options.sequence);
        const auto online_shared =
            (kTile * kTile + 3 * kTile + kTile * width_size) * sizeof(float) +
            kTile * kTile * sizeof(BFloat16);
        const auto scalar_shared =
            (sequence_size + kScalarThreads) * sizeof(float);
        const dim3 online_grid(
            static_cast<unsigned>(options.sequence / kTile),
            static_cast<unsigned>(options.heads));
        const auto launch_online = [&] {
            const auto sequence = static_cast<std::uint32_t>(options.sequence);
            const auto kv_heads = static_cast<std::uint32_t>(options.kv_heads);
            auto* query_pointer = query_device.as<BFloat16>();
            auto* key_pointer = key_device.as<BFloat16>();
            auto* value_pointer = value_device.as<BFloat16>();
            auto* output_pointer = online_output_device.as<float>();
#define MICROLLM_LAUNCH_ONLINE(WIDTH, THREADS) \
    launch_online_attention<WIDTH, THREADS>( \
        online_grid, online_shared, sequence, kv_heads, query_pointer, \
        key_pointer, value_pointer, scale, output_pointer)
            if (options.width == 64 && options.worker_threads == 64) {
                MICROLLM_LAUNCH_ONLINE(64, 64);
            } else if (options.width == 64 && options.worker_threads == 128) {
                MICROLLM_LAUNCH_ONLINE(64, 128);
            } else if (options.width == 64 && options.worker_threads == 256) {
                MICROLLM_LAUNCH_ONLINE(64, 256);
            } else if (options.width == 64) {
                MICROLLM_LAUNCH_ONLINE(64, 512);
            } else if (options.worker_threads == 64) {
                MICROLLM_LAUNCH_ONLINE(128, 64);
            } else if (options.worker_threads == 128) {
                MICROLLM_LAUNCH_ONLINE(128, 128);
            } else if (options.worker_threads == 256) {
                MICROLLM_LAUNCH_ONLINE(128, 256);
            } else {
                MICROLLM_LAUNCH_ONLINE(128, 512);
            }
#undef MICROLLM_LAUNCH_ONLINE
            check_hip(hipGetLastError(), "rocwmma_online_attention_kernel");
        };
        const dim3 scalar_grid(static_cast<unsigned>(options.sequence),
                               static_cast<unsigned>(options.heads));
        const auto launch_scalar = [&] {
            if (options.width == 64) {
                scalar_fused_attention_kernel<64>
                    <<<scalar_grid, kScalarThreads, scalar_shared>>>(
                        static_cast<std::uint32_t>(options.sequence),
                        static_cast<std::uint32_t>(options.kv_heads),
                        query_device.as<BFloat16>(), key_device.as<BFloat16>(),
                        value_device.as<BFloat16>(), scale,
                        scalar_output_device.as<float>());
            } else {
                scalar_fused_attention_kernel<128>
                    <<<scalar_grid, kScalarThreads, scalar_shared>>>(
                        static_cast<std::uint32_t>(options.sequence),
                        static_cast<std::uint32_t>(options.kv_heads),
                        query_device.as<BFloat16>(), key_device.as<BFloat16>(),
                        value_device.as<BFloat16>(), scale,
                        scalar_output_device.as<float>());
            }
            check_hip(hipGetLastError(), "scalar_fused_attention_kernel");
        };
        microllm::Tensor current_output;
        const auto launch_current = [&] {
            current_output = microllm::ops::causal_gqa_attention_bthd(
                current_query, current_key, current_value,
                options.heads / options.kv_heads, scale);
        };

        launch_online();
        launch_scalar();
        launch_current();
        check_hip(hipDeviceSynchronize(), "hipDeviceSynchronize(correctness)");
        std::vector<float> online_output(query_elements);
        std::vector<float> scalar_output(query_elements);
        std::vector<float> current_output_hstd(query_elements);
        check_hip(hipMemcpy(online_output.data(), online_output_device.as<float>(),
                            query_elements * sizeof(float), hipMemcpyDeviceToHost),
                  "hipMemcpy(online output)");
        check_hip(hipMemcpy(scalar_output.data(), scalar_output_device.as<float>(),
                            query_elements * sizeof(float), hipMemcpyDeviceToHost),
                  "hipMemcpy(scalar output)");
        const auto current_output_bthd = current_output.to_vector();
        for (std::int64_t head = 0; head < options.heads; ++head) {
            for (std::int64_t token = 0; token < options.sequence; ++token) {
                for (std::int64_t inner = 0; inner < options.width; ++inner) {
                    const auto source = static_cast<std::size_t>(
                        (token * options.heads + head) * options.width + inner);
                    const auto destination = static_cast<std::size_t>(
                        (head * options.sequence + token) * options.width + inner);
                    current_output_hstd[destination] = current_output_bthd[source];
                }
            }
        }
        const auto online_error = compare(online_output, reference);
        const auto scalar_error = compare(scalar_output, reference);
        const auto current_error = compare(current_output_hstd, reference);
        const bool accuracy_passed =
            online_error.finite && scalar_error.finite && current_error.finite &&
            online_error.maximum <= 2.0e-3F && online_error.rms <= 2.0e-4 &&
            scalar_error.maximum <= 3.0e-4F && scalar_error.rms <= 3.0e-5 &&
            current_error.maximum <= 3.0e-4F && current_error.rms <= 3.0e-5;
        if (!accuracy_passed) {
            std::cerr << "online error max=" << online_error.maximum
                      << " rms=" << online_error.rms
                      << "; scalar error max=" << scalar_error.maximum
                      << " rms=" << scalar_error.rms
                      << "; current error max=" << current_error.maximum
                      << " rms=" << current_error.rms << '\n';
            throw std::runtime_error(
                "rocWMMA online Attention complete-output gate failed");
        }

        const auto online_times = time_kernel(options, launch_online);
        const auto scalar_times = time_kernel(options, launch_scalar);
        const auto current_times = time_kernel(options, launch_current);
        const auto online_p50 = percentile(online_times, 0.50);
        const auto scalar_p50 = percentile(scalar_times, 0.50);
        const auto current_p50 = percentile(current_times, 0.50);
        const auto current_score_bytes = static_cast<std::uint64_t>(options.heads) *
            static_cast<std::uint64_t>(options.sequence) *
            static_cast<std::uint64_t>(options.sequence) * sizeof(float);
        std::cout << std::setprecision(9)
                  << "{\"schema_version\":1,\"status\":\"pass\""
                  << ",\"op\":\"bf16_causal_online_attention\""
                  << ",\"architecture\":\"" << properties.gcnArchName << "\""
                  << ",\"rocwmma_version\":\"" << MICROLLM_ROCWMMA_VERSION << "\""
                  << ",\"sequence\":" << options.sequence
                  << ",\"width\":" << options.width
                  << ",\"query_tile\":" << kTile
                  << ",\"key_tile\":" << kTile
                  << ",\"worker_threads\":" << options.worker_threads
                  << ",\"causal\":true,\"batch\":1"
                  << ",\"heads\":" << options.heads
                  << ",\"kv_heads\":" << options.kv_heads
                  << ",\"gqa\":" << (options.heads != options.kv_heads ? "true" : "false")
                  << ",\"tail_support\":false"
                  << ",\"pv_path\":\"rocwmma_bf16\""
                  << ",\"global_score_bytes\":0"
                  << ",\"current_score_bytes\":" << current_score_bytes
                  << ",\"online_shared_bytes\":" << online_shared
                  << ",\"scalar_shared_bytes\":" << scalar_shared
                  << ",\"complete_output_elements\":" << query_elements
                  << ",\"online_max_error\":" << online_error.maximum
                  << ",\"online_rms_error\":" << online_error.rms
                  << ",\"scalar_max_error\":" << scalar_error.maximum
                  << ",\"scalar_rms_error\":" << scalar_error.rms
                  << ",\"current_max_error\":" << current_error.maximum
                  << ",\"current_rms_error\":" << current_error.rms
                  << ",\"accuracy_passed\":" << (accuracy_passed ? "true" : "false")
                  << ",\"warmup\":" << options.warmup
                  << ",\"repetitions\":" << options.repetitions
                  << ",\"online_event_ms_p50\":" << online_p50
                  << ",\"online_event_ms_p95\":" << percentile(online_times, 0.95)
                  << ",\"scalar_event_ms_p50\":" << scalar_p50
                  << ",\"scalar_event_ms_p95\":" << percentile(scalar_times, 0.95)
                  << ",\"current_event_ms_p50\":" << current_p50
                  << ",\"current_event_ms_p95\":" << percentile(current_times, 0.95)
                  << ",\"online_over_scalar\":" << scalar_p50 / online_p50
                  << ",\"online_over_current\":" << current_p50 / online_p50
                  << "}\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "benchmark_rocwmma_online_attention: " << error.what() << '\n';
        return 2;
    }
}
