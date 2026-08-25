#include "kernels.h"

#include <cstdint>
#include <limits>

#include <hip/hip_runtime.h>
#include <rocwmma/rocwmma.hpp>

namespace microllm::ops::hip {
namespace {

constexpr std::uint32_t kTile = 32;
constexpr std::uint32_t kTileK = 16;
constexpr std::uint32_t kWaveSize = 64;
constexpr std::uint32_t kWorkerThreads = 512;
using BFloat16 = rocwmma::bfloat16_t;

template <std::uint32_t Width>
__global__ __launch_bounds__(kWorkerThreads)
void online_gqa_attention_bthd_kernel(
    std::uint32_t sequence, std::uint32_t kv_heads,
    const BFloat16* query, const BFloat16* key, const BFloat16* value,
    float scale, float* output) {
    using QueryFragment = rocwmma::fragment<
        rocwmma::matrix_a, kTile, kTile, kTileK,
        BFloat16, rocwmma::row_major>;
    using KeyFragment = rocwmma::fragment<
        rocwmma::matrix_b, kTile, kTile, kTileK,
        BFloat16, rocwmma::col_major>;
    using WeightFragment = rocwmma::fragment<
        rocwmma::matrix_a, kTile, kTile, kTileK,
        BFloat16, rocwmma::row_major>;
    using ValueFragment = rocwmma::fragment<
        rocwmma::matrix_b, kTile, kTile, kTileK,
        BFloat16, rocwmma::row_major>;
    using AccumulatorFragment = rocwmma::fragment<
        rocwmma::accumulator, kTile, kTile, kTileK, float>;

    extern __shared__ float shared[];
    float* weights = shared;
    float* row_maximum = weights + kTile * kTile;
    float* row_sum = row_maximum + kTile;
    float* row_alpha = row_sum + kTile;
    float* accumulator_output = row_alpha + kTile;
    BFloat16* weights_bf16 = reinterpret_cast<BFloat16*>(
        accumulator_output + kTile * Width);

    const auto thread = threadIdx.x;
    const auto batch = blockIdx.z;
    const auto query_head = blockIdx.y;
    const auto query_heads = gridDim.y;
    const auto kv_head = query_head / (query_heads / kv_heads);
    const auto query_base = blockIdx.x * kTile;
    const auto query_batch_stride =
        static_cast<std::uint64_t>(query_heads) * sequence * Width;
    const auto kv_batch_stride =
        static_cast<std::uint64_t>(kv_heads) * sequence * Width;
    const auto value_batch_stride =
        static_cast<std::uint64_t>(sequence) * kv_heads * Width;
    const auto output_batch_stride =
        static_cast<std::uint64_t>(sequence) * query_heads * Width;
    const auto query_head_offset =
        static_cast<std::uint64_t>(batch) * query_batch_stride +
        static_cast<std::uint64_t>(query_head) * sequence * Width;
    const auto kv_head_offset =
        static_cast<std::uint64_t>(batch) * kv_batch_stride +
        static_cast<std::uint64_t>(kv_head) * sequence * Width;
    const auto value_batch_offset =
        static_cast<std::uint64_t>(batch) * value_batch_stride;
    const auto output_batch_offset =
        static_cast<std::uint64_t>(batch) * output_batch_stride;

    for (std::uint32_t index = thread; index < kTile * Width;
         index += kWorkerThreads) {
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
            QueryFragment query_fragment;
            KeyFragment key_fragment;
            AccumulatorFragment score_fragment;
            rocwmma::fill_fragment(score_fragment, 0.0F);
            for (std::uint32_t offset = 0; offset < Width; offset += kTileK) {
                rocwmma::load_matrix_sync(
                    query_fragment,
                    query + query_head_offset + query_base * Width + offset,
                    Width);
                rocwmma::load_matrix_sync(
                    key_fragment,
                    key + kv_head_offset + key_base * Width + offset,
                    Width);
                rocwmma::mma_sync(
                    score_fragment, query_fragment, key_fragment,
                    score_fragment);
            }
            rocwmma::store_matrix_sync(
                weights, score_fragment, kTile, rocwmma::mem_row_major);
        }
        __syncthreads();

        if (thread < kTile) {
            const auto query_index = query_base + thread;
            float tile_maximum = -std::numeric_limits<float>::infinity();
            for (std::uint32_t column = 0; column < kTile; ++column) {
                const auto key_index = key_base + column;
                if (key_index <= query_index) {
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
                const auto weight = key_index <= query_index
                    ? expf(weights[thread * kTile + column] * scale - new_maximum)
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
             index += kWorkerThreads) {
            weights_bf16[index] = BFloat16(weights[index]);
        }
        for (std::uint32_t index = thread; index < kTile * Width;
             index += kWorkerThreads) {
            accumulator_output[index] *= row_alpha[index / Width];
        }
        __syncthreads();

        const auto output_tile = thread / kWaveSize;
        if (output_tile < Width / kTile) {
            const auto output_column = output_tile * kTile;
            WeightFragment weight_fragment;
            ValueFragment value_fragment;
            AccumulatorFragment output_fragment;
            rocwmma::load_matrix_sync(
                output_fragment, accumulator_output + output_column,
                Width, rocwmma::mem_row_major);
            for (std::uint32_t offset = 0; offset < kTile; offset += kTileK) {
                rocwmma::load_matrix_sync(
                    weight_fragment, weights_bf16 + offset, kTile);
                const auto value_offset =
                    value_batch_offset +
                    static_cast<std::uint64_t>(key_base + offset) *
                        kv_heads * Width +
                    static_cast<std::uint64_t>(kv_head) * Width +
                    output_column;
                rocwmma::load_matrix_sync(
                    value_fragment, value + value_offset, kv_heads * Width);
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
         index += kWorkerThreads) {
        const auto row = index / Width;
        const auto column = index % Width;
        const auto output_offset =
            output_batch_offset +
            static_cast<std::uint64_t>(query_base + row) * query_heads * Width +
            static_cast<std::uint64_t>(query_head) * Width + column;
        output[output_offset] = accumulator_output[index] / row_sum[row];
    }
}

}  // namespace

void launch_rocwmma_online_gqa_attention_bthd(
    const void* query_bf16, const void* key_bf16, const void* value_bf16,
    float* output, std::int64_t batches, std::int64_t heads,
    std::int64_t kv_heads, std::int64_t sequence, std::int64_t width,
    float scale, void* stream) {
    const auto shared_bytes =
        static_cast<std::size_t>(
            kTile * kTile + 3 * kTile + kTile * width) * sizeof(float) +
        kTile * kTile * sizeof(BFloat16);
    const dim3 grid(
        static_cast<unsigned>(sequence / kTile),
        static_cast<unsigned>(heads), static_cast<unsigned>(batches));
    auto* native_stream = reinterpret_cast<hipStream_t>(stream);
    if (width == 64) {
        online_gqa_attention_bthd_kernel<64>
            <<<grid, kWorkerThreads, shared_bytes, native_stream>>>(
                static_cast<std::uint32_t>(sequence),
                static_cast<std::uint32_t>(kv_heads),
                static_cast<const BFloat16*>(query_bf16),
                static_cast<const BFloat16*>(key_bf16),
                static_cast<const BFloat16*>(value_bf16), scale, output);
    } else {
        online_gqa_attention_bthd_kernel<128>
            <<<grid, kWorkerThreads, shared_bytes, native_stream>>>(
                static_cast<std::uint32_t>(sequence),
                static_cast<std::uint32_t>(kv_heads),
                static_cast<const BFloat16*>(query_bf16),
                static_cast<const BFloat16*>(key_bf16),
                static_cast<const BFloat16*>(value_bf16), scale, output);
    }
}

}  // namespace microllm::ops::hip
