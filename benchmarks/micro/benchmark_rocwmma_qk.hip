#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include <hip/hip_runtime.h>
#include <hipblaslt/hipblaslt.h>
#include <rocwmma/rocwmma.hpp>

#ifndef MICROLLM_ROCWMMA_VERSION
#define MICROLLM_ROCWMMA_VERSION "unknown"
#endif

namespace {

constexpr std::uint32_t kTileK = 16;
constexpr std::uint32_t kWaveSize = 64;
using BFloat16 = rocwmma::bfloat16_t;

void check_hip(hipError_t status, const char* operation) {
    if (status != hipSuccess) {
        throw std::runtime_error(std::string(operation) + ": " +
                                 hipGetErrorString(status));
    }
}

void check_blas(hipblasStatus_t status, const char* operation) {
    if (status != HIPBLAS_STATUS_SUCCESS) {
        throw std::runtime_error(std::string(operation) + " failed with status " +
                                 std::to_string(static_cast<int>(status)));
    }
}

class BlasHandle {
public:
    BlasHandle() { check_blas(hipblasLtCreate(&value_), "hipblasLtCreate"); }
    ~BlasHandle() { (void)hipblasLtDestroy(value_); }
    hipblasLtHandle_t get() const noexcept { return value_; }
private:
    hipblasLtHandle_t value_ = nullptr;
};

class BlasDescription {
public:
    BlasDescription() {
        check_blas(hipblasLtMatmulDescCreate(&value_, HIPBLAS_COMPUTE_32F,
                                             HIP_R_32F),
                   "hipblasLtMatmulDescCreate");
    }
    ~BlasDescription() { (void)hipblasLtMatmulDescDestroy(value_); }
    hipblasLtMatmulDesc_t get() const noexcept { return value_; }
private:
    hipblasLtMatmulDesc_t value_ = nullptr;
};

class BlasLayout {
public:
    BlasLayout(hipDataType dtype, std::uint64_t rows, std::uint64_t columns,
               std::int64_t leading) {
        check_blas(hipblasLtMatrixLayoutCreate(&value_, dtype, rows, columns,
                                               leading),
                   "hipblasLtMatrixLayoutCreate");
    }
    ~BlasLayout() { (void)hipblasLtMatrixLayoutDestroy(value_); }
    hipblasLtMatrixLayout_t get() const noexcept { return value_; }
private:
    hipblasLtMatrixLayout_t value_ = nullptr;
};

struct Options {
    std::int64_t rows = 512;
    std::int64_t columns = 512;
    std::int64_t inner = 128;
    std::int64_t tile = 32;
    std::int64_t waves_per_block = 1;
    int warmup = 10;
    int repetitions = 50;
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
        if (name == "--rows") result.rows = integer(argv[index + 1], "rows");
        else if (name == "--columns") {
            result.columns = integer(argv[index + 1], "columns");
        }
        else if (name == "--inner") {
            result.inner = integer(argv[index + 1], "inner");
        }
        else if (name == "--tile") result.tile = integer(argv[index + 1], "tile");
        else if (name == "--waves-per-block") {
            result.waves_per_block = integer(argv[index + 1], "waves-per-block");
        } else if (name == "--warmup") {
            result.warmup = static_cast<int>(integer(argv[index + 1], "warmup"));
        } else if (name == "--repetitions") {
            result.repetitions =
                static_cast<int>(integer(argv[index + 1], "repetitions"));
        } else {
            throw std::invalid_argument("unknown option: " + std::string(name));
        }
    }
    if (result.rows <= 0 || result.columns <= 0 || result.inner <= 0 ||
        result.rows > 4096 || result.columns > 4096 || result.inner > 512 ||
        (result.tile != 16 && result.tile != 32) ||
        result.rows % result.tile != 0 || result.columns % result.tile != 0 ||
        result.inner % kTileK != 0 || result.warmup < 0 ||
        result.repetitions <= 0 ||
        (result.waves_per_block != 1 && result.waves_per_block != 2 &&
         result.waves_per_block != 4 && result.waves_per_block != 8 &&
         result.waves_per_block != 16)) {
        throw std::invalid_argument(
            "rocWMMA QK requires tile-aligned M/N/K within 4096/4096/512");
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

template <std::uint32_t Tile>
__global__ void rocwmma_qk_kernel(std::uint32_t rows, std::uint32_t columns,
                                  std::uint32_t inner, const BFloat16* query,
                                  const BFloat16* key_column_major,
                                  float* output) {
    using FragA = rocwmma::fragment<rocwmma::matrix_a, Tile, Tile, kTileK,
                                    BFloat16, rocwmma::row_major>;
    using FragB = rocwmma::fragment<rocwmma::matrix_b, Tile, Tile, kTileK,
                                    BFloat16, rocwmma::col_major>;
    using FragAcc = rocwmma::fragment<rocwmma::accumulator, Tile, Tile,
                                      kTileK, float>;
    const auto global_wave =
        (blockIdx.x * blockDim.x + threadIdx.x) / kWaveSize;
    const auto row = global_wave * Tile;
    const auto column = blockIdx.y * Tile;
    if (row >= rows || column >= columns) return;

    FragA fragment_a;
    FragB fragment_b;
    FragAcc accumulator;
    rocwmma::fill_fragment(accumulator, 0.0F);
    for (std::uint32_t offset = 0; offset < inner; offset += kTileK) {
        rocwmma::load_matrix_sync(fragment_a, query + row * inner + offset, inner);
        rocwmma::load_matrix_sync(
            fragment_b, key_column_major + offset + column * inner, inner);
        rocwmma::mma_sync(accumulator, fragment_a, fragment_b, accumulator);
    }
    rocwmma::store_matrix_sync(output + row * columns + column, accumulator,
                               columns, rocwmma::mem_row_major);
}

__global__ void scalar_qk_kernel(std::uint32_t rows, std::uint32_t columns,
                                 std::uint32_t inner, const BFloat16* query,
                                 const BFloat16* key_column_major,
                                 float* output) {
    const auto column = blockIdx.x * blockDim.x + threadIdx.x;
    const auto row = blockIdx.y * blockDim.y + threadIdx.y;
    if (row >= rows || column >= columns) return;
    float sum = 0.0F;
    for (std::uint32_t offset = 0; offset < inner; ++offset) {
        sum += static_cast<float>(query[row * inner + offset]) *
               static_cast<float>(key_column_major[offset + column * inner]);
    }
    output[row * columns + column] = sum;
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
    times.reserve(static_cast<std::size_t>(options.repetitions));
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
            throw std::runtime_error("rocWMMA benchmark needs a HIP device");
        }
        hipDeviceProp_t properties{};
        check_hip(hipGetDeviceProperties(&properties, 0), "hipGetDeviceProperties");
        if (properties.warpSize != static_cast<int>(kWaveSize)) {
            throw std::runtime_error("this rocWMMA QK contract requires wave64");
        }

        const auto query_elements = static_cast<std::size_t>(options.rows * options.inner);
        const auto key_elements = static_cast<std::size_t>(options.columns * options.inner);
        const auto output_elements = static_cast<std::size_t>(options.rows * options.columns);
        std::vector<BFloat16> query(query_elements);
        std::vector<BFloat16> key(key_elements);
        std::vector<BFloat16> right(key_elements);
        for (std::size_t index = 0; index < query.size(); ++index) {
            query[index] = BFloat16(
                static_cast<float>(static_cast<int>(index % 29) - 14) / 32.0F);
        }
        for (std::size_t index = 0; index < key.size(); ++index) {
            key[index] = BFloat16(
                static_cast<float>(static_cast<int>(index % 31) - 15) / 31.0F);
        }
        for (std::int64_t inner = 0; inner < options.inner; ++inner) {
            for (std::int64_t column = 0; column < options.columns; ++column) {
                right[static_cast<std::size_t>(inner * options.columns + column)] =
                    key[static_cast<std::size_t>(inner + column * options.inner)];
            }
        }
        std::vector<float> reference(output_elements, 0.0F);
        for (std::int64_t row = 0; row < options.rows; ++row) {
            for (std::int64_t column = 0; column < options.columns; ++column) {
                float sum = 0.0F;
                for (std::int64_t inner = 0; inner < options.inner; ++inner) {
                    const auto q_index = static_cast<std::size_t>(
                        row * options.inner + inner);
                    const auto k_index = static_cast<std::size_t>(
                        inner + column * options.inner);
                    sum += static_cast<float>(query[q_index]) *
                           static_cast<float>(key[k_index]);
                }
                reference[static_cast<std::size_t>(row * options.columns + column)] = sum;
            }
        }

        DeviceBuffer query_device(query_elements * sizeof(BFloat16));
        DeviceBuffer key_device(key_elements * sizeof(BFloat16));
        DeviceBuffer right_device(key_elements * sizeof(BFloat16));
        DeviceBuffer rocwmma_output_device(output_elements * sizeof(float));
        DeviceBuffer scalar_output_device(output_elements * sizeof(float));
        DeviceBuffer blas_output_device(output_elements * sizeof(float));
        check_hip(hipMemcpy(query_device.as<BFloat16>(), query.data(),
                            query_elements * sizeof(BFloat16), hipMemcpyHostToDevice),
                  "hipMemcpy(query)");
        check_hip(hipMemcpy(key_device.as<BFloat16>(), key.data(),
                            key_elements * sizeof(BFloat16), hipMemcpyHostToDevice),
                  "hipMemcpy(key)");
        check_hip(hipMemcpy(right_device.as<BFloat16>(), right.data(),
                            key_elements * sizeof(BFloat16), hipMemcpyHostToDevice),
                  "hipMemcpy(right)");

        const auto tile = static_cast<std::uint32_t>(options.tile);
        const auto waves = static_cast<std::uint32_t>(options.waves_per_block);
        const dim3 rocwmma_block(kWaveSize * waves);
        const dim3 rocwmma_grid(
            static_cast<unsigned>((options.rows + options.tile * options.waves_per_block - 1) /
                                  (options.tile * options.waves_per_block)),
            static_cast<unsigned>((options.columns + options.tile - 1) / options.tile));
        const auto launch_rocwmma = [&] {
            if (tile == 16) {
                rocwmma_qk_kernel<16><<<rocwmma_grid, rocwmma_block>>>(
                    static_cast<std::uint32_t>(options.rows),
                    static_cast<std::uint32_t>(options.columns),
                    static_cast<std::uint32_t>(options.inner),
                    query_device.as<BFloat16>(), key_device.as<BFloat16>(),
                    rocwmma_output_device.as<float>());
            } else {
                rocwmma_qk_kernel<32><<<rocwmma_grid, rocwmma_block>>>(
                    static_cast<std::uint32_t>(options.rows),
                    static_cast<std::uint32_t>(options.columns),
                    static_cast<std::uint32_t>(options.inner),
                    query_device.as<BFloat16>(), key_device.as<BFloat16>(),
                    rocwmma_output_device.as<float>());
            }
            check_hip(hipGetLastError(), "rocwmma_qk_kernel");
        };
        const dim3 scalar_block(16, 16);
        const dim3 scalar_grid(
            static_cast<unsigned>((options.columns + scalar_block.x - 1) /
                                  scalar_block.x),
            static_cast<unsigned>((options.rows + scalar_block.y - 1) /
                                  scalar_block.y));
        const auto launch_scalar = [&] {
            scalar_qk_kernel<<<scalar_grid, scalar_block>>>(
                static_cast<std::uint32_t>(options.rows),
                static_cast<std::uint32_t>(options.columns),
                static_cast<std::uint32_t>(options.inner),
                query_device.as<BFloat16>(), key_device.as<BFloat16>(),
                scalar_output_device.as<float>());
            check_hip(hipGetLastError(), "scalar_qk_kernel");
        };
        BlasHandle blas_handle;
        BlasDescription blas_description;
        BlasLayout blas_right(
            HIP_R_16BF, static_cast<std::uint64_t>(options.columns),
            static_cast<std::uint64_t>(options.inner), options.columns);
        BlasLayout blas_query(
            HIP_R_16BF, static_cast<std::uint64_t>(options.inner),
            static_cast<std::uint64_t>(options.rows), options.inner);
        BlasLayout blas_output(
            HIP_R_32F, static_cast<std::uint64_t>(options.columns),
            static_cast<std::uint64_t>(options.rows), options.columns);
        const float alpha = 1.0F;
        const float beta = 0.0F;
        const auto launch_blas = [&] {
            check_blas(hipblasLtMatmul(
                           blas_handle.get(), blas_description.get(), &alpha,
                           right_device.as<BFloat16>(), blas_right.get(),
                           query_device.as<BFloat16>(), blas_query.get(), &beta,
                           blas_output_device.as<float>(), blas_output.get(),
                           blas_output_device.as<float>(), blas_output.get(),
                           nullptr, nullptr, 0, nullptr),
                       "hipblasLtMatmul(BF16 QK)");
        };

        launch_rocwmma();
        launch_scalar();
        launch_blas();
        check_hip(hipDeviceSynchronize(), "hipDeviceSynchronize(correctness)");
        std::vector<float> rocwmma_output(output_elements);
        std::vector<float> scalar_output(output_elements);
        std::vector<float> blas_output_values(output_elements);
        check_hip(hipMemcpy(rocwmma_output.data(), rocwmma_output_device.as<float>(),
                            output_elements * sizeof(float), hipMemcpyDeviceToHost),
                  "hipMemcpy(rocwmma output)");
        check_hip(hipMemcpy(scalar_output.data(), scalar_output_device.as<float>(),
                            output_elements * sizeof(float), hipMemcpyDeviceToHost),
                  "hipMemcpy(scalar output)");
        check_hip(hipMemcpy(blas_output_values.data(), blas_output_device.as<float>(),
                            output_elements * sizeof(float), hipMemcpyDeviceToHost),
                  "hipMemcpy(hipBLASLt output)");
        const auto rocwmma_error = compare(rocwmma_output, reference);
        const auto scalar_error = compare(scalar_output, reference);
        const auto blas_error = compare(blas_output_values, reference);
        const bool accuracy_passed =
            rocwmma_error.finite && scalar_error.finite && blas_error.finite &&
            rocwmma_error.maximum <= 2.0e-3F && rocwmma_error.rms <= 3.0e-4 &&
            scalar_error.maximum <= 1.0e-5F && scalar_error.rms <= 1.0e-6 &&
            blas_error.maximum <= 2.0e-3F && blas_error.rms <= 3.0e-4;
        if (!accuracy_passed) {
            std::cerr << "rocWMMA error max=" << rocwmma_error.maximum
                      << " rms=" << rocwmma_error.rms
                      << "; scalar error max=" << scalar_error.maximum
                      << " rms=" << scalar_error.rms
                      << "; hipBLASLt error max=" << blas_error.maximum
                      << " rms=" << blas_error.rms << '\n';
            throw std::runtime_error("rocWMMA QK complete-output gate failed");
        }

        const auto rocwmma_times = time_kernel(options, launch_rocwmma);
        const auto scalar_times = time_kernel(options, launch_scalar);
        const auto blas_times = time_kernel(options, launch_blas);
        const auto rocwmma_p50 = percentile(rocwmma_times, 0.50);
        const auto scalar_p50 = percentile(scalar_times, 0.50);
        const auto blas_p50 = percentile(blas_times, 0.50);
        const auto operations = 2.0 * static_cast<double>(options.rows) *
                                static_cast<double>(options.columns) *
                                static_cast<double>(options.inner);
        const auto rocwmma_tflops = operations / (rocwmma_p50 / 1000.0) / 1.0e12;
        const auto scalar_tflops = operations / (scalar_p50 / 1000.0) / 1.0e12;
        const auto blas_tflops = operations / (blas_p50 / 1000.0) / 1.0e12;
        std::cout << std::setprecision(9)
                  << "{\"schema_version\":1,\"status\":\"pass\""
                  << ",\"op\":\"bf16_qk_tile\""
                  << ",\"architecture\":\"" << properties.gcnArchName << "\""
                  << ",\"rocwmma_version\":\"" << MICROLLM_ROCWMMA_VERSION << "\""
                  << ",\"rows\":" << options.rows
                  << ",\"columns\":" << options.columns
                  << ",\"inner\":" << options.inner
                  << ",\"tile\":[" << options.tile << ',' << options.tile
                  << ',' << kTileK << ']'
                  << ",\"waves_per_block\":" << options.waves_per_block
                  << ",\"wave_size\":" << properties.warpSize
                  << ",\"warmup\":" << options.warmup
                  << ",\"repetitions\":" << options.repetitions
                  << ",\"complete_output_elements\":" << output_elements
                  << ",\"rocwmma_max_error\":" << rocwmma_error.maximum
                  << ",\"rocwmma_rms_error\":" << rocwmma_error.rms
                  << ",\"scalar_max_error\":" << scalar_error.maximum
                  << ",\"scalar_rms_error\":" << scalar_error.rms
                  << ",\"hipblaslt_max_error\":" << blas_error.maximum
                  << ",\"hipblaslt_rms_error\":" << blas_error.rms
                  << ",\"accuracy_passed\":" << (accuracy_passed ? "true" : "false")
                  << ",\"rocwmma_event_ms_p50\":" << rocwmma_p50
                  << ",\"rocwmma_event_ms_p95\":" << percentile(rocwmma_times, 0.95)
                  << ",\"scalar_event_ms_p50\":" << scalar_p50
                  << ",\"scalar_event_ms_p95\":" << percentile(scalar_times, 0.95)
                  << ",\"hipblaslt_event_ms_p50\":" << blas_p50
                  << ",\"hipblaslt_event_ms_p95\":" << percentile(blas_times, 0.95)
                  << ",\"rocwmma_tflops\":" << rocwmma_tflops
                  << ",\"scalar_tflops\":" << scalar_tflops
                  << ",\"hipblaslt_tflops\":" << blas_tflops
                  << ",\"rocwmma_over_scalar\":" << scalar_p50 / rocwmma_p50
                  << ",\"rocwmma_over_hipblaslt\":" << blas_p50 / rocwmma_p50
                  << "}\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "benchmark_rocwmma_qk: " << error.what() << '\n';
        return 2;
    }
}
