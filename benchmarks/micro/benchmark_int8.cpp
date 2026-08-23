#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include <hip/hip_runtime.h>
#include <hipblaslt/hipblaslt.h>

namespace {

void check_hip(hipError_t status, const char* operation) {
    if (status != hipSuccess) {
        throw std::runtime_error(
            std::string(operation) + ": " + hipGetErrorString(status));
    }
}

void check_blas(hipblasStatus_t status, const char* operation) {
    if (status != HIPBLAS_STATUS_SUCCESS) {
        throw std::runtime_error(
            std::string(operation) + " failed with status " +
            std::to_string(static_cast<int>(status)));
    }
}

struct Options {
    std::int64_t size = 512;
    int warmup = 5;
    int repetitions = 20;
};

Options options(int argc, char** argv) {
    Options result;
    for (int index = 1; index < argc; index += 2) {
        if (index + 1 >= argc) throw std::invalid_argument("missing INT8 benchmark value");
        const std::string name = argv[index];
        const auto value = std::strtoll(argv[index + 1], nullptr, 10);
        if (name == "--size") result.size = value;
        else if (name == "--warmup") result.warmup = static_cast<int>(value);
        else if (name == "--repetitions") result.repetitions = static_cast<int>(value);
        else throw std::invalid_argument("unknown INT8 benchmark option: " + name);
    }
    if (result.size <= 0 || result.size > 4096 || result.warmup < 0 ||
        result.repetitions <= 0) {
        throw std::invalid_argument("invalid INT8 benchmark range");
    }
    return result;
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

class DeviceBuffer {
public:
    explicit DeviceBuffer(std::size_t bytes) {
        check_hip(hipMalloc(&data_, bytes), "hipMalloc");
    }
    ~DeviceBuffer() { if (data_ != nullptr) (void)hipFree(data_); }
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
    void* data() noexcept { return data_; }
    const void* data() const noexcept { return data_; }
private:
    void* data_ = nullptr;
};

class Layout {
public:
    Layout(hipDataType type, std::uint64_t rows, std::uint64_t columns,
           std::int64_t leading_dimension) {
        check_blas(hipblasLtMatrixLayoutCreate(
                       &value_, type, rows, columns, leading_dimension),
                   "hipblasLtMatrixLayoutCreate");
    }
    ~Layout() { if (value_ != nullptr) (void)hipblasLtMatrixLayoutDestroy(value_); }
    hipblasLtMatrixLayout_t get() const noexcept { return value_; }
private:
    hipblasLtMatrixLayout_t value_ = nullptr;
};

std::int32_t cpu_dot(const std::vector<std::int8_t>& left,
                     const std::vector<std::int8_t>& right,
                     std::int64_t size, std::int64_t row,
                     std::int64_t column) {
    std::int32_t value = 0;
    for (std::int64_t inner = 0; inner < size; ++inner) {
        // Column-major A(row, inner) and B(inner, column).
        value += static_cast<std::int32_t>(left[
                     static_cast<std::size_t>(row + inner * size)]) *
                 static_cast<std::int32_t>(right[
                     static_cast<std::size_t>(inner + column * size)]);
    }
    return value;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const auto config = options(argc, argv);
        int devices = 0;
        check_hip(hipGetDeviceCount(&devices), "hipGetDeviceCount");
        if (devices <= 0) throw std::runtime_error("INT8 benchmark needs a HIP device");
        const auto elements = static_cast<std::size_t>(config.size * config.size);
        std::vector<std::int8_t> left(elements), right(elements);
        for (std::size_t index = 0; index < elements; ++index) {
            left[index] = static_cast<std::int8_t>(static_cast<int>(index % 7) - 3);
            right[index] = static_cast<std::int8_t>(static_cast<int>(index % 5) - 2);
        }
        DeviceBuffer left_device(elements);
        DeviceBuffer right_device(elements);
        DeviceBuffer output_device(elements * sizeof(std::int32_t));
        check_hip(hipMemcpy(left_device.data(), left.data(), elements,
                            hipMemcpyHostToDevice), "hipMemcpy(left)");
        check_hip(hipMemcpy(right_device.data(), right.data(), elements,
                            hipMemcpyHostToDevice), "hipMemcpy(right)");

        hipblasLtHandle_t handle = nullptr;
        hipblasLtMatmulDesc_t operation = nullptr;
        check_blas(hipblasLtCreate(&handle), "hipblasLtCreate");
        check_blas(hipblasLtMatmulDescCreate(
                       &operation, HIPBLAS_COMPUTE_32I, HIP_R_32I),
                   "hipblasLtMatmulDescCreate(INT8)");
        Layout matrix_a(HIP_R_8I, config.size, config.size, config.size);
        Layout matrix_b(HIP_R_8I, config.size, config.size, config.size);
        Layout matrix_c(HIP_R_32I, config.size, config.size, config.size);
        const std::int32_t alpha = 1;
        const std::int32_t beta = 0;
        const auto launch = [&] {
            check_blas(hipblasLtMatmul(
                           handle, operation, &alpha,
                           left_device.data(), matrix_a.get(),
                           right_device.data(), matrix_b.get(), &beta,
                           output_device.data(), matrix_c.get(),
                           output_device.data(), matrix_c.get(), nullptr,
                           nullptr, 0, nullptr),
                       "hipblasLtMatmul(INT8)");
        };
        for (int index = 0; index < config.warmup; ++index) launch();
        check_hip(hipDeviceSynchronize(), "hipDeviceSynchronize(warmup)");
        std::vector<float> milliseconds;
        for (int index = 0; index < config.repetitions; ++index) {
            hipEvent_t start = nullptr;
            hipEvent_t finish = nullptr;
            check_hip(hipEventCreate(&start), "hipEventCreate(start)");
            check_hip(hipEventCreate(&finish), "hipEventCreate(finish)");
            check_hip(hipEventRecord(start), "hipEventRecord(start)");
            launch();
            check_hip(hipEventRecord(finish), "hipEventRecord(finish)");
            check_hip(hipEventSynchronize(finish), "hipEventSynchronize");
            float elapsed = 0.0F;
            check_hip(hipEventElapsedTime(&elapsed, start, finish),
                      "hipEventElapsedTime");
            milliseconds.push_back(elapsed);
            (void)hipEventDestroy(start);
            (void)hipEventDestroy(finish);
        }
        std::vector<std::int32_t> output(elements);
        check_hip(hipMemcpy(output.data(), output_device.data(),
                            output.size() * sizeof(std::int32_t),
                            hipMemcpyDeviceToHost), "hipMemcpy(output)");
        const std::vector<std::pair<std::int64_t, std::int64_t>> samples{
            {0, 0}, {0, config.size - 1}, {config.size / 2, config.size / 2},
            {config.size - 1, 0}, {config.size - 1, config.size - 1}};
        std::int32_t maximum_error = 0;
        for (const auto [row, column] : samples) {
            const auto expected = cpu_dot(left, right, config.size, row, column);
            const auto actual = output[static_cast<std::size_t>(row + column * config.size)];
            maximum_error = std::max(maximum_error, std::abs(actual - expected));
        }
        const auto median_ms = percentile(milliseconds, 0.50);
        const auto operations = 2.0 * static_cast<double>(config.size) *
                                static_cast<double>(config.size) *
                                static_cast<double>(config.size);
        const auto achieved_tops = operations / (median_ms / 1000.0) / 1.0e12;
        std::cout << std::setprecision(9)
                  << "{\"schema_version\":1,\"op\":\"int8_matmul\""
                  << ",\"shape\":[" << config.size << ',' << config.size << ','
                  << config.size << "],\"input_dtype\":\"int8\""
                  << ",\"output_dtype\":\"int32\",\"warmup\":"
                  << config.warmup << ",\"repetitions\":" << config.repetitions
                  << ",\"median_ms\":" << median_ms
                  << ",\"p95_ms\":" << percentile(milliseconds, 0.95)
                  << ",\"achieved_tops\":" << achieved_tops
                  << ",\"official_peak_tops\":2614.9"
                  << ",\"official_peak_utilization\":"
                  << achieved_tops / 2614.9
                  << ",\"sample_count\":" << samples.size()
                  << ",\"maximum_sample_error\":" << maximum_error
                  << ",\"accuracy_passed\":"
                  << (maximum_error == 0 ? "true" : "false") << "}\n";
        (void)hipblasLtMatmulDescDestroy(operation);
        (void)hipblasLtDestroy(handle);
        return maximum_error == 0 ? 0 : 2;
    } catch (const std::exception& error) {
        std::cerr << "benchmark_int8: " << error.what() << '\n';
        return 1;
    }
}
