#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <functional>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include <hip/hip_runtime_api.h>

#include <microllm/ops/ops.h>
#include <microllm/runtime/memory.h>
#include <microllm/runtime/runtime.h>

namespace {

struct Options {
    std::string path = "island";
    std::int64_t tokens = 1;
    std::int64_t hidden = 896;
    std::int64_t intermediate = 4864;
    int warmup = 5;
    int repetitions = 20;
};

std::int64_t integer(const char* value, const char* name) {
    char* end = nullptr;
    const auto parsed = std::strtoll(value, &end, 10);
    if (end == value || *end != '\0') {
        throw std::invalid_argument(std::string("invalid ") + name);
    }
    return parsed;
}

Options parse_options(int argc, char** argv) {
    Options result;
    for (int index = 1; index < argc; index += 2) {
        if (index + 1 >= argc) throw std::invalid_argument("benchmark option lacks a value");
        const std::string_view name(argv[index]);
        if (name == "--path") result.path = argv[index + 1];
        else if (name == "--tokens") result.tokens = integer(argv[index + 1], "tokens");
        else if (name == "--hidden") result.hidden = integer(argv[index + 1], "hidden");
        else if (name == "--intermediate") {
            result.intermediate = integer(argv[index + 1], "intermediate");
        } else if (name == "--warmup") {
            result.warmup = static_cast<int>(integer(argv[index + 1], "warmup"));
        } else if (name == "--repetitions") {
            result.repetitions = static_cast<int>(integer(argv[index + 1], "repetitions"));
        } else {
            throw std::invalid_argument("unknown benchmark option: " + std::string(name));
        }
    }
    if (result.path != "fp32" && result.path != "per-linear" && result.path != "island") {
        throw std::invalid_argument("path must be fp32, per-linear, or island");
    }
    if (result.tokens <= 0 || result.tokens > 4096 || result.hidden <= 0 ||
        result.hidden > 16384 || result.intermediate <= 0 || result.intermediate > 65536 ||
        result.warmup < 0 || result.repetitions <= 0 || result.repetitions > 10000) {
        throw std::invalid_argument("benchmark dimensions or repetitions are outside limits");
    }
    return result;
}

void check_hip(hipError_t status, const char* operation) {
    if (status != hipSuccess) {
        throw std::runtime_error(std::string(operation) + ": " + hipGetErrorString(status));
    }
}

class Event {
public:
    Event() { check_hip(hipEventCreate(&event_), "hipEventCreate"); }
    ~Event() { if (event_ != nullptr) (void)hipEventDestroy(event_); }
    Event(const Event&) = delete;
    Event& operator=(const Event&) = delete;
    Event(Event&&) = delete;
    Event& operator=(Event&&) = delete;
    hipEvent_t get() const noexcept { return event_; }

private:
    hipEvent_t event_ = nullptr;
};

double percentile(std::vector<double> values, double fraction) {
    std::sort(values.begin(), values.end());
    const auto position = fraction * static_cast<double>(values.size() - 1);
    const auto low = static_cast<std::size_t>(std::floor(position));
    const auto high = static_cast<std::size_t>(std::ceil(position));
    const auto weight = position - static_cast<double>(low);
    return values[low] * (1.0 - weight) + values[high] * weight;
}

std::vector<float> values(std::size_t count, int period, float scale, int phase = 0) {
    std::vector<float> result(count);
    for (std::size_t index = 0; index < count; ++index) {
        result[index] = static_cast<float>(
            (static_cast<int>(index) + phase) % period - period / 2) * scale;
    }
    return result;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        using microllm::DType;
        using microllm::Shape;
        using microllm::Tensor;
        using microllm::ops::MatmulImplementation;
        const auto options = parse_options(argc, argv);
        if (microllm::runtime::hip_device_count() == 0 ||
            !microllm::ops::hipblaslt_available()) {
            throw std::runtime_error("BF16 FFN benchmark requires HIP and hipBLASLt");
        }
        const auto gpu = microllm::Device::hip(0);
        microllm::runtime::enable_hip_caching_allocator(gpu);
        const auto weight_scale = 0.125F / std::sqrt(static_cast<float>(options.hidden));
        const auto input_cpu = Tensor::from_vector(
            values(static_cast<std::size_t>(options.tokens * options.hidden), 31, 1.0F / 31.0F),
            {options.tokens, options.hidden});
        const auto gate_cpu = Tensor::from_vector(
            values(static_cast<std::size_t>(options.hidden * options.intermediate),
                   37, weight_scale), {options.hidden, options.intermediate});
        const auto up_cpu = Tensor::from_vector(
            values(static_cast<std::size_t>(options.hidden * options.intermediate),
                   41, weight_scale, 7), {options.hidden, options.intermediate});
        const auto down_cpu = Tensor::from_vector(
            values(static_cast<std::size_t>(options.intermediate * options.hidden),
                   43, weight_scale, 13), {options.intermediate, options.hidden});

        const auto input = input_cpu.to(gpu);
        const auto gate = gate_cpu.to(gpu);
        const auto up = up_cpu.to(gpu);
        const auto down = down_cpu.to(gpu);
        const auto gate_bf16 = gate_cpu.cast(DType::BFloat16).to(gpu);
        const auto up_bf16 = up_cpu.cast(DType::BFloat16).to(gpu);
        const auto down_bf16 = down_cpu.cast(DType::BFloat16).to(gpu);

        const auto fp32_operation = [&] {
            const auto gate_output = microllm::ops::matmul_with_implementation(
                input, gate, MatmulImplementation::HipBLASLt);
            const auto up_output = microllm::ops::matmul_with_implementation(
                input, up, MatmulImplementation::HipBLASLt);
            return microllm::ops::matmul_with_implementation(
                microllm::ops::swiglu(gate_output, up_output), down,
                MatmulImplementation::HipBLASLt);
        };
        const auto per_linear_operation = [&] {
            const auto gate_output = microllm::ops::bf16_matmul(input, gate_bf16);
            const auto up_output = microllm::ops::bf16_matmul(input, up_bf16);
            return microllm::ops::bf16_matmul(
                microllm::ops::swiglu(gate_output, up_output), down_bf16);
        };
        const auto island_operation = [&] {
            return microllm::ops::bf16_ffn(input, gate_bf16, up_bf16, down_bf16);
        };
        const auto operation = options.path == "fp32"
                                   ? std::function<Tensor()>(fp32_operation)
                                   : options.path == "per-linear"
                                         ? std::function<Tensor()>(per_linear_operation)
                                         : std::function<Tensor()>(island_operation);

        const auto reference = fp32_operation();
        microllm::runtime::synchronize(gpu);
        const auto reference_values = reference.to_vector();
        for (int iteration = 0; iteration < options.warmup; ++iteration) (void)operation();
        microllm::runtime::synchronize(gpu);
        microllm::runtime::reset_allocation_peak(gpu);
        microllm::runtime::reset_transfer_stats();

        Event start;
        Event finish;
        std::vector<double> device_times;
        std::vector<double> wall_times;
        Tensor output;
        for (int iteration = 0; iteration < options.repetitions; ++iteration) {
            const auto wall_start = std::chrono::steady_clock::now();
            check_hip(hipEventRecord(start.get(), nullptr), "hipEventRecord(start)");
            output = operation();
            check_hip(hipEventRecord(finish.get(), nullptr), "hipEventRecord(finish)");
            check_hip(hipEventSynchronize(finish.get()), "hipEventSynchronize(finish)");
            float elapsed = 0.0F;
            check_hip(hipEventElapsedTime(&elapsed, start.get(), finish.get()),
                      "hipEventElapsedTime");
            device_times.push_back(elapsed);
            wall_times.push_back(std::chrono::duration<double, std::milli>(
                                     std::chrono::steady_clock::now() - wall_start).count());
        }
        const auto allocation = microllm::runtime::allocation_stats(gpu);
        const auto transfers = microllm::runtime::transfer_stats();
        const auto actual_values = output.to_vector();
        double squared_error = 0.0;
        double squared_reference = 0.0;
        float maximum_error = 0.0F;
        for (std::size_t index = 0; index < actual_values.size(); ++index) {
            const auto error = actual_values[index] - reference_values[index];
            maximum_error = std::max(maximum_error, std::abs(error));
            squared_error += static_cast<double>(error) * error;
            squared_reference += static_cast<double>(reference_values[index]) *
                                 reference_values[index];
        }
        const auto relative_l2 = std::sqrt(squared_error / std::max(squared_reference, 1.0e-30));
        const auto device = microllm::runtime::device_info(gpu);
        const auto accuracy_passed = options.path == "fp32" ||
                                     (maximum_error <= 0.25F && relative_l2 <= 0.05);
        const auto explicit_input_cast_kernels = options.path == "per-linear" ? 3 :
                                                 options.path == "island" ? 1 : 0;
        std::cout << std::setprecision(9)
                  << "{\"schema_version\":1,\"benchmark\":\"bf16_ffn\""
                  << ",\"path\":\"" << options.path << "\""
                  << ",\"device_name\":\"" << device.name << "\""
                  << ",\"architecture\":\"" << device.architecture << "\""
                  << ",\"hip_runtime_version\":" << microllm::runtime::hip_runtime_version()
                  << ",\"tokens\":" << options.tokens
                  << ",\"hidden\":" << options.hidden
                  << ",\"intermediate\":" << options.intermediate
                  << ",\"warmup\":" << options.warmup
                  << ",\"repetitions\":" << options.repetitions
                  << ",\"device_ms_median\":" << percentile(device_times, 0.5)
                  << ",\"device_ms_p95\":" << percentile(device_times, 0.95)
                  << ",\"wall_ms_median\":" << percentile(wall_times, 0.5)
                  << ",\"explicit_input_cast_kernels\":" << explicit_input_cast_kernels
                  << ",\"max_abs_error_vs_fp32\":" << maximum_error
                  << ",\"relative_l2_error_vs_fp32\":" << relative_l2
                  << ",\"accuracy_passed\":" << (accuracy_passed ? "true" : "false")
                  << ",\"host_to_device_calls_measured\":" << transfers.host_to_device_calls
                  << ",\"device_to_host_calls_measured\":" << transfers.device_to_host_calls
                  << ",\"allocation_calls\":" << allocation.allocation_calls
                  << ",\"backend_allocation_calls\":" << allocation.backend_allocation_calls
                  << ",\"cache_reuse_calls\":" << allocation.cache_reuse_calls
                  << ",\"peak_active_bytes\":" << allocation.peak_bytes
                  << ",\"cached_bytes\":" << allocation.cached_bytes
                  << ",\"reserved_bytes\":" << allocation.reserved_bytes << "}\n";
        return accuracy_passed && transfers.host_to_device_calls == 0 &&
                       transfers.device_to_host_calls == 0 ? 0 : 2;
    } catch (const std::exception& error) {
        std::cerr << "benchmark_bf16_ffn: " << error.what() << '\n';
        return 1;
    }
}
