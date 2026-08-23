#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <functional>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

#include <microllm/ops/ops.h>
#include <microllm/runtime/runtime.h>

namespace {

struct Options {
    std::int64_t size = 512;
    int warmup = 3;
    int repetitions = 10;
    std::string reference = "cpu";
};

Options options(int argc, char** argv) {
    Options result;
    for (int index = 1; index < argc; index += 2) {
        if (index + 1 >= argc) throw std::invalid_argument("missing benchmark value");
        const std::string name = argv[index];
        if (name == "--reference") {
            result.reference = argv[index + 1];
            continue;
        }
        const auto value = std::strtoll(argv[index + 1], nullptr, 10);
        if (name == "--size") result.size = value;
        else if (name == "--warmup") result.warmup = static_cast<int>(value);
        else if (name == "--repetitions") result.repetitions = static_cast<int>(value);
        else throw std::invalid_argument("unknown benchmark option: " + name);
    }
    if (result.size <= 0 || result.size > 4096 || result.warmup < 0 ||
        result.repetitions <= 0 ||
        (result.reference != "cpu" && result.reference != "fp32")) {
        throw std::invalid_argument("invalid benchmark range or reference");
    }
    return result;
}

double percentile(std::vector<double> values, double fraction) {
    std::sort(values.begin(), values.end());
    const auto position = fraction * static_cast<double>(values.size() - 1);
    const auto lower = static_cast<std::size_t>(std::floor(position));
    const auto upper = static_cast<std::size_t>(std::ceil(position));
    const auto weight = position - static_cast<double>(lower);
    return values[lower] * (1.0 - weight) + values[upper] * weight;
}

struct Result { std::string dtype; double median; double p95; float error; };

Result measure(const std::string& dtype, int warmup, int repetitions,
               microllm::runtime::Stream& stream,
               const std::vector<float>& reference,
               const std::function<microllm::Tensor()>& operation) {
    for (int index = 0; index < warmup; ++index) (void)operation();
    stream.synchronize();
    std::vector<double> times;
    microllm::Tensor output;
    for (int index = 0; index < repetitions; ++index) {
        microllm::runtime::Event start(stream.device());
        microllm::runtime::Event finish(stream.device());
        start.record(stream);
        output = operation();
        finish.record(stream);
        finish.synchronize();
        times.push_back(finish.elapsed_ms_since(start));
    }
    const auto actual = output.to_vector();
    float error = 0.0F;
    for (std::size_t index = 0; index < actual.size(); ++index) {
        error = std::max(error, std::abs(actual[index] - reference[index]));
    }
    return {dtype, percentile(times, 0.5), percentile(times, 0.95), error};
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const auto config = options(argc, argv);
        if (microllm::runtime::hip_device_count() == 0 ||
            !microllm::ops::hipblaslt_available()) {
            throw std::runtime_error("precision benchmark requires HIP and hipBLASLt");
        }
        const auto gpu = microllm::Device::hip(0);
        const auto count = static_cast<std::size_t>(config.size * config.size);
        std::vector<float> left_values(count), right_values(count);
        for (std::size_t index = 0; index < count; ++index) {
            left_values[index] = static_cast<float>(static_cast<int>(index % 31) - 15) / 31.0F;
            right_values[index] = static_cast<float>(static_cast<int>(index % 19) - 9) / 19.0F;
        }
        const microllm::Shape shape{config.size, config.size};
        const auto left_cpu = microllm::Tensor::from_vector(left_values, shape);
        const auto right_cpu = microllm::Tensor::from_vector(right_values, shape);
        const auto left32 = left_cpu.to(gpu);
        const auto right32 = right_cpu.to(gpu);
        const auto left16 = microllm::Tensor::from_vector(left_values, shape, microllm::DType::Float16).to(gpu);
        const auto right16 = microllm::Tensor::from_vector(right_values, shape, microllm::DType::Float16).to(gpu);
        const auto leftbf = microllm::Tensor::from_vector(left_values, shape, microllm::DType::BFloat16).to(gpu);
        const auto rightbf = microllm::Tensor::from_vector(right_values, shape, microllm::DType::BFloat16).to(gpu);
        microllm::runtime::Stream stream(gpu);
        const microllm::ops::OpContext context{&stream, nullptr, 0};
        const auto reference = config.reference == "cpu"
                                   ? microllm::ops::matmul(
                                         left_cpu, right_cpu).to_vector()
                                   : microllm::ops::matmul_with_implementation(
                                         left32, right32,
                                         microllm::ops::MatmulImplementation::HipBLASLt,
                                         context).to_vector();
        const auto left8 = microllm::ops::quantize_fp8(left32, microllm::DType::Float8E4M3FNUZ, 1.0F / 240.0F, context);
        const auto right8 = microllm::ops::quantize_fp8(right32, microllm::DType::Float8E4M3FNUZ, 1.0F / 240.0F, context);
        stream.synchronize();
        std::vector<Result> results;
        results.push_back(measure("fp32_readable", config.warmup, config.repetitions, stream, reference,
            [&] { return microllm::ops::matmul_with_implementation(left32, right32, microllm::ops::MatmulImplementation::Readable, context); }));
        results.push_back(measure("fp32", config.warmup, config.repetitions, stream, reference,
            [&] { return microllm::ops::matmul_with_implementation(left32, right32, microllm::ops::MatmulImplementation::HipBLASLt, context); }));
        results.push_back(measure("fp16", config.warmup, config.repetitions, stream, reference,
            [&] { return microllm::ops::matmul_with_implementation(left16, right16, microllm::ops::MatmulImplementation::HipBLASLt, context); }));
        results.push_back(measure("bf16", config.warmup, config.repetitions, stream, reference,
            [&] { return microllm::ops::matmul_with_implementation(leftbf, rightbf, microllm::ops::MatmulImplementation::HipBLASLt, context); }));
        results.push_back(measure("fp8_e4m3_fnuz", config.warmup, config.repetitions, stream, reference,
            [&] { return microllm::ops::fp8_matmul(left8, right8, microllm::DType::BFloat16, context); }));
        const auto baseline = results.front().median;
        const auto fp32_hipblas = results[1].median;
        bool accuracy_passed = true;
        for (const auto& result : results) {
            const auto scale = std::max(
                1.0F, static_cast<float>(config.size) / 512.0F);
            const auto tolerance = result.dtype == "fp32_readable" || result.dtype == "fp32"
                                       ? 2.0e-4F
                                       : result.dtype == "fp16" ? 2.0e-3F * scale
                                       : result.dtype == "bf16" ? 1.0e-2F * scale
                                       : 1.0e-1F * scale;
            accuracy_passed = accuracy_passed && result.error <= tolerance;
            std::cout << std::setprecision(9)
                      << "{\"schema_version\":1,\"op\":\"matmul\",\"shape\":["
                      << config.size << ',' << config.size << ',' << config.size
                      << "],\"reference\":\"" << config.reference
                      << "\",\"dtype\":\"" << result.dtype << "\",\"median_ms\":"
                      << result.median << ",\"p95_ms\":" << result.p95
                      << ",\"max_abs_error\":" << result.error
                      << ",\"speedup_vs_readable_fp32\":" << baseline / result.median
                      << ",\"speedup_vs_hipblaslt_fp32\":" << fp32_hipblas / result.median
                      << ",\"accuracy_passed\":" << (result.error <= tolerance ? "true" : "false")
                      << "}\n";
        }
        return accuracy_passed ? 0 : 2;
    } catch (const std::exception& error) {
        std::cerr << "benchmark_precision: " << error.what() << '\n';
        return 1;
    }
}
