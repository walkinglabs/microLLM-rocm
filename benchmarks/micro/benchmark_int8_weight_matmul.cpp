#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <string_view>
#include <vector>

#include <microllm/ops/ops.h>
#include <microllm/runtime/runtime.h>

namespace {

struct Timing { double event_ms; double wall_ms; };

double median(std::vector<double> values) {
    std::sort(values.begin(), values.end());
    return values[values.size() / 2];
}

Timing measure(const microllm::Tensor& input,
               const microllm::ops::Int8ScaledTensor& weight,
               microllm::ops::Int8WeightMatmulImplementation implementation) {
    const auto gpu = input.device();
    microllm::runtime::Stream stream(gpu);
    microllm::ops::OpContext context;
    context.stream = &stream;
    for (int iteration = 0; iteration < 2; ++iteration) {
        (void)microllm::ops::int8_weight_matmul_with_implementation(
            input, weight, implementation, context);
        stream.synchronize();
    }
    std::vector<double> event;
    std::vector<double> wall;
    for (int iteration = 0; iteration < 5; ++iteration) {
        microllm::runtime::Event start(gpu, true);
        microllm::runtime::Event finish(gpu, true);
        const auto wall_start = std::chrono::steady_clock::now();
        start.record(stream);
        auto output = microllm::ops::int8_weight_matmul_with_implementation(
            input, weight, implementation, context);
        finish.record(stream);
        finish.synchronize();
        const auto wall_finish = std::chrono::steady_clock::now();
        event.push_back(finish.elapsed_ms_since(start));
        wall.push_back(std::chrono::duration<double, std::milli>(
                           wall_finish - wall_start).count());
        if (output.numel() == 0) throw std::runtime_error("empty benchmark output");
    }
    return {median(event), median(wall)};
}

void run(std::string_view model, std::int64_t inner, std::int64_t columns) {
    const auto gpu = microllm::Device::hip(0);
    std::vector<float> input_values(static_cast<std::size_t>(inner));
    std::vector<std::int8_t> weight_values(
        static_cast<std::size_t>(inner * columns));
    for (std::size_t index = 0; index < input_values.size(); ++index) {
        input_values[index] =
            static_cast<float>(static_cast<int>(index % 29) - 14) / 64.0F;
    }
    for (std::size_t index = 0; index < weight_values.size(); ++index) {
        weight_values[index] = static_cast<std::int8_t>(
            static_cast<int>(index % 31) - 15);
    }
    const auto input = microllm::Tensor::from_vector(
        input_values, {1, inner}).to(gpu);
    const microllm::ops::Int8ScaledTensor weight{
        microllm::Tensor::from_int8_vector(
            weight_values, {inner, columns}).to(gpu),
        microllm::Tensor::from_vector({0.03125F}, {}).to(gpu),
        0.03125F, true};
    const auto control = microllm::ops::int8_weight_matmul_with_implementation(
        input, weight,
        microllm::ops::Int8WeightMatmulImplementation::ExplicitDequantize);
    const auto candidate = microllm::ops::int8_weight_matmul_with_implementation(
        input, weight,
        microllm::ops::Int8WeightMatmulImplementation::FusedDecode);
    const auto left = control.to_vector();
    const auto right = candidate.to_vector();
    double squared = 0.0;
    float maximum = 0.0F;
    for (std::size_t index = 0; index < left.size(); ++index) {
        const auto difference = std::abs(left[index] - right[index]);
        maximum = std::max(maximum, difference);
        squared += static_cast<double>(difference) * difference;
    }
    const auto rms = std::sqrt(squared / static_cast<double>(left.size()));
    const auto explicit_timing = measure(
        input, weight,
        microllm::ops::Int8WeightMatmulImplementation::ExplicitDequantize);
    const auto fused_timing = measure(
        input, weight,
        microllm::ops::Int8WeightMatmulImplementation::FusedDecode);
    std::cout << "{\"model\":\"" << model << "\",\"m\":1,\"k\":"
              << inner << ",\"n\":" << columns
              << ",\"warmup\":2,\"repetitions\":5"
              << ",\"explicit_event_ms\":" << explicit_timing.event_ms
              << ",\"fused_event_ms\":" << fused_timing.event_ms
              << ",\"event_speedup\":"
              << explicit_timing.event_ms / fused_timing.event_ms
              << ",\"explicit_wall_ms\":" << explicit_timing.wall_ms
              << ",\"fused_wall_ms\":" << fused_timing.wall_ms
              << ",\"wall_speedup\":"
              << explicit_timing.wall_ms / fused_timing.wall_ms
              << ",\"maximum_absolute_error\":" << maximum
              << ",\"rms_error\":" << rms
              << ",\"int8_weight_bytes\":" << inner * columns
              << ",\"avoided_float_weight_bytes\":"
              << inner * columns * 4 << "}\n";
}

}  // namespace

int main() {
    if (microllm::runtime::hip_device_count() == 0) return 77;
    run("qwen2.5-0.5b", 896, 4864);
    run("deepseek-distill-1.5b", 1536, 8960);
}
