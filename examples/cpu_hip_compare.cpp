#include <algorithm>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <vector>

#include <microllm/ops/ops.h>
#include <microllm/runtime/runtime.h>

int main() {
    using microllm::Device;
    using microllm::Tensor;
    try {
        if (!microllm::runtime::hip_compiled() || microllm::runtime::hip_device_count() == 0) {
            std::cout << "SKIP: no compiled and visible HIP device\n";
            return 0;
        }
        const auto gpu = Device::hip();
        const auto info = microllm::runtime::device_info(gpu);
        std::vector<float> left_values(4096);
        std::vector<float> right_values(4096);
        for (std::size_t index = 0; index < left_values.size(); ++index) {
            left_values[index] = static_cast<float>(index % 31) / 31.0F;
            right_values[index] = static_cast<float>(index % 17) / 17.0F;
        }
        const auto left_cpu = Tensor::from_vector(left_values, {64, 64});
        const auto right_cpu = Tensor::from_vector(right_values, {64, 64});
        const auto reference = microllm::ops::add(left_cpu, right_cpu).to_vector();

        const auto left_gpu = left_cpu.to(gpu);
        const auto right_gpu = right_cpu.to(gpu);
        microllm::runtime::Stream stream(gpu);
        microllm::runtime::Event start(gpu);
        microllm::runtime::Event finish(gpu);
        const microllm::ops::OpContext context{&stream, nullptr, 0};
        start.record(stream);
        const auto output_gpu = microllm::ops::add(left_gpu, right_gpu, context);
        finish.record(stream);
        finish.synchronize();
        const auto actual = output_gpu.to_vector();

        float maximum_error = 0.0F;
        for (std::size_t index = 0; index < actual.size(); ++index) {
            maximum_error = std::max(maximum_error, std::abs(actual[index] - reference[index]));
        }
        std::cout << "gpu=" << info.name << '\n';
        std::cout << "arch=" << info.architecture << '\n';
        std::cout << "elements=" << actual.size() << '\n';
        std::cout << "maximum_absolute_error=" << maximum_error << '\n';
        std::cout << "kernel_elapsed_ms=" << finish.elapsed_ms_since(start) << '\n';
        if (maximum_error != 0.0F) throw std::runtime_error("CPU/HIP add mismatch");
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "microllm_cpu_hip_compare: " << error.what() << '\n';
        return 1;
    }
}
