#include <iostream>
#include <stdexcept>

#include <microllm/base/device.h>
#include <microllm/model/config.h>
#include <microllm/ops/tuning.h>

int main() {
    const auto device = microllm::Device::cpu();
    const auto config = microllm::model::ModelConfig::model_s();
    bool rejected_cpu_tuning = false;
    try {
        const microllm::Tensor left({2, 2});
        const microllm::Tensor right({2, 2});
        (void)microllm::ops::autotune_matmul(left, right);
    } catch (const std::invalid_argument&) {
        rejected_cpu_tuning = true;
    }
    if (!device.is_cpu() || config.parameter_count() == 0 ||
        !rejected_cpu_tuning) return 1;
    std::cout << "microLLM package consumer: pass\n";
    return 0;
}
