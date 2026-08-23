#include <iostream>

#include <microllm/base/device.h>
#include <microllm/model/config.h>

int main() {
    const auto device = microllm::Device::cpu();
    const auto config = microllm::model::ModelConfig::model_s();
    if (!device.is_cpu() || config.parameter_count() == 0) return 1;
    std::cout << "microLLM package consumer: pass\n";
    return 0;
}
