#include <iostream>

#include <microllm/base/device.h>
#include <microllm/model/config.h>

int main() {
    const auto device = microllm::Device::cpu();
    const auto config = microllm::model::ModelConfig::model_s();

    std::cout << "microLLM package example: " << device.str()
              << ", parameters=" << config.parameter_count() << '\n';
    return device.is_cpu() && config.parameter_count() > 0 ? 0 : 1;
}
