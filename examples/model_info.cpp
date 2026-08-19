#include <iostream>

#include <microllm/model/config.h>

int main() {
    for (const auto& [name, config] : {
             std::pair{"Model-S", microllm::model::ModelConfig::model_s()},
             std::pair{"Model-M", microllm::model::ModelConfig::model_m()}}) {
        std::cout << name << '\n';
        std::cout << "  " << config.summary() << '\n';
        std::cout << "  fp32_weight_bytes=" << config.weight_bytes(4) << '\n';
        std::cout << "  fp16_weight_bytes=" << config.weight_bytes(2) << '\n';
    }
    return 0;
}
