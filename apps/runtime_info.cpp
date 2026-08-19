#include <iostream>

#include <microllm/core/tensor.h>

#ifndef MICROLLM_VERSION
#define MICROLLM_VERSION "unknown"
#endif

int main() {
    const microllm::Tensor tensor({2, 3});
    std::cout << "microLLM-rocm " << MICROLLM_VERSION << '\n';
    std::cout << "core:         ready (CPU float32)\n";
    std::cout << "HIP build:    " << (MICROLLM_HAS_HIP ? "available" : "disabled") << '\n';
    std::cout << "hipBLASLt:    " << (MICROLLM_HAS_HIPBLASLT ? "available" : "disabled") << '\n';
    std::cout << "RCCL:         " << (MICROLLM_HAS_RCCL ? "available" : "disabled") << '\n';
    std::cout << "smoke tensor: " << tensor << '\n';
    return 0;
}
