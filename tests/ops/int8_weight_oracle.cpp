#include <cstdint>
#include <iostream>
#include <vector>

#include <microllm/ops/ops.h>

namespace {

template <typename Value>
void print_values(const char* name, const std::vector<Value>& values) {
    std::cout << name << '=';
    for (std::size_t index = 0; index < values.size(); ++index) {
        if (index != 0) std::cout << ',';
        std::cout << +values[index];
    }
    std::cout << '\n';
}

}  // namespace

int main() {
    const auto input = microllm::Tensor::from_vector(
        {-31.75F, -3.0F, -1.625F, -0.375F, 0.0F,
          0.375F, 1.625F, 3.0F, 31.75F, 100.0F},
        {2, 5});
    const auto quantized = microllm::ops::quantize_int8(input, 0.25F);
    const auto restored = microllm::ops::dequantize_int8(quantized);
    std::cout << "shape=2,5\nscale=0.25\n";
    print_values("quantized", quantized.values.to_int8_vector());
    print_values("restored", restored.to_vector());
    const auto activation = microllm::Tensor::from_vector(
        {1.0F, -2.0F, -1.0F, 0.5F, 2.0F, 3.0F}, {3, 2});
    print_values("matmul",
                 microllm::ops::int8_weight_matmul(
                     activation, quantized).to_vector());
    return 0;
}
