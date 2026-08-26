#include <iostream>
#include <vector>

#include <microllm/core/tensor.h>

int main() {
    const auto tensor = microllm::Tensor::from_vector(
        {1.0F, 2.0F, 3.0F, 4.0F}, {2, 2});
    const auto transposed = tensor.transpose(0, 1);
    const auto values = transposed.to_vector();
    const std::vector<float> expected{1.0F, 3.0F, 2.0F, 4.0F};
    if (values != expected ||
        transposed.storage().data() != tensor.storage().data()) {
        return 1;
    }
    std::cout << "microLLM core package consumer: pass\n";
    return 0;
}
