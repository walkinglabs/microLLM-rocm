#include <cmath>
#include <cstdint>
#include <iostream>
#include <stdexcept>

#include <microllm/model/model.h>
#include <microllm/runtime/runtime.h>
#include <microllm/training/trainer.h>

int main() {
    try {
        if (!microllm::runtime::hip_compiled() || microllm::runtime::hip_device_count() == 0) {
            std::cout << "SKIP: no compiled and visible HIP device\n";
            return 0;
        }
        const microllm::model::ModelConfig config{.vocabulary_size = 8,
                                                   .dimension = 8,
                                                   .layers = 1,
                                                   .heads = 2,
                                                   .kv_heads = 1,
                                                   .ffn_dimension = 16,
                                                   .max_sequence_length = 4,
                                                   .rope_base = 10000.0F,
                                                   .tie_embeddings = false};
        microllm::model::TransformerModel model(config, 71);
        model.to(microllm::Device::hip());
        microllm::training::AdamW optimizer(
            model.parameters(), {.learning_rate = 0.02F,
                                 .beta1 = 0.9F,
                                 .beta2 = 0.99F,
                                 .epsilon = 1.0e-8F,
                                 .weight_decay = 0.0F});
        const microllm::io::TokenBatch batch{
            microllm::Tensor::from_int32_vector({0, 1, 2, 3}, {1, 4}),
            microllm::Tensor::from_int32_vector({1, 2, 3, 0}, {1, 4})};
        float first_loss = 0.0F;
        float final_loss = 0.0F;
        for (std::uint64_t step = 1; step <= 5; ++step) {
            const auto metrics = microllm::training::train_step(model, optimizer, batch, step);
            if (step == 1) first_loss = metrics.loss;
            final_loss = metrics.loss;
            std::cout << "step=" << step << " loss=" << metrics.loss
                      << " gradient_l2_norm=" << metrics.gradient_l2_norm << '\n';
        }
        const auto info = microllm::runtime::device_info(model.device());
        std::cout << "gpu=" << info.name << '\n';
        std::cout << "arch=" << info.architecture << '\n';
        if (!std::isfinite(final_loss) || !(final_loss < first_loss)) {
            throw std::runtime_error("tiny HIP training loss did not decrease");
        }
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "tiny_hip_train: " << error.what() << '\n';
        return 1;
    }
}
