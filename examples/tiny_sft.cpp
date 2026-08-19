#include <cmath>
#include <cstdint>
#include <iostream>
#include <stdexcept>

#include <microllm/io/sft.h>
#include <microllm/model/model.h>
#include <microllm/training/trainer.h>

int main() {
    try {
        const microllm::model::ModelConfig config{.vocabulary_size = 8,
                                                   .dimension = 8,
                                                   .layers = 1,
                                                   .heads = 2,
                                                   .kv_heads = 1,
                                                   .ffn_dimension = 16,
                                                   .max_sequence_length = 8,
                                                   .rope_base = 10000.0F,
                                                   .tie_embeddings = false};
        microllm::model::TransformerModel model(config, 109);
        microllm::training::AdamW optimizer(
            model.parameters(), {.learning_rate = 0.02F,
                                 .beta1 = 0.9F,
                                 .beta2 = 0.99F,
                                 .epsilon = 1.0e-8F,
                                 .weight_decay = 0.0F});
        const auto batch = microllm::io::make_sft_batch({1, 2, 3}, {4, 5});
        float first_loss = 0.0F;
        float final_loss = 0.0F;
        for (std::uint64_t step = 1; step <= 30; ++step) {
            const auto metrics = microllm::training::train_step(model, optimizer, batch, step);
            if (step == 1) first_loss = metrics.loss;
            final_loss = metrics.loss;
            if (step == 1 || step % 10 == 0) {
                std::cout << "step=" << step << " response_loss=" << metrics.loss << '\n';
            }
        }
        std::cout << "first_response_loss=" << first_loss << '\n';
        std::cout << "final_response_loss=" << final_loss << '\n';
        if (!std::isfinite(final_loss) || !(final_loss < first_loss * 0.2F)) {
            throw std::runtime_error("tiny SFT did not pass response-loss gate");
        }
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "tiny_sft: " << error.what() << '\n';
        return 1;
    }
}
