#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <vector>

#include <microllm/io/token_dataset.h>
#include <microllm/inference/generator.h>
#include <microllm/model/model.h>
#include <microllm/training/trainer.h>

int main() {
    try {
        const microllm::model::ModelConfig config{
            .vocabulary_size = 8,
            .dimension = 8,
            .layers = 1,
            .heads = 2,
            .kv_heads = 1,
            .ffn_dimension = 16,
            .max_sequence_length = 8,
            .rope_base = 10000.0F,
            .tie_embeddings = false};
        std::vector<std::int32_t> tokens;
        for (int repeat = 0; repeat < 32; ++repeat) {
            for (const auto token : {0, 1, 2, 3}) tokens.push_back(token);
        }
        microllm::io::TokenDataset dataset(std::move(tokens), 4);
        microllm::model::TransformerModel model(config, 20260819);
        const microllm::training::AdamWConfig optimizer_config{
            .learning_rate = 0.02F,
            .beta1 = 0.9F,
            .beta2 = 0.99F,
            .epsilon = 1.0e-8F,
            .weight_decay = 0.0F};
        microllm::training::AdamW optimizer(model.parameters(), optimizer_config);

        float first_loss = 0.0F;
        float final_loss = 0.0F;
        for (std::uint64_t step = 1; step <= 40; ++step) {
            const auto metrics =
                microllm::training::train_step(model, optimizer, dataset.next_batch(2), step);
            if (step == 1) first_loss = metrics.loss;
            final_loss = metrics.loss;
            if (step == 1 || step % 10 == 0) {
                std::cout << "step=" << step << " loss=" << metrics.loss
                          << " grad_norm=" << metrics.gradient_l2_norm << '\n';
            }
        }
        std::cout << "first_loss=" << first_loss << '\n';
        std::cout << "final_loss=" << final_loss << '\n';
        if (!std::isfinite(final_loss) || !(final_loss < first_loss * 0.35F)) {
            throw std::runtime_error("tiny Transformer did not reach the overfit gate");
        }
        const auto generated = microllm::inference::generate(
            model, {0}, {.max_new_tokens = 7, .temperature = 0.0F, .top_k = 0,
                         .seed = 1, .kv_cache_layer_dtypes = {}});
        const std::vector<std::int32_t> expected{0, 1, 2, 3, 0, 1, 2, 3};
        const std::vector<std::int32_t> expected_trained_prefix{0, 1, 2, 3, 0};
        std::cout << "generated=";
        for (std::size_t index = 0; index < generated.size(); ++index) {
            if (index != 0) std::cout << ',';
            std::cout << generated[index];
        }
        std::cout << '\n';
        const auto trained_prefix_matches =
            std::equal(expected_trained_prefix.begin(), expected_trained_prefix.end(),
                       generated.begin());
        std::cout << "trained_prefix_matches=" << (trained_prefix_matches ? "true" : "false")
                  << '\n';
        std::cout << "beyond_training_context_failure="
                  << (generated == expected ? "false" : "true") << '\n';
        if (!trained_prefix_matches) {
            throw std::runtime_error("model did not learn the cycle inside its training context");
        }
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "tiny_overfit: " << error.what() << '\n';
        return 1;
    }
}
