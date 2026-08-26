#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>

#include <microllm/io/safetensors.h>
#include <microllm/model/model.h>

int main(int argc, char** argv) {
    try {
        if (argc != 2) {
            throw std::invalid_argument(
                "hf_cli_fixture requires one output directory");
        }
        const std::filesystem::path directory(argv[1]);
        std::filesystem::create_directories(directory);
        microllm::model::ModelConfig config;
        config.vocabulary_size = 8;
        config.dimension = 8;
        config.layers = 1;
        config.heads = 2;
        config.kv_heads = 1;
        config.ffn_dimension = 16;
        config.max_sequence_length = 8;
        config.tie_embeddings = true;
        config.attention_bias = true;
        config.validate();
        microllm::model::TransformerModel model(config, 173);
        const auto native = model.state_dict();
        const auto mapping =
            microllm::model::qwen_style_weight_mapping(config);
        microllm::io::StateDict external;
        for (const auto& [target, source] : mapping) {
            auto tensor = native.at(target);
            if (source.transform ==
                microllm::model::WeightTransform::Transpose2D) {
                tensor = tensor.transpose(0, 1).contiguous();
            }
            external.emplace(source.name, std::move(tensor));
        }
        microllm::io::save_safetensors(
            directory / "model.safetensors", external);
        std::ofstream output(directory / "config.json",
                             std::ios::trunc);
        output
            << "{\"model_type\":\"qwen2\","
            << "\"bos_token_id\":0,\"eos_token_id\":0,"
            << "\"vocab_size\":8,\"hidden_size\":8,"
            << "\"hidden_act\":\"silu\","
            << "\"initializer_range\":0.02,"
            << "\"intermediate_size\":16,"
            << "\"num_hidden_layers\":1,"
            << "\"num_attention_heads\":2,"
            << "\"num_key_value_heads\":1,"
            << "\"max_position_embeddings\":8,"
            << "\"rms_norm_eps\":0.00001,"
            << "\"rope_theta\":10000,"
            << "\"use_sliding_window\":false,"
            << "\"sliding_window\":8,"
            << "\"max_window_layers\":1,"
            << "\"use_mrope\":false,"
            << "\"torch_dtype\":\"float32\","
            << "\"use_cache\":true,"
            << "\"tie_word_embeddings\":true,"
            << "\"attention_bias\":true}\n";
        if (!output) {
            throw std::runtime_error(
                "failed writing hf_cli_fixture config");
        }
        std::cout << "hf_cli_fixture: pass\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "hf_cli_fixture failed: " << error.what() << '\n';
        return 1;
    }
}
