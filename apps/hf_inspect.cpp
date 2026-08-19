#include <filesystem>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>

#include <microllm/model/huggingface.h>

int main(int argc, char** argv) {
    try {
        if (argc != 3 || std::string(argv[1]) != "--config") {
            throw std::invalid_argument("usage: microllm_hf_inspect --config config.json");
        }
        const auto config = microllm::model::load_huggingface_config(
            std::filesystem::path(argv[2]));
        const auto& model = config.model;
        std::cout << std::setprecision(9)
                  << "{\"schema_version\":1,\"status\":\"compatible\""
                  << ",\"model_type\":\"" << config.model_type << "\""
                  << ",\"torch_dtype\":\"" << config.torch_dtype << "\""
                  << ",\"vocabulary_size\":" << model.vocabulary_size
                  << ",\"dimension\":" << model.dimension
                  << ",\"layers\":" << model.layers
                  << ",\"heads\":" << model.heads
                  << ",\"kv_heads\":" << model.kv_heads
                  << ",\"ffn_dimension\":" << model.ffn_dimension
                  << ",\"max_sequence_length\":" << model.max_sequence_length
                  << ",\"rope_base\":" << model.rope_base
                  << ",\"rms_norm_epsilon\":" << model.rms_norm_epsilon
                  << ",\"attention_bias\":" << (model.attention_bias ? "true" : "false")
                  << ",\"tie_embeddings\":" << (model.tie_embeddings ? "true" : "false")
                  << ",\"parameter_count\":" << model.parameter_count()
                  << ",\"bf16_weight_bytes\":" << model.weight_bytes(2) << "}\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "microllm_hf_inspect: " << error.what() << '\n';
        return 1;
    }
}
