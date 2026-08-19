#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>

#include <microllm/io/huggingface_bpe_tokenizer.h>

int main(int argc, char** argv) {
    try {
        if (argc != 7 || std::string(argv[1]) != "--vocab" ||
            std::string(argv[3]) != "--merges" || std::string(argv[5]) != "--text") {
            throw std::invalid_argument(
                "usage: microllm_hf_tokenize --vocab vocab.json --merges merges.txt --text TEXT");
        }
        auto tokenizer = microllm::io::HuggingFaceBpeTokenizer::load(
            std::filesystem::path(argv[2]), std::filesystem::path(argv[4]));
        tokenizer.add_special_token("<|endoftext|>", 151643);
        tokenizer.add_special_token("<|im_start|>", 151644);
        tokenizer.add_special_token("<|im_end|>", 151645);
        const auto tokens = tokenizer.encode(argv[6]);
        std::cout << "{\"schema_version\":1,\"tokens\":[";
        for (std::size_t index = 0; index < tokens.size(); ++index) {
            if (index != 0) std::cout << ',';
            std::cout << tokens[index];
        }
        std::cout << "],\"roundtrip\":\"";
        for (const auto character : tokenizer.decode(tokens)) {
            if (character == '"' || character == '\\') std::cout << '\\';
            if (character == '\n') std::cout << "\\n";
            else if (character == '\r') std::cout << "\\r";
            else if (character == '\t') std::cout << "\\t";
            else std::cout << character;
        }
        std::cout << "\"}\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "microllm_hf_tokenize: " << error.what() << '\n';
        return 1;
    }
}
