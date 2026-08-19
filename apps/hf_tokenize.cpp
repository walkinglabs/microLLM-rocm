#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>

#include <microllm/io/huggingface_bpe_tokenizer.h>

int main(int argc, char** argv) {
    try {
        if ((argc != 7 && argc != 9) || std::string(argv[1]) != "--vocab" ||
            std::string(argv[3]) != "--merges" || std::string(argv[5]) != "--text") {
            throw std::invalid_argument(
                "usage: microllm_hf_tokenize --vocab vocab.json --merges merges.txt --text TEXT");
        }
        auto tokenizer = microllm::io::HuggingFaceBpeTokenizer::load(
            std::filesystem::path(argv[2]), std::filesystem::path(argv[4]));
        const auto family = argc == 9 && std::string(argv[7]) == "--family"
                                ? std::string(argv[8]) : "qwen2";
        if (family == "deepseek-distill") {
            tokenizer.add_special_token("<｜end▁of▁sentence｜>", 151643);
            tokenizer.add_special_token("<｜User｜>", 151644);
            tokenizer.add_special_token("<｜Assistant｜>", 151645);
            tokenizer.add_special_token("<｜begin▁of▁sentence｜>", 151646);
            tokenizer.add_special_token("<think>", 151648);
            tokenizer.add_special_token("</think>", 151649);
        } else if (family == "qwen2") {
            tokenizer.add_special_token("<|endoftext|>", 151643);
            tokenizer.add_special_token("<|im_start|>", 151644);
            tokenizer.add_special_token("<|im_end|>", 151645);
        } else {
            throw std::invalid_argument("--family must be qwen2 or deepseek-distill");
        }
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
