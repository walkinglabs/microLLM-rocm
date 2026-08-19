#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <stdexcept>
#include <string>
#include <string_view>

#include <microllm/io/byte_tokenizer.h>
#include <microllm/io/token_dataset.h>
#include <microllm/model/model.h>
#include <microllm/training/checkpoint.h>
#include <microllm/training/trainer.h>

namespace {

struct Options {
    std::filesystem::path data;
    std::filesystem::path checkpoint;
    std::filesystem::path resume;
    std::string model = "tiny";
    std::string device = "cpu";
    std::uint64_t seed = 1;
    std::uint64_t steps = 10;
    std::int64_t batch = 1;
    std::int64_t context = 16;
    float learning_rate = 1.0e-3F;
};

std::int64_t integer(const char* value, const char* name) {
    char* end = nullptr;
    const auto parsed = std::strtoll(value, &end, 10);
    if (end == value || *end != '\0') throw std::invalid_argument(std::string("invalid ") + name);
    return parsed;
}

float floating(const char* value, const char* name) {
    char* end = nullptr;
    const auto parsed = std::strtof(value, &end);
    if (end == value || *end != '\0') throw std::invalid_argument(std::string("invalid ") + name);
    return parsed;
}

Options parse(int argc, char** argv) {
    Options options;
    for (int index = 1; index < argc; index += 2) {
        if (index + 1 >= argc) throw std::invalid_argument("option is missing a value");
        const std::string_view name(argv[index]);
        if (name == "--data") options.data = argv[index + 1];
        else if (name == "--checkpoint") options.checkpoint = argv[index + 1];
        else if (name == "--resume") options.resume = argv[index + 1];
        else if (name == "--model") options.model = argv[index + 1];
        else if (name == "--device") options.device = argv[index + 1];
        else if (name == "--seed") options.seed = static_cast<std::uint64_t>(integer(argv[index + 1], "seed"));
        else if (name == "--steps") options.steps = static_cast<std::uint64_t>(integer(argv[index + 1], "steps"));
        else if (name == "--batch") options.batch = integer(argv[index + 1], "batch");
        else if (name == "--context") options.context = integer(argv[index + 1], "context");
        else if (name == "--learning-rate") options.learning_rate = floating(argv[index + 1], "learning-rate");
        else throw std::invalid_argument("unknown option: " + std::string(name));
    }
    if (options.data.empty()) throw std::invalid_argument("--data is required");
    if (options.model != "tiny" && options.model != "s" && options.model != "m") {
        throw std::invalid_argument("--model must be tiny, s, or m");
    }
    if (options.device != "cpu" && options.device != "hip") {
        throw std::invalid_argument("--device must be cpu or hip");
    }
    if (options.steps == 0 || options.batch <= 0 || options.context <= 0 ||
        !(options.learning_rate > 0.0F)) {
        throw std::invalid_argument("training numeric options are outside valid ranges");
    }
    return options;
}

microllm::model::ModelConfig config(const Options& options) {
    if (options.model == "s") return microllm::model::ModelConfig::model_s();
    if (options.model == "m") return microllm::model::ModelConfig::model_m();
    return {.vocabulary_size = 256,
            .dimension = 32,
            .layers = 2,
            .heads = 4,
            .kv_heads = 2,
            .ffn_dimension = 64,
            .max_sequence_length = 128,
            .rope_base = 10000.0F,
            .tie_embeddings = false};
}

std::string read_file(const std::filesystem::path& path) {
    std::ifstream input(path, std::ios::binary);
    if (!input) throw std::runtime_error("cannot open training data");
    return {std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>()};
}

microllm::training::NamedParameters named_parameters(microllm::model::TransformerModel& model) {
    const auto values = model.named_parameters();
    return {values.begin(), values.end()};
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const auto options = parse(argc, argv);
        const auto model_config = config(options);
        if (options.context > model_config.max_sequence_length) {
            throw std::invalid_argument("context exceeds model configuration");
        }
        const auto text = read_file(options.data);
        const microllm::io::ByteTokenizer tokenizer;
        microllm::io::TokenDataset dataset(tokenizer.encode(text), options.context);
        microllm::model::TransformerModel model(model_config, options.seed);
        if (options.device == "hip") model.to(microllm::Device::hip());
        const microllm::training::AdamWConfig optimizer_config{
            .learning_rate = options.learning_rate,
            .beta1 = 0.9F,
            .beta2 = 0.999F,
            .epsilon = 1.0e-8F,
            .weight_decay = 0.01F};
        microllm::training::AdamW optimizer(model.parameters(), optimizer_config);
        microllm::training::ExperimentState experiment{
            .global_step = 0,
            .data_cursor = 0,
            .rng_state = "seed=" + std::to_string(options.seed),
            .model_config = model_config.summary(),
            .data_config = dataset.summary()};
        if (!options.resume.empty()) {
            microllm::training::restore_checkpoint(
                microllm::training::load_checkpoint(options.resume), named_parameters(model),
                optimizer, experiment);
            dataset.set_cursor(experiment.data_cursor);
        }
        std::cout << "model=" << model_config.summary() << '\n';
        std::cout << "data=" << dataset.summary() << '\n';
        std::cout << "device=" << model.device().str() << '\n';
        for (std::uint64_t iteration = 0; iteration < options.steps; ++iteration) {
            ++experiment.global_step;
            const auto metrics = microllm::training::train_step(
                model, optimizer, dataset.next_batch(options.batch), experiment.global_step);
            experiment.data_cursor = dataset.cursor();
            std::cout << std::setprecision(9)
                      << "{\"step\":" << metrics.step << ",\"loss\":" << metrics.loss
                      << ",\"gradient_l2_norm\":" << metrics.gradient_l2_norm
                      << ",\"data_cursor\":" << experiment.data_cursor << "}\n";
        }
        if (!options.checkpoint.empty()) {
            microllm::training::save_checkpoint(options.checkpoint, named_parameters(model),
                                                optimizer, experiment);
            std::cout << "checkpoint=" << options.checkpoint << '\n';
        }
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "microllm_train: " << error.what() << '\n';
        return 1;
    }
}
