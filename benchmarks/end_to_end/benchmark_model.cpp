#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include <microllm/inference/generator.h>
#include <microllm/model/model.h>
#include <microllm/runtime/memory.h>
#include <microllm/runtime/runtime.h>
#include <microllm/training/trainer.h>

namespace {

struct Options {
    std::string mode = "train";
    std::string model = "tiny";
    std::string device = "cpu";
    int steps = 5;
    int warmup = 2;
    std::int64_t batch = 1;
    std::int64_t context = 8;
    std::int64_t new_tokens = 16;
};

std::int64_t integer(const char* value, const char* name) {
    char* end = nullptr;
    const auto parsed = std::strtoll(value, &end, 10);
    if (end == value || *end != '\0') throw std::invalid_argument(std::string("invalid ") + name);
    return parsed;
}

Options parse(int argc, char** argv) {
    Options options;
    for (int index = 1; index < argc; index += 2) {
        if (index + 1 >= argc) throw std::invalid_argument("option is missing a value");
        const std::string_view name(argv[index]);
        if (name == "--mode") options.mode = argv[index + 1];
        else if (name == "--model") options.model = argv[index + 1];
        else if (name == "--device") options.device = argv[index + 1];
        else if (name == "--steps") options.steps = static_cast<int>(integer(argv[index + 1], "steps"));
        else if (name == "--warmup") options.warmup = static_cast<int>(integer(argv[index + 1], "warmup"));
        else if (name == "--batch") options.batch = integer(argv[index + 1], "batch");
        else if (name == "--context") options.context = integer(argv[index + 1], "context");
        else if (name == "--new-tokens") options.new_tokens = integer(argv[index + 1], "new-tokens");
        else throw std::invalid_argument("unknown option: " + std::string(name));
    }
    if (options.mode != "train" && options.mode != "generate") {
        throw std::invalid_argument("mode must be train or generate");
    }
    if (options.model != "tiny" && options.model != "model-s" &&
        options.model != "model-m") {
        throw std::invalid_argument("model must be tiny, model-s, or model-m");
    }
    if (options.device != "cpu" && options.device != "hip") {
        throw std::invalid_argument("device must be cpu or hip");
    }
    if (options.steps <= 0 || options.warmup < 0 || options.batch <= 0 ||
        options.context <= 0 || options.new_tokens <= 0) {
        throw std::invalid_argument("numeric options are outside valid ranges");
    }
    if (options.mode == "generate" && options.batch != 1) {
        throw std::invalid_argument("the first generation benchmark supports batch one");
    }
    return options;
}

microllm::model::ModelConfig config_for(const Options& options) {
    if (options.model == "model-s") return microllm::model::ModelConfig::model_s();
    if (options.model == "model-m") return microllm::model::ModelConfig::model_m();
    return {.vocabulary_size = 32,
            .dimension = 16,
            .layers = 2,
            .heads = 4,
            .kv_heads = 2,
            .ffn_dimension = 32,
            .max_sequence_length = 128,
            .rope_base = 10000.0F,
            .tie_embeddings = false};
}

microllm::io::TokenBatch fixed_batch(const microllm::model::ModelConfig& config,
                                     std::int64_t batch, std::int64_t context) {
    std::vector<std::int32_t> input(static_cast<std::size_t>(batch * context));
    std::vector<std::int32_t> target(input.size());
    for (std::size_t index = 0; index < input.size(); ++index) {
        input[index] = static_cast<std::int32_t>(index % static_cast<std::size_t>(config.vocabulary_size));
        target[index] = static_cast<std::int32_t>((input[index] + 1) % config.vocabulary_size);
    }
    return {microllm::Tensor::from_int32_vector(input, {batch, context}),
            microllm::Tensor::from_int32_vector(target, {batch, context})};
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const auto options = parse(argc, argv);
        auto config = config_for(options);
        if (options.context > config.max_sequence_length ||
            (options.mode == "generate" &&
             options.context + options.new_tokens > config.max_sequence_length)) {
            throw std::invalid_argument("benchmark sequence exceeds model context");
        }
        const auto device = options.device == "cpu" ? microllm::Device::cpu()
                                                     : microllm::Device::hip();
        if (device.is_hip() && microllm::runtime::hip_device_count() == 0) {
            throw std::runtime_error("HIP benchmark requested without a visible GPU");
        }
        microllm::runtime::reset_allocation_peak(microllm::Device::cpu());
        microllm::runtime::reset_allocation_peak(device);
        const auto wall_start = std::chrono::steady_clock::now();
        microllm::model::TransformerModel model(config, 20260819);
        if (device.is_hip()) model.to(device);
        microllm::runtime::synchronize(device);
        const auto construction_finish = std::chrono::steady_clock::now();
        const auto construction_seconds =
            std::chrono::duration<double>(construction_finish - wall_start).count();

        float first_loss = 0.0F;
        float final_loss = 0.0F;
        double output_guard = 0.0;
        double measured_seconds = 0.0;
        double warmup_seconds = 0.0;
        std::int64_t measured_tokens = 0;
        if (options.mode == "train") {
            microllm::training::AdamW optimizer(
                model.parameters(), {.learning_rate = 1.0e-3F,
                                     .beta1 = 0.9F,
                                     .beta2 = 0.999F,
                                     .epsilon = 1.0e-8F,
                                     .weight_decay = 0.01F});
            const auto batch = fixed_batch(config, options.batch, options.context);
            const auto warmup_start = std::chrono::steady_clock::now();
            for (int iteration = 0; iteration < options.warmup; ++iteration) {
                (void)microllm::training::train_step(model, optimizer, batch,
                                                      static_cast<std::uint64_t>(iteration));
            }
            microllm::runtime::synchronize(device);
            const auto warmup_finish = std::chrono::steady_clock::now();
            warmup_seconds =
                std::chrono::duration<double>(warmup_finish - warmup_start).count();
            const auto measured_start = std::chrono::steady_clock::now();
            for (int iteration = 0; iteration < options.steps; ++iteration) {
                const auto metrics = microllm::training::train_step(
                    model, optimizer, batch, static_cast<std::uint64_t>(iteration));
                if (iteration == 0) first_loss = metrics.loss;
                final_loss = metrics.loss;
                microllm::runtime::synchronize(device);
            }
            const auto measured_finish = std::chrono::steady_clock::now();
            measured_tokens = static_cast<std::int64_t>(options.steps) * options.batch * options.context;
            measured_seconds =
                std::chrono::duration<double>(measured_finish - measured_start).count();
            output_guard = final_loss;
        } else {
            const std::vector<std::int32_t> prompt(static_cast<std::size_t>(options.context), 1);
            const auto warmup_start = std::chrono::steady_clock::now();
            for (int iteration = 0; iteration < options.warmup; ++iteration) {
                (void)microllm::inference::generate(
                    model, prompt, {.max_new_tokens = options.new_tokens,
                                    .temperature = 0.0F,
                                    .top_k = 0,
                                    .seed = static_cast<std::uint64_t>(iteration)});
            }
            microllm::runtime::synchronize(device);
            const auto warmup_finish = std::chrono::steady_clock::now();
            warmup_seconds =
                std::chrono::duration<double>(warmup_finish - warmup_start).count();
            const auto measured_start = std::chrono::steady_clock::now();
            for (int iteration = 0; iteration < options.steps; ++iteration) {
                const auto generated = microllm::inference::generate(
                    model, prompt, {.max_new_tokens = options.new_tokens,
                                    .temperature = 0.0F,
                                    .top_k = 0,
                                    .seed = static_cast<std::uint64_t>(iteration)});
                for (const auto token : generated) output_guard += token;
                microllm::runtime::synchronize(device);
            }
            const auto measured_finish = std::chrono::steady_clock::now();
            measured_tokens = static_cast<std::int64_t>(options.steps) * options.new_tokens;
            measured_seconds =
                std::chrono::duration<double>(measured_finish - measured_start).count();
        }
        const auto wall_finish = std::chrono::steady_clock::now();
        const auto total_wall = std::chrono::duration<double>(wall_finish - wall_start).count();
        const auto tokens_per_second = static_cast<double>(measured_tokens) / measured_seconds;
        const auto milliseconds_per_token =
            measured_seconds * 1000.0 / static_cast<double>(measured_tokens);
        const auto tokens_per_second_with_setup =
            static_cast<double>(measured_tokens) / total_wall;
        const auto cpu_memory = microllm::runtime::allocation_stats(microllm::Device::cpu());
        const auto device_memory = microllm::runtime::allocation_stats(device);
        const auto info = device.is_cpu() ? microllm::runtime::DeviceInfo{device, "host CPU", "host"}
                                          : microllm::runtime::device_info(device);
        std::cout << std::setprecision(9)
                  << "{\"schema_version\":1,\"engine_version\":\"" << MICROLLM_VERSION
                  << "\",\"mode\":\"" << options.mode
                  << "\",\"model\":\"" << options.model
                  << "\",\"device\":\"" << options.device
                  << "\",\"device_name\":\"" << info.name
                  << "\",\"architecture\":\"" << info.architecture
                  << "\",\"hip_runtime_version\":" << microllm::runtime::hip_runtime_version()
                  << ",\"hip_driver_version\":" << microllm::runtime::hip_driver_version()
                  << ",\"dtype\":\"float32\",\"batch\":" << options.batch
                  << ",\"parameter_count\":" << model.parameter_count()
                  << ",\"fp32_weight_bytes\":" << config.weight_bytes(sizeof(float))
                  << ",\"context\":" << options.context
                  << ",\"steps\":" << options.steps
                  << ",\"warmup\":" << options.warmup
                  << ",\"new_tokens\":" << options.new_tokens
                  << ",\"measured_tokens\":" << measured_tokens
                  << ",\"measured_wall_seconds\":" << measured_seconds
                  << ",\"tokens_per_second\":" << tokens_per_second
                  << ",\"milliseconds_per_token\":" << milliseconds_per_token
                  << ",\"model_construction_seconds\":" << construction_seconds
                  << ",\"warmup_seconds\":" << warmup_seconds
                  << ",\"wall_seconds_with_setup\":" << total_wall
                  << ",\"tokens_per_second_with_setup\":" << tokens_per_second_with_setup
                  << ",\"first_loss\":" << first_loss
                  << ",\"final_loss\":" << final_loss
                  << ",\"cpu_current_engine_bytes\":" << cpu_memory.current_bytes
                  << ",\"cpu_peak_engine_bytes\":" << cpu_memory.peak_bytes
                  << ",\"cpu_total_allocated_engine_bytes\":"
                  << cpu_memory.total_allocated_bytes
                  << ",\"device_current_engine_bytes\":" << device_memory.current_bytes
                  << ",\"device_peak_engine_bytes\":" << device_memory.peak_bytes
                  << ",\"device_total_allocated_engine_bytes\":"
                  << device_memory.total_allocated_bytes
                  << ",\"output_guard\":" << output_guard << "}\n";
        return std::isfinite(tokens_per_second) && tokens_per_second > 0.0 ? 0 : 2;
    } catch (const std::exception& error) {
        std::cerr << "benchmark_model: " << error.what() << '\n';
        return 1;
    }
}
