#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include <microllm/multi_gpu/data_parallel.h>
#include <microllm/profiling/trace.h>

namespace {

struct Options {
    std::uint64_t steps = 3;
    std::size_t bucket_bytes = 4U * 1024U * 1024U;
    std::uint64_t seed = 601;
    std::filesystem::path trace;
};

std::uint64_t number(const std::string& value, const char* name) {
    std::size_t consumed = 0;
    const auto parsed = std::stoull(value, &consumed);
    if (consumed != value.size()) throw std::invalid_argument(std::string(name) + " is invalid");
    return parsed;
}

Options parse(int argc, char** argv) {
    Options options;
    for (int index = 1; index < argc; ++index) {
        const std::string argument = argv[index];
        auto next = [&](const char* name) {
            if (++index >= argc) throw std::invalid_argument(std::string(name) + " needs a value");
            return std::string(argv[index]);
        };
        if (argument == "--steps") options.steps = number(next("--steps"), "steps");
        else if (argument == "--bucket-bytes") {
            options.bucket_bytes = static_cast<std::size_t>(
                number(next("--bucket-bytes"), "bucket bytes"));
        } else if (argument == "--seed") options.seed = number(next("--seed"), "seed");
        else if (argument == "--trace") options.trace = next("--trace");
        else throw std::invalid_argument("unknown argument: " + argument);
    }
    if (options.steps == 0 || options.bucket_bytes < sizeof(float)) {
        throw std::invalid_argument("steps and bucket size must be positive");
    }
    return options;
}

microllm::model::ModelConfig config() {
    return {.vocabulary_size = 8,
            .dimension = 8,
            .layers = 1,
            .heads = 2,
            .kv_heads = 1,
            .ffn_dimension = 16,
            .max_sequence_length = 4,
            .rope_base = 10000.0F,
            .tie_embeddings = false};
}

std::vector<microllm::io::TokenBatch> batches() {
    return {
        {microllm::Tensor::from_int32_vector({0, 1, 2, 3}, {1, 4}),
         microllm::Tensor::from_int32_vector({1, 2, 3, 0}, {1, 4})},
        {microllm::Tensor::from_int32_vector({3, 2, 1, 0}, {1, 4}),
         microllm::Tensor::from_int32_vector({2, 1, 0, 3}, {1, 4})},
    };
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const auto options = parse(argc, argv);
        const microllm::training::AdamWConfig optimizer{
            .learning_rate = 0.005F,
            .beta1 = 0.9F,
            .beta2 = 0.99F,
            .epsilon = 1.0e-8F,
            .weight_decay = 0.0F,
        };
        microllm::multi_gpu::DataParallelTrainer trainer(
            config(), options.seed,
            {.device_indices = {0, 1},
             .maximum_bucket_bytes = options.bucket_bytes,
             .optimizer = optimizer});
        microllm::profiling::TraceOptions trace_options;
        trace_options.phase = "distributed_training";
        trace_options.record_operators = false;
        trace_options.record_layers = true;
        trace_options.record_model = true;
        trace_options.capture_values = false;
        microllm::profiling::TraceSession trace("microllm", "distributed-train",
                                                trace_options);
        {
            microllm::profiling::ScopedTraceSession active(trace);
            for (std::uint64_t step = 1; step <= options.steps; ++step) {
                trace.set_iteration(step - 1);
                const auto metrics = trainer.step(batches(), step);
                std::cout << "{\"step\":" << metrics.step
                          << ",\"mean_loss\":" << metrics.mean_loss
                          << ",\"bucket_count\":" << metrics.buckets.bucket_count
                          << ",\"parameter_max_difference\":"
                          << metrics.maximum_parameter_difference
                          << ",\"forward_backward_ms\":" << metrics.forward_backward_ms
                          << ",\"communication_ms\":" << metrics.communication_ms
                          << ",\"optimizer_ms\":" << metrics.optimizer_ms
                          << ",\"total_ms\":" << metrics.total_ms << "}\n";
            }
        }
        if (!options.trace.empty()) trace.write_jsonl(options.trace);
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "microllm_distributed_train: " << error.what() << '\n';
        return 1;
    }
}
