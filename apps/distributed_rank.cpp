#include <chrono>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include <microllm/io/token_dataset.h>
#include <microllm/model/model.h>
#include <microllm/multi_gpu/communicator.h>
#include <microllm/runtime/runtime.h>
#include <microllm/training/optimizer.h>

namespace {

struct Options {
    std::string mode = "rank";
    int rank = -1;
    int world_size = 2;
    int local_rank = -1;
    std::filesystem::path id_file;
    std::uint64_t steps = 3;
    std::uint64_t seed = 607;
    std::uint64_t timeout_ms = 10000;
};

std::uint64_t number(const std::string& value, const char* name) {
    std::size_t consumed = 0;
    const auto parsed = std::stoull(value, &consumed);
    if (consumed != value.size()) {
        throw std::invalid_argument(std::string(name) + " is invalid");
    }
    return parsed;
}

Options parse(int argc, char** argv) {
    Options options;
    for (int index = 1; index < argc; ++index) {
        const std::string argument = argv[index];
        auto next = [&](const char* name) {
            if (++index >= argc) {
                throw std::invalid_argument(std::string(name) + " needs a value");
            }
            return std::string(argv[index]);
        };
        if (argument == "--mode") options.mode = next("--mode");
        else if (argument == "--rank") {
            options.rank = static_cast<int>(number(next("--rank"), "rank"));
        } else if (argument == "--world-size") {
            options.world_size = static_cast<int>(
                number(next("--world-size"), "world size"));
        } else if (argument == "--local-rank") {
            options.local_rank = static_cast<int>(
                number(next("--local-rank"), "local rank"));
        } else if (argument == "--id-file") {
            options.id_file = next("--id-file");
        } else if (argument == "--steps") {
            options.steps = number(next("--steps"), "steps");
        } else if (argument == "--seed") {
            options.seed = number(next("--seed"), "seed");
        } else if (argument == "--timeout-ms") {
            options.timeout_ms = number(next("--timeout-ms"), "timeout");
        } else {
            throw std::invalid_argument("unknown argument: " + argument);
        }
    }
    if ((options.mode != "rank" && options.mode != "reference") ||
        options.steps == 0 || options.timeout_ms == 0) {
        throw std::invalid_argument("distributed rank options are invalid");
    }
    if (options.mode == "rank" &&
        (options.world_size != 2 || options.rank < 0 ||
         options.rank >= options.world_size || options.local_rank < 0 ||
         options.id_file.empty())) {
        throw std::invalid_argument(
            "rank mode requires world size two, rank/local-rank, and ID file");
    }
    return options;
}

microllm::model::ModelConfig tiny_config() {
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

microllm::io::TokenBatch local_batch(int rank) {
    if (rank == 0) {
        return {microllm::Tensor::from_int32_vector({0, 1, 2, 3}, {1, 4}),
                microllm::Tensor::from_int32_vector({1, 2, 3, 0}, {1, 4})};
    }
    return {microllm::Tensor::from_int32_vector({3, 2, 1, 0}, {1, 4}),
            microllm::Tensor::from_int32_vector({2, 1, 0, 3}, {1, 4})};
}

microllm::io::TokenBatch global_batch() {
    return {microllm::Tensor::from_int32_vector(
                {0, 1, 2, 3, 3, 2, 1, 0}, {2, 4}),
            microllm::Tensor::from_int32_vector(
                {1, 2, 3, 0, 2, 1, 0, 3}, {2, 4})};
}

microllm::training::AdamWConfig optimizer_config() {
    return {.learning_rate = 0.005F,
            .beta1 = 0.9F,
            .beta2 = 0.99F,
            .epsilon = 1.0e-8F,
            .weight_decay = 0.0F};
}

void write_id(const std::filesystem::path& path,
              const microllm::multi_gpu::CommunicatorId& id) {
    const auto temporary = path.string() + ".rank0.tmp";
    {
        std::ofstream output(temporary, std::ios::binary | std::ios::trunc);
        if (!output) throw std::runtime_error("cannot create communicator ID file");
        output.write(reinterpret_cast<const char*>(id.data()),
                     static_cast<std::streamsize>(id.size()));
        if (!output) throw std::runtime_error("cannot write communicator ID file");
    }
    std::error_code error;
    std::filesystem::remove(path, error);
    error.clear();
    std::filesystem::rename(temporary, path, error);
    if (error) throw std::runtime_error("cannot publish communicator ID file");
}

microllm::multi_gpu::CommunicatorId wait_for_id(
    const std::filesystem::path& path, std::uint64_t timeout_ms) {
    const auto deadline = std::chrono::steady_clock::now() +
                          std::chrono::milliseconds(timeout_ms);
    while (std::chrono::steady_clock::now() < deadline) {
        std::error_code error;
        const auto size = std::filesystem::file_size(path, error);
        if (!error && size == microllm::multi_gpu::communicator_id_bytes()) {
            microllm::multi_gpu::CommunicatorId id(
                static_cast<std::size_t>(size));
            std::ifstream input(path, std::ios::binary);
            input.read(reinterpret_cast<char*>(id.data()),
                       static_cast<std::streamsize>(id.size()));
            if (input) return id;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    throw std::runtime_error("timed out waiting for communicator ID file");
}

void write_float_array(const std::vector<float>& values) {
    std::cout << '[';
    for (std::size_t index = 0; index < values.size(); ++index) {
        if (index != 0) std::cout << ',';
        std::cout << values[index];
    }
    std::cout << ']';
}

void write_result(const char* mode, int rank,
                  const std::vector<float>& losses,
                  microllm::model::TransformerModel& model) {
    std::cout << std::setprecision(9)
              << "{\"schema_version\":1,\"status\":\"pass\""
              << ",\"mode\":\"" << mode << "\",\"rank\":" << rank
              << ",\"losses\":";
    write_float_array(losses);
    std::cout << ",\"parameter_names\":[";
    const auto named = model.named_parameters();
    for (std::size_t index = 0; index < named.size(); ++index) {
        if (index != 0) std::cout << ',';
        std::cout << '\"' << named[index].first << '\"';
    }
    std::cout << "],\"parameters\":[";
    for (std::size_t index = 0; index < named.size(); ++index) {
        if (index != 0) std::cout << ',';
        write_float_array(named[index].second->data().to_vector());
    }
    std::cout << "]}\n";
}

void run_reference(const Options& options) {
    microllm::model::TransformerModel model(tiny_config(), options.seed);
    microllm::training::AdamW optimizer(model.parameters(), optimizer_config());
    const auto batch = global_batch();
    std::vector<float> losses;
    for (std::uint64_t step = 0; step < options.steps; ++step) {
        optimizer.zero_grad();
        const auto loss = model.loss(batch.inputs, batch.targets);
        losses.push_back(loss.data().to_vector()[0]);
        loss.backward();
        optimizer.step();
    }
    write_result("reference", -1, losses, model);
}

void run_rank(const Options& options) {
    microllm::multi_gpu::CommunicatorId id;
    if (options.rank == 0) {
        id = microllm::multi_gpu::create_communicator_id();
        write_id(options.id_file, id);
    } else {
        id = wait_for_id(options.id_file, options.timeout_ms);
    }
    microllm::multi_gpu::RankCommunicator communicator(
        options.rank, options.world_size, options.local_rank, id);
    microllm::model::TransformerModel model(tiny_config(), options.seed);
    model.to(communicator.device());
    microllm::training::AdamW optimizer(model.parameters(), optimizer_config());
    const auto batch = local_batch(options.rank);
    std::vector<float> losses;
    for (std::uint64_t step = 0; step < options.steps; ++step) {
        optimizer.zero_grad();
        const auto loss = model.loss(batch.inputs, batch.targets);
        losses.push_back(loss.data().to_vector()[0]);
        loss.backward();
        microllm::runtime::synchronize(communicator.device());
        for (auto* parameter : model.parameters()) {
            auto gradient = parameter->grad();
            communicator.enqueue_all_reduce_average_in_place(gradient);
        }
        communicator.synchronize();
        optimizer.step();
        microllm::runtime::synchronize(communicator.device());
    }
    write_result("rank", options.rank, losses, model);
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const auto options = parse(argc, argv);
        if (options.mode == "reference") run_reference(options);
        else run_rank(options);
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "microllm_distributed_rank: " << error.what() << '\n';
        return 1;
    }
}
