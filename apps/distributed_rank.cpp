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
#include <microllm/io/safetensors.h>
#include <microllm/model/model.h>
#include <microllm/multi_gpu/communicator.h>
#include <microllm/multi_gpu/gradient_bucket.h>
#include <microllm/runtime/memory.h>
#include <microllm/runtime/runtime.h>
#include <microllm/training/checkpoint.h>
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
    std::string reducer = "per-parameter";
    std::size_t bucket_bytes = 4096;
    std::string model = "tiny";
    std::size_t context = 0;
    std::filesystem::path parameter_file;
    std::filesystem::path checkpoint_file;
    std::filesystem::path checkpoint_ready_file;
    std::filesystem::path resume_file;
    bool inject_checkpoint_failure = false;
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
        } else if (argument == "--reducer") {
            options.reducer = next("--reducer");
        } else if (argument == "--bucket-bytes") {
            options.bucket_bytes = static_cast<std::size_t>(
                number(next("--bucket-bytes"), "bucket bytes"));
        } else if (argument == "--model") {
            options.model = next("--model");
        } else if (argument == "--context") {
            options.context = static_cast<std::size_t>(
                number(next("--context"), "context"));
        } else if (argument == "--parameter-file") {
            options.parameter_file = next("--parameter-file");
        } else if (argument == "--checkpoint-file") {
            options.checkpoint_file = next("--checkpoint-file");
        } else if (argument == "--checkpoint-ready-file") {
            options.checkpoint_ready_file = next("--checkpoint-ready-file");
        } else if (argument == "--resume-file") {
            options.resume_file = next("--resume-file");
        } else if (argument == "--inject-checkpoint-failure") {
            options.inject_checkpoint_failure = true;
        } else {
            throw std::invalid_argument("unknown argument: " + argument);
        }
    }
    if ((options.mode != "rank" && options.mode != "reference") ||
        options.steps == 0 || options.timeout_ms == 0 ||
        (options.reducer != "per-parameter" && options.reducer != "bucket" &&
         options.reducer != "persistent-bucket" &&
         options.reducer != "bucket-views" &&
         options.reducer != "overlap-views") ||
        options.bucket_bytes < sizeof(float) ||
        (options.model != "tiny" && options.model != "model-s")) {
        throw std::invalid_argument("distributed rank options are invalid");
    }
    if (options.context == 0) {
        options.context = options.model == "tiny" ? 4U : 32U;
    }
    if ((options.model == "tiny" && options.context != 4) ||
        (options.model == "model-s" && options.context > 512)) {
        throw std::invalid_argument(
            "distributed rank context exceeds the model contract");
    }
    if (options.mode == "rank" &&
        (options.world_size != 2 || options.rank < 0 ||
         options.rank >= options.world_size || options.local_rank < 0 ||
         options.id_file.empty())) {
        throw std::invalid_argument(
            "rank mode requires world size two, rank/local-rank, and ID file");
    }
    if (options.model == "model-s" && options.parameter_file.empty()) {
        throw std::invalid_argument(
            "Model-S rank/reference mode requires a parameter file");
    }
    if (options.mode == "reference" &&
        (!options.checkpoint_file.empty() ||
         !options.checkpoint_ready_file.empty() ||
         !options.resume_file.empty() || options.inject_checkpoint_failure)) {
        throw std::invalid_argument(
            "checkpoint ownership options require rank mode");
    }
    if (options.checkpoint_file.empty() !=
        options.checkpoint_ready_file.empty()) {
        throw std::invalid_argument(
            "checkpoint file and ready file must be provided together");
    }
    if (options.inject_checkpoint_failure &&
        options.checkpoint_file.empty()) {
        throw std::invalid_argument(
            "checkpoint failure injection requires a checkpoint file");
    }
    return options;
}

microllm::model::ModelConfig model_config(const std::string& model) {
    if (model == "model-s") return microllm::model::ModelConfig::model_s();
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

microllm::io::TokenBatch local_batch(
    const microllm::model::ModelConfig& config, int rank,
    std::size_t context) {
    if (config.vocabulary_size == 8 && rank == 0) {
        return {microllm::Tensor::from_int32_vector({0, 1, 2, 3}, {1, 4}),
                microllm::Tensor::from_int32_vector({1, 2, 3, 0}, {1, 4})};
    }
    if (config.vocabulary_size == 8) {
        return {microllm::Tensor::from_int32_vector({3, 2, 1, 0}, {1, 4}),
                microllm::Tensor::from_int32_vector({2, 1, 0, 3}, {1, 4})};
    }
    std::vector<std::int32_t> inputs(context);
    std::vector<std::int32_t> targets(context);
    for (std::size_t index = 0; index < context; ++index) {
        const auto token = static_cast<std::int32_t>(
            (static_cast<std::size_t>(rank) * 97 + index * 13 + 7) %
            static_cast<std::size_t>(config.vocabulary_size));
        inputs[index] = token;
        targets[index] = (token + 1) %
                         static_cast<std::int32_t>(config.vocabulary_size);
    }
    const auto context_dimension = static_cast<std::int64_t>(context);
    return {microllm::Tensor::from_int32_vector(
                inputs, {1, context_dimension}),
            microllm::Tensor::from_int32_vector(
                targets, {1, context_dimension})};
}

microllm::io::TokenBatch global_batch(
    const microllm::model::ModelConfig& config, std::size_t context) {
    const auto first = local_batch(config, 0, context);
    const auto second = local_batch(config, 1, context);
    auto inputs = first.inputs.to_int32_vector();
    const auto second_inputs = second.inputs.to_int32_vector();
    inputs.insert(inputs.end(), second_inputs.begin(), second_inputs.end());
    auto targets = first.targets.to_int32_vector();
    const auto second_targets = second.targets.to_int32_vector();
    targets.insert(targets.end(), second_targets.begin(), second_targets.end());
    const auto context_dimension = first.inputs.size(1);
    return {microllm::Tensor::from_int32_vector(
                inputs, {2, context_dimension}),
            microllm::Tensor::from_int32_vector(
                targets, {2, context_dimension})};
}

microllm::training::AdamWConfig optimizer_config() {
    return {.learning_rate = 0.005F,
            .beta1 = 0.9F,
            .beta2 = 0.99F,
            .epsilon = 1.0e-8F,
            .weight_decay = 0.0F};
}

microllm::training::NamedParameters named_parameters(
    microllm::model::TransformerModel& model) {
    const auto values = model.named_parameters();
    return {values.begin(), values.end()};
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

void publish_checkpoint_ready(
    const std::filesystem::path& path, std::uint64_t step) {
    const auto temporary = path.string() + ".rank0.tmp";
    {
        std::ofstream output(temporary, std::ios::trunc);
        if (!output) {
            throw std::runtime_error(
                "cannot create checkpoint ready temporary file");
        }
        output << "step=" << step << '\n';
        output.flush();
        if (!output) {
            throw std::runtime_error("cannot write checkpoint ready file");
        }
    }
    std::error_code error;
    std::filesystem::remove(path, error);
    error.clear();
    std::filesystem::rename(temporary, path, error);
    if (error) {
        throw std::runtime_error("cannot publish checkpoint ready file");
    }
}

void wait_for_checkpoint_ready(
    const std::filesystem::path& path, std::uint64_t step,
    std::uint64_t timeout_ms) {
    const auto expected = "step=" + std::to_string(step);
    const auto deadline = std::chrono::steady_clock::now() +
                          std::chrono::milliseconds(timeout_ms);
    while (std::chrono::steady_clock::now() < deadline) {
        std::ifstream input(path);
        std::string value;
        if (input && std::getline(input, value) && value == expected) return;
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    throw std::runtime_error("timed out waiting for checkpoint publication");
}

template <typename T>
void write_number_array(const std::vector<T>& values) {
    std::cout << '[';
    for (std::size_t index = 0; index < values.size(); ++index) {
        if (index != 0) std::cout << ',';
        std::cout << values[index];
    }
    std::cout << ']';
}

struct ReducerStats {
    std::size_t collectives = 0;
    std::size_t buckets = 0;
    std::size_t pack_copies = 0;
    std::size_t unpack_copies = 0;
    std::size_t gradient_views = 0;
    std::vector<std::size_t> step_collectives;
    std::vector<std::size_t> step_buckets;
    std::vector<std::size_t> step_pack_copies;
    std::vector<std::size_t> step_unpack_copies;
    std::vector<std::size_t> step_gradient_views;
    std::vector<std::size_t> step_allocation_calls;
    std::vector<std::size_t> step_backend_allocation_calls;
    std::vector<std::size_t> step_deallocation_calls;
    std::vector<std::size_t> step_total_allocated_bytes;
    std::vector<std::size_t> step_plan_reused;
    std::vector<std::size_t> step_current_bytes_before;
    std::vector<std::size_t> step_current_bytes_after;
    std::vector<std::size_t> step_peak_bytes_after;
    bool persistent_storage = false;
    std::size_t plan_reuses = 0;
    std::size_t plan_capacity_elements = 0;
    std::size_t plan_capacity_bytes = 0;
    std::size_t overlap_steps = 0;
    std::size_t overlapped_buckets = 0;
    std::vector<std::size_t> step_overlap_enabled;
    std::vector<std::size_t> step_overlapped_buckets;
};

struct PhaseTimings {
    double training_ms = 0.0;
    double forward_backward_ms = 0.0;
    double reducer_ms = 0.0;
    double optimizer_ms = 0.0;
    std::vector<double> step_training_ms;
    std::vector<double> step_forward_backward_ms;
    std::vector<double> step_reducer_ms;
    std::vector<double> step_optimizer_ms;
};

struct CheckpointReport {
    bool resumed = false;
    bool checkpoint_requested = false;
    bool checkpoint_written = false;
    bool checkpoint_verified = false;
    std::uint64_t initial_step = 0;
    std::uint64_t final_step = 0;
    std::uint64_t optimizer_step = 0;
    double resume_ms = 0.0;
    double checkpoint_write_ms = 0.0;
    double checkpoint_wait_ms = 0.0;
    double checkpoint_verify_ms = 0.0;
    std::filesystem::path checkpoint_file;
    std::filesystem::path checkpoint_ready_file;
};

using SteadyClock = std::chrono::steady_clock;

double elapsed_ms(SteadyClock::time_point begin, SteadyClock::time_point end) {
    return std::chrono::duration<double, std::milli>(end - begin).count();
}

std::size_t counter_delta(std::size_t after, std::size_t before) {
    if (after < before) {
        throw std::runtime_error("allocation counter moved backwards");
    }
    return after - before;
}

void write_result(const char* mode, int rank,
                  const std::vector<float>& losses,
                  microllm::model::TransformerModel& model,
                  const std::string& reducer, const ReducerStats& reducer_stats,
                  const PhaseTimings& timings,
                  const microllm::runtime::AllocationStats& allocation,
                  const CheckpointReport& checkpoint,
                  const std::string& model_name,
                  std::size_t context,
                  const std::filesystem::path& parameter_file) {
    std::cout << std::setprecision(9)
              << "{\"schema_version\":1,\"status\":\"pass\""
              << ",\"mode\":\"" << mode << "\",\"rank\":" << rank
              << ",\"model\":\"" << model_name << "\""
              << ",\"context\":" << context
              << ",\"reducer\":\"" << reducer << "\""
              << ",\"collectives\":" << reducer_stats.collectives
              << ",\"buckets\":" << reducer_stats.buckets
              << ",\"pack_copies\":" << reducer_stats.pack_copies
              << ",\"unpack_copies\":" << reducer_stats.unpack_copies
              << ",\"gradient_views\":" << reducer_stats.gradient_views
              << ",\"persistent_storage\":"
              << (reducer_stats.persistent_storage ? "true" : "false")
              << ",\"plan_reuses\":" << reducer_stats.plan_reuses
              << ",\"plan_capacity_elements\":"
              << reducer_stats.plan_capacity_elements
              << ",\"plan_capacity_bytes\":"
              << reducer_stats.plan_capacity_bytes
              << ",\"overlap_steps\":" << reducer_stats.overlap_steps
              << ",\"overlapped_buckets\":"
              << reducer_stats.overlapped_buckets
              << ",\"engine_current_bytes\":" << allocation.current_bytes
              << ",\"engine_peak_bytes\":" << allocation.peak_bytes
              << ",\"engine_cached_bytes\":" << allocation.cached_bytes
              << ",\"engine_reserved_bytes\":" << allocation.reserved_bytes
              << ",\"engine_allocation_calls\":"
              << allocation.allocation_calls
              << ",\"engine_backend_allocation_calls\":"
              << allocation.backend_allocation_calls
              << ",\"resumed\":"
              << (checkpoint.resumed ? "true" : "false")
              << ",\"checkpoint_requested\":"
              << (checkpoint.checkpoint_requested ? "true" : "false")
              << ",\"checkpoint_written\":"
              << (checkpoint.checkpoint_written ? "true" : "false")
              << ",\"checkpoint_verified\":"
              << (checkpoint.checkpoint_verified ? "true" : "false")
              << ",\"initial_step\":" << checkpoint.initial_step
              << ",\"final_step\":" << checkpoint.final_step
              << ",\"optimizer_step\":" << checkpoint.optimizer_step
              << ",\"resume_ms\":" << checkpoint.resume_ms
              << ",\"checkpoint_write_ms\":"
              << checkpoint.checkpoint_write_ms
              << ",\"checkpoint_wait_ms\":"
              << checkpoint.checkpoint_wait_ms
              << ",\"checkpoint_verify_ms\":"
              << checkpoint.checkpoint_verify_ms
              << ",\"checkpoint_file\":\""
              << checkpoint.checkpoint_file.string() << "\""
              << ",\"checkpoint_ready_file\":\""
              << checkpoint.checkpoint_ready_file.string() << "\""
              << ",\"training_ms\":" << timings.training_ms
              << ",\"forward_backward_ms\":"
              << timings.forward_backward_ms
              << ",\"reducer_ms\":" << timings.reducer_ms
              << ",\"optimizer_ms\":" << timings.optimizer_ms
              << ",\"step_training_ms\":";
    write_number_array(timings.step_training_ms);
    std::cout << ",\"step_forward_backward_ms\":";
    write_number_array(timings.step_forward_backward_ms);
    std::cout << ",\"step_reducer_ms\":";
    write_number_array(timings.step_reducer_ms);
    std::cout << ",\"step_optimizer_ms\":";
    write_number_array(timings.step_optimizer_ms);
    std::cout << ",\"step_collectives\":";
    write_number_array(reducer_stats.step_collectives);
    std::cout << ",\"step_buckets\":";
    write_number_array(reducer_stats.step_buckets);
    std::cout << ",\"step_pack_copies\":";
    write_number_array(reducer_stats.step_pack_copies);
    std::cout << ",\"step_unpack_copies\":";
    write_number_array(reducer_stats.step_unpack_copies);
    std::cout << ",\"step_gradient_views\":";
    write_number_array(reducer_stats.step_gradient_views);
    std::cout << ",\"step_reducer_allocation_calls\":";
    write_number_array(reducer_stats.step_allocation_calls);
    std::cout << ",\"step_reducer_backend_allocation_calls\":";
    write_number_array(reducer_stats.step_backend_allocation_calls);
    std::cout << ",\"step_reducer_deallocation_calls\":";
    write_number_array(reducer_stats.step_deallocation_calls);
    std::cout << ",\"step_reducer_total_allocated_bytes\":";
    write_number_array(reducer_stats.step_total_allocated_bytes);
    std::cout << ",\"step_plan_reused\":";
    write_number_array(reducer_stats.step_plan_reused);
    std::cout << ",\"step_reducer_current_bytes_before\":";
    write_number_array(reducer_stats.step_current_bytes_before);
    std::cout << ",\"step_reducer_current_bytes_after\":";
    write_number_array(reducer_stats.step_current_bytes_after);
    std::cout << ",\"step_reducer_peak_bytes_after\":";
    write_number_array(reducer_stats.step_peak_bytes_after);
    std::cout << ",\"step_overlap_enabled\":";
    write_number_array(reducer_stats.step_overlap_enabled);
    std::cout << ",\"step_overlapped_buckets\":";
    write_number_array(reducer_stats.step_overlapped_buckets);
    std::cout << ",\"losses\":";
    write_number_array(losses);
    std::cout << ",\"parameter_names\":[";
    const auto named = model.named_parameters();
    for (std::size_t index = 0; index < named.size(); ++index) {
        if (index != 0) std::cout << ',';
        std::cout << '\"' << named[index].first << '\"';
    }
    std::cout << "],\"parameter_file\":\"" << parameter_file.string() << "\""
              << ",\"parameter_count\":" << model.parameter_count();
    if (model_name == "tiny") {
        std::cout << ",\"parameters\":[";
        for (std::size_t index = 0; index < named.size(); ++index) {
            if (index != 0) std::cout << ',';
            write_number_array(named[index].second->data().to_vector());
        }
        std::cout << ']';
    }
    std::cout << "}\n";
}

void run_reference(const Options& options) {
    const auto config = model_config(options.model);
    microllm::model::TransformerModel model(config, options.seed);
    microllm::training::AdamW optimizer(model.parameters(), optimizer_config());
    const auto batch = global_batch(config, options.context);
    std::vector<float> losses;
    PhaseTimings timings;
    const auto training_begin = SteadyClock::now();
    for (std::uint64_t step = 0; step < options.steps; ++step) {
        const auto step_begin = SteadyClock::now();
        const auto forward_backward_begin = SteadyClock::now();
        optimizer.zero_grad();
        const auto loss = model.loss(batch.inputs, batch.targets);
        losses.push_back(loss.data().to_vector()[0]);
        loss.backward();
        const auto forward_backward_end = SteadyClock::now();
        const auto forward_backward_elapsed = elapsed_ms(
            forward_backward_begin, forward_backward_end);
        timings.forward_backward_ms += forward_backward_elapsed;
        timings.step_forward_backward_ms.push_back(forward_backward_elapsed);
        timings.step_reducer_ms.push_back(0.0);
        const auto optimizer_begin = SteadyClock::now();
        optimizer.step();
        const auto optimizer_elapsed = elapsed_ms(
            optimizer_begin, SteadyClock::now());
        timings.optimizer_ms += optimizer_elapsed;
        timings.step_optimizer_ms.push_back(optimizer_elapsed);
        timings.step_training_ms.push_back(
            elapsed_ms(step_begin, SteadyClock::now()));
    }
    timings.training_ms = elapsed_ms(training_begin, SteadyClock::now());
    if (!options.parameter_file.empty()) {
        microllm::io::save_safetensors(
            options.parameter_file, model.state_dict());
    }
    const auto allocation =
        microllm::runtime::allocation_stats(microllm::Device::cpu());
    write_result("reference", -1, losses, model, "reference", {}, timings,
                 allocation, {}, options.model, options.context,
                 options.parameter_file);
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
    const auto config = model_config(options.model);
    microllm::model::TransformerModel model(config, options.seed);
    model.to(communicator.device());
    microllm::training::AdamW optimizer(model.parameters(), optimizer_config());
    const auto parameters = model.parameters();
    microllm::training::ExperimentState experiment{
        .global_step = 0,
        .data_cursor = 0,
        .rng_state = "seed=" + std::to_string(options.seed),
        .model_config = config.summary(),
        .data_config = "ranked-synthetic:context=" +
                       std::to_string(options.context) + ":world=2"};
    CheckpointReport checkpoint_report{
        .checkpoint_requested = !options.checkpoint_file.empty(),
        .checkpoint_file = options.checkpoint_file,
        .checkpoint_ready_file = options.checkpoint_ready_file};
    if (!options.resume_file.empty()) {
        const auto resume_begin = SteadyClock::now();
        const auto loaded =
            microllm::training::load_checkpoint(options.resume_file);
        if (loaded.experiment.model_config != experiment.model_config ||
            loaded.experiment.data_config != experiment.data_config ||
            loaded.experiment.rng_state != experiment.rng_state) {
            throw std::invalid_argument(
                "rank checkpoint experiment contract changed");
        }
        microllm::training::restore_checkpoint(
            loaded, named_parameters(model), optimizer, experiment);
        checkpoint_report.resume_ms = elapsed_ms(
            resume_begin, SteadyClock::now());
        checkpoint_report.resumed = true;
    }
    checkpoint_report.initial_step = experiment.global_step;
    const auto batch = local_batch(config, options.rank, options.context);
    std::vector<float> losses;
    ReducerStats reducer_stats;
    microllm::multi_gpu::RankGradientBucketPlan persistent_bucket_plan;
    if (options.reducer == "overlap-views") {
        for (std::size_t index = 0; index < parameters.size(); ++index) {
            parameters[index]->set_gradient_ready_hook(
                [&persistent_bucket_plan, index] {
                    if (persistent_bucket_plan.overlap_active()) {
                        persistent_bucket_plan.mark_parameter_ready(index);
                    }
                });
        }
    }
    PhaseTimings timings;
    microllm::runtime::synchronize(communicator.device());
    const auto training_begin = SteadyClock::now();
    for (std::uint64_t step = 0; step < options.steps; ++step) {
        ++experiment.global_step;
        experiment.data_cursor += options.context;
        const auto step_begin = SteadyClock::now();
        const auto forward_backward_begin = SteadyClock::now();
        optimizer.zero_grad();
        const auto overlap_step =
            options.reducer == "overlap-views" &&
            persistent_bucket_plan.initialized();
        if (overlap_step) {
            persistent_bucket_plan.begin_overlap_step(
                communicator, parameters);
        }
        const auto loss = model.loss(batch.inputs, batch.targets);
        losses.push_back(loss.data().to_vector()[0]);
        loss.backward();
        if (!overlap_step) {
            microllm::runtime::synchronize(communicator.device());
        }
        const auto forward_backward_elapsed = elapsed_ms(
            forward_backward_begin, SteadyClock::now());
        timings.forward_backward_ms += forward_backward_elapsed;
        timings.step_forward_backward_ms.push_back(forward_backward_elapsed);
        const auto allocation_before =
            microllm::runtime::allocation_stats(communicator.device());
        const auto reducer_begin = SteadyClock::now();
        std::size_t step_collectives = 0;
        std::size_t step_buckets = 0;
        std::size_t step_pack_copies = 0;
        std::size_t step_unpack_copies = 0;
        std::size_t step_gradient_views = 0;
        std::size_t step_plan_reused = 0;
        std::size_t step_overlap_enabled = 0;
        std::size_t step_overlapped_buckets = 0;
        if (overlap_step) {
            const auto buckets =
                persistent_bucket_plan.finish_overlap_step();
            reducer_stats.collectives += buckets.bucket_count;
            reducer_stats.buckets += buckets.bucket_count;
            reducer_stats.pack_copies += buckets.pack_copy_calls;
            reducer_stats.unpack_copies += buckets.unpack_copy_calls;
            reducer_stats.gradient_views += buckets.gradient_view_count;
            step_collectives = buckets.bucket_count;
            step_buckets = buckets.bucket_count;
            step_pack_copies = buckets.pack_copy_calls;
            step_unpack_copies = buckets.unpack_copy_calls;
            step_gradient_views = buckets.gradient_view_count;
            step_plan_reused = buckets.plan_reused ? 1U : 0U;
            step_overlap_enabled = buckets.overlap_enabled ? 1U : 0U;
            step_overlapped_buckets = buckets.overlapped_bucket_count;
            reducer_stats.persistent_storage = buckets.persistent_storage;
            reducer_stats.plan_reuses += step_plan_reused;
            reducer_stats.plan_capacity_elements =
                buckets.plan_capacity_elements;
            reducer_stats.plan_capacity_bytes = buckets.plan_capacity_bytes;
            reducer_stats.overlap_steps += step_overlap_enabled;
            reducer_stats.overlapped_buckets += step_overlapped_buckets;
        } else if (options.reducer == "bucket" ||
            options.reducer == "persistent-bucket" ||
            options.reducer == "bucket-views" ||
            options.reducer == "overlap-views") {
            auto* plan = options.reducer != "bucket"
                             ? &persistent_bucket_plan
                             : nullptr;
            const auto gradient_views =
                options.reducer == "bucket-views" ||
                options.reducer == "overlap-views";
            const auto buckets = microllm::multi_gpu::all_reduce_rank_gradients(
                communicator, parameters, options.bucket_bytes, plan,
                gradient_views);
            reducer_stats.collectives += buckets.bucket_count;
            reducer_stats.buckets += buckets.bucket_count;
            reducer_stats.pack_copies += buckets.pack_copy_calls;
            reducer_stats.unpack_copies += buckets.unpack_copy_calls;
            reducer_stats.gradient_views += buckets.gradient_view_count;
            step_collectives = buckets.bucket_count;
            step_buckets = buckets.bucket_count;
            step_pack_copies = buckets.pack_copy_calls;
            step_unpack_copies = buckets.unpack_copy_calls;
            step_gradient_views = buckets.gradient_view_count;
            step_plan_reused = buckets.plan_reused ? 1U : 0U;
            reducer_stats.persistent_storage = buckets.persistent_storage;
            reducer_stats.plan_reuses += step_plan_reused;
            reducer_stats.plan_capacity_elements =
                buckets.plan_capacity_elements;
            reducer_stats.plan_capacity_bytes = buckets.plan_capacity_bytes;
        } else {
            for (auto* parameter : parameters) {
                auto gradient = parameter->grad();
                communicator.enqueue_all_reduce_average_in_place(gradient);
                ++reducer_stats.collectives;
                ++step_collectives;
            }
            communicator.synchronize();
        }
        const auto reducer_elapsed = elapsed_ms(
            reducer_begin, SteadyClock::now());
        timings.reducer_ms += reducer_elapsed;
        timings.step_reducer_ms.push_back(reducer_elapsed);
        const auto allocation_after =
            microllm::runtime::allocation_stats(communicator.device());
        reducer_stats.step_collectives.push_back(step_collectives);
        reducer_stats.step_buckets.push_back(step_buckets);
        reducer_stats.step_pack_copies.push_back(step_pack_copies);
        reducer_stats.step_unpack_copies.push_back(step_unpack_copies);
        reducer_stats.step_gradient_views.push_back(step_gradient_views);
        reducer_stats.step_allocation_calls.push_back(counter_delta(
            allocation_after.allocation_calls,
            allocation_before.allocation_calls));
        reducer_stats.step_backend_allocation_calls.push_back(counter_delta(
            allocation_after.backend_allocation_calls,
            allocation_before.backend_allocation_calls));
        reducer_stats.step_deallocation_calls.push_back(counter_delta(
            allocation_after.deallocation_calls,
            allocation_before.deallocation_calls));
        reducer_stats.step_total_allocated_bytes.push_back(counter_delta(
            allocation_after.total_allocated_bytes,
            allocation_before.total_allocated_bytes));
        reducer_stats.step_plan_reused.push_back(step_plan_reused);
        reducer_stats.step_current_bytes_before.push_back(
            allocation_before.current_bytes);
        reducer_stats.step_current_bytes_after.push_back(
            allocation_after.current_bytes);
        reducer_stats.step_peak_bytes_after.push_back(
            allocation_after.peak_bytes);
        reducer_stats.step_overlap_enabled.push_back(step_overlap_enabled);
        reducer_stats.step_overlapped_buckets.push_back(
            step_overlapped_buckets);
        const auto optimizer_begin = SteadyClock::now();
        optimizer.step();
        microllm::runtime::synchronize(communicator.device());
        const auto optimizer_elapsed = elapsed_ms(
            optimizer_begin, SteadyClock::now());
        timings.optimizer_ms += optimizer_elapsed;
        timings.step_optimizer_ms.push_back(optimizer_elapsed);
        timings.step_training_ms.push_back(
            elapsed_ms(step_begin, SteadyClock::now()));
    }
    timings.training_ms = elapsed_ms(training_begin, SteadyClock::now());
    const auto allocation =
        microllm::runtime::allocation_stats(communicator.device());
    if (options.reducer == "overlap-views") {
        for (auto* parameter : parameters) {
            parameter->clear_gradient_ready_hook();
        }
    }
    checkpoint_report.final_step = experiment.global_step;
    checkpoint_report.optimizer_step = optimizer.step_count();
    if (checkpoint_report.optimizer_step != checkpoint_report.final_step) {
        throw std::runtime_error(
            "rank checkpoint optimizer and experiment steps diverged");
    }
    if (checkpoint_report.checkpoint_requested) {
        auto barrier = microllm::Tensor::from_vector({1.0F}, {1}).to(
            communicator.device());
        communicator.enqueue_all_reduce_average_in_place(barrier);
        communicator.synchronize();
        if (options.rank == 0) {
            if (options.inject_checkpoint_failure) {
                throw std::runtime_error("injected rank0 checkpoint failure");
            }
            const auto write_begin = SteadyClock::now();
            microllm::training::save_checkpoint(
                options.checkpoint_file, named_parameters(model), optimizer,
                experiment);
            publish_checkpoint_ready(
                options.checkpoint_ready_file, experiment.global_step);
            checkpoint_report.checkpoint_write_ms = elapsed_ms(
                write_begin, SteadyClock::now());
            checkpoint_report.checkpoint_written = true;
        } else {
            const auto wait_begin = SteadyClock::now();
            wait_for_checkpoint_ready(
                options.checkpoint_ready_file, experiment.global_step,
                options.timeout_ms);
            checkpoint_report.checkpoint_wait_ms = elapsed_ms(
                wait_begin, SteadyClock::now());
        }
        const auto verify_begin = SteadyClock::now();
        const auto published =
            microllm::training::load_checkpoint(options.checkpoint_file);
        if (published.experiment.global_step != experiment.global_step ||
            published.optimizer_state.step != optimizer.step_count() ||
            published.experiment.data_cursor != experiment.data_cursor) {
            throw std::runtime_error(
                "published rank checkpoint state changed");
        }
        checkpoint_report.checkpoint_verify_ms = elapsed_ms(
            verify_begin, SteadyClock::now());
        checkpoint_report.checkpoint_verified = true;
    }
    if (!options.parameter_file.empty()) {
        microllm::io::save_safetensors(
            options.parameter_file, model.state_dict());
    }
    write_result("rank", options.rank, losses, model,
                 options.reducer, reducer_stats, timings, allocation,
                 checkpoint_report, options.model, options.context,
                 options.parameter_file);
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
