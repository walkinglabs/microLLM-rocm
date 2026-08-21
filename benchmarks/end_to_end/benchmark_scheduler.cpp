#include <chrono>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include <microllm/inference/generator.h>
#include <microllm/inference/scheduler.h>
#include <microllm/model/model.h>
#include <microllm/runtime/memory.h>
#include <microllm/runtime/runtime.h>

namespace {

struct Options {
    std::string device = "cpu";
    std::int64_t requests = 4;
    int warmup = 1;
    int repetitions = 3;
};

Options options(int argc, char** argv) {
    Options result;
    for (int index = 1; index < argc; index += 2) {
        if (index + 1 >= argc) throw std::invalid_argument("missing CLI value");
        const std::string name = argv[index];
        if (name == "--device") result.device = argv[index + 1];
        else if (name == "--requests") result.requests = std::stoll(argv[index + 1]);
        else if (name == "--warmup") result.warmup = std::stoi(argv[index + 1]);
        else if (name == "--repetitions") result.repetitions = std::stoi(argv[index + 1]);
        else throw std::invalid_argument("unknown option: " + name);
    }
    if ((result.device != "cpu" && result.device != "hip") || result.requests <= 0 ||
        result.warmup < 0 || result.repetitions <= 0) {
        throw std::invalid_argument("invalid scheduler benchmark options");
    }
    return result;
}

microllm::model::ModelConfig config() {
    return {.vocabulary_size = 256,
            .dimension = 64,
            .layers = 2,
            .heads = 4,
            .kv_heads = 2,
            .ffn_dimension = 128,
            .max_sequence_length = 64,
            .rope_base = 10000.0F,
            .tie_embeddings = false};
}

struct WorkItem {
    std::vector<std::int32_t> prompt;
    microllm::inference::GenerationConfig generation;
};

std::vector<WorkItem> workload(std::int64_t count) {
    std::vector<WorkItem> result;
    result.reserve(static_cast<std::size_t>(count));
    for (std::int64_t request = 0; request < count; ++request) {
        const auto length = 4 + (request % 4) * 4;
        std::vector<std::int32_t> prompt(static_cast<std::size_t>(length));
        for (std::int64_t token = 0; token < length; ++token) {
            prompt[static_cast<std::size_t>(token)] =
                static_cast<std::int32_t>((request * 17 + token * 7 + 1) % 256);
        }
        result.push_back({std::move(prompt),
                          {.max_new_tokens = 3 + request % 3,
                           .temperature = 0.0F,
                           .top_k = 1,
                           .seed = static_cast<std::uint64_t>(request + 1),
                           .kv_cache_layer_dtypes = {}}});
    }
    return result;
}

struct Run {
    std::vector<std::vector<std::int32_t>> generated;
    microllm::inference::SchedulerMetrics metrics;
};

Run run_scheduler(microllm::model::TransformerModel& model,
                  const std::vector<WorkItem>& work) {
    microllm::inference::ReferenceScheduler scheduler(model);
    std::vector<microllm::inference::RequestId> ids;
    ids.reserve(work.size());
    const auto initial = (work.size() + 1U) / 2U;
    for (std::size_t index = 0; index < initial; ++index) {
        ids.push_back(scheduler.submit(work[index].prompt, work[index].generation));
    }
    scheduler.step();
    for (std::size_t index = initial; index < work.size(); ++index) {
        ids.push_back(scheduler.submit(work[index].prompt, work[index].generation));
    }
    scheduler.run_until_idle();
    Run result;
    for (const auto id : ids) result.generated.push_back(scheduler.request(id).generated);
    result.metrics = scheduler.metrics();
    return result;
}

std::vector<std::vector<std::int32_t>> run_sequential(
    microllm::model::TransformerModel& model,
    const std::vector<WorkItem>& work) {
    std::vector<std::vector<std::int32_t>> result;
    for (const auto& item : work) {
        const auto tokens = microllm::inference::generate(
            model, item.prompt, item.generation);
        result.emplace_back(tokens.begin() + static_cast<std::ptrdiff_t>(item.prompt.size()),
                            tokens.end());
    }
    return result;
}

void synchronize(microllm::Device device) {
    if (device.is_hip()) microllm::runtime::synchronize(device);
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const auto command = options(argc, argv);
        const auto device = command.device == "hip" ? microllm::Device::hip()
                                                      : microllm::Device::cpu();
        if (device.is_hip() && microllm::runtime::hip_device_count() == 0) {
            throw std::runtime_error("HIP benchmark requested without a visible device");
        }
        auto scheduled_model = microllm::model::TransformerModel(config(), 101);
        auto sequential_model = microllm::model::TransformerModel(config(), 101);
        scheduled_model.to(device);
        sequential_model.to(device);
        const auto work = workload(command.requests);
        for (int iteration = 0; iteration < command.warmup; ++iteration) {
            (void)run_scheduler(scheduled_model, work);
            (void)run_sequential(sequential_model, work);
        }
        synchronize(device);
        microllm::runtime::reset_allocation_peak(device);
        double scheduler_ms = 0.0;
        double sequential_ms = 0.0;
        Run last_scheduler;
        std::vector<std::vector<std::int32_t>> last_sequential;
        for (int iteration = 0; iteration < command.repetitions; ++iteration) {
            auto start = std::chrono::steady_clock::now();
            last_scheduler = run_scheduler(scheduled_model, work);
            synchronize(device);
            auto finish = std::chrono::steady_clock::now();
            scheduler_ms += std::chrono::duration<double, std::milli>(finish - start).count();
            start = std::chrono::steady_clock::now();
            last_sequential = run_sequential(sequential_model, work);
            synchronize(device);
            finish = std::chrono::steady_clock::now();
            sequential_ms += std::chrono::duration<double, std::milli>(finish - start).count();
        }
        if (last_scheduler.generated != last_sequential) {
            throw std::runtime_error("scheduler output differs from sequential generate");
        }
        std::int64_t generated_per_repetition = 0;
        std::uint64_t checksum = 0;
        for (const auto& item : work) generated_per_repetition += item.generation.max_new_tokens;
        for (const auto& request : last_scheduler.generated) {
            for (const auto token : request) checksum = checksum * 131U + token;
        }
        const auto allocation = microllm::runtime::allocation_stats(device);
        const auto measured_tokens = generated_per_repetition * command.repetitions;
        std::cout << "{\"schema_version\":1,\"status\":\"pass\""
                  << ",\"scheduler\":\"serial_reference\""
                  << ",\"device\":\"" << device.str() << "\""
                  << ",\"requests\":" << command.requests
                  << ",\"parameter_count\":" << scheduled_model.parameter_count()
                  << ",\"model_dimension\":" << config().dimension
                  << ",\"model_layers\":" << config().layers
                  << ",\"warmup\":" << command.warmup
                  << ",\"repetitions\":" << command.repetitions
                  << ",\"measured_tokens\":" << measured_tokens
                  << ",\"scheduler_ms\":" << scheduler_ms
                  << ",\"sequential_ms\":" << sequential_ms
                  << ",\"scheduler_tokens_per_second\":"
                  << static_cast<double>(measured_tokens) * 1000.0 / scheduler_ms
                  << ",\"sequential_tokens_per_second\":"
                  << static_cast<double>(measured_tokens) * 1000.0 / sequential_ms
                  << ",\"scheduler_over_sequential\":"
                  << sequential_ms / scheduler_ms
                  << ",\"scheduler_steps\":" << last_scheduler.metrics.scheduler_steps
                  << ",\"prefill_calls\":" << last_scheduler.metrics.prefill_calls
                  << ",\"decode_calls\":" << last_scheduler.metrics.decode_calls
                  << ",\"peak_active_requests\":"
                  << last_scheduler.metrics.peak_active_requests
                  << ",\"peak_cache_bytes\":" << last_scheduler.metrics.peak_cache_bytes
                  << ",\"engine_peak_bytes\":" << allocation.peak_bytes
                  << ",\"outputs_equal\":true"
                  << ",\"token_checksum\":" << checksum << "}\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
