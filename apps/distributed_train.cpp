#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include <microllm/multi_gpu/data_parallel.h>
#include <microllm/profiling/trace.h>
#include <microllm/runtime/memory.h>

namespace {

struct Options {
    std::string model = "tiny";
    std::uint64_t steps = 3;
    std::size_t bucket_bytes = 4U * 1024U * 1024U;
    std::size_t parameter_check_interval = 1;
    bool in_place_bucket_average = true;
    bool persistent_gradient_buckets = false;
    bool gradient_bucket_views = false;
    bool record_gradient_ready_order = false;
    bool overlap_gradient_communication = false;
    std::uint64_t seed = 601;
    std::size_t batch = 1;
    std::size_t context = 0;
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
        if (argument == "--model") options.model = next("--model");
        else if (argument == "--steps") options.steps = number(next("--steps"), "steps");
        else if (argument == "--bucket-bytes") {
            options.bucket_bytes = static_cast<std::size_t>(
                number(next("--bucket-bytes"), "bucket bytes"));
        } else if (argument == "--parameter-check-interval") {
            options.parameter_check_interval = static_cast<std::size_t>(
                number(next("--parameter-check-interval"),
                       "parameter check interval"));
        } else if (argument == "--inplace-bucket-average") {
            const auto value = next("--inplace-bucket-average");
            if (value != "true" && value != "false") {
                throw std::invalid_argument(
                    "--inplace-bucket-average must be true or false");
            }
            options.in_place_bucket_average = value == "true";
        } else if (argument == "--persistent-gradient-buckets") {
            const auto value = next("--persistent-gradient-buckets");
            if (value != "true" && value != "false") {
                throw std::invalid_argument(
                    "--persistent-gradient-buckets must be true or false");
            }
            options.persistent_gradient_buckets = value == "true";
        } else if (argument == "--gradient-bucket-views") {
            const auto value = next("--gradient-bucket-views");
            if (value != "true" && value != "false") {
                throw std::invalid_argument(
                    "--gradient-bucket-views must be true or false");
            }
            options.gradient_bucket_views = value == "true";
        } else if (argument == "--record-gradient-ready-order") {
            const auto value = next("--record-gradient-ready-order");
            if (value != "true" && value != "false") {
                throw std::invalid_argument(
                    "--record-gradient-ready-order must be true or false");
            }
            options.record_gradient_ready_order = value == "true";
        } else if (argument == "--overlap-gradient-communication") {
            const auto value = next("--overlap-gradient-communication");
            if (value != "true" && value != "false") {
                throw std::invalid_argument(
                    "--overlap-gradient-communication must be true or false");
            }
            options.overlap_gradient_communication = value == "true";
        } else if (argument == "--seed") options.seed = number(next("--seed"), "seed");
        else if (argument == "--batch") {
            options.batch = static_cast<std::size_t>(number(next("--batch"), "batch"));
        } else if (argument == "--context") {
            options.context = static_cast<std::size_t>(number(next("--context"), "context"));
        }
        else if (argument == "--trace") options.trace = next("--trace");
        else throw std::invalid_argument("unknown argument: " + argument);
    }
    if ((options.model != "tiny" && options.model != "model-s") ||
        options.steps == 0 || options.bucket_bytes < sizeof(float) ||
        options.batch == 0) {
        throw std::invalid_argument("steps and bucket size must be positive");
    }
    if (options.context == 0) options.context = options.model == "tiny" ? 4 : 32;
    return options;
}

microllm::model::ModelConfig config(const std::string& model) {
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

std::vector<microllm::io::TokenBatch> batches(
    const microllm::model::ModelConfig& model, std::size_t batch,
    std::size_t context) {
    if (context == 0 || context > static_cast<std::size_t>(model.max_sequence_length)) {
        throw std::invalid_argument("context is outside the selected model capacity");
    }
    if (model.vocabulary_size == 8 && batch == 1 && context == 4) {
        return {
            {microllm::Tensor::from_int32_vector({0, 1, 2, 3}, {1, 4}),
             microllm::Tensor::from_int32_vector({1, 2, 3, 0}, {1, 4})},
            {microllm::Tensor::from_int32_vector({3, 2, 1, 0}, {1, 4}),
             microllm::Tensor::from_int32_vector({2, 1, 0, 3}, {1, 4})},
        };
    }
    std::vector<microllm::io::TokenBatch> result;
    result.reserve(2);
    for (std::size_t rank = 0; rank < 2; ++rank) {
        std::vector<std::int32_t> inputs(batch * context);
        std::vector<std::int32_t> targets(inputs.size());
        for (std::size_t index = 0; index < inputs.size(); ++index) {
            const auto token = static_cast<std::int32_t>(
                (rank * 97 + index * 13 + 7) %
                static_cast<std::size_t>(model.vocabulary_size));
            inputs[index] = token;
            targets[index] = (token + 1) %
                             static_cast<std::int32_t>(model.vocabulary_size);
        }
        const microllm::Shape shape{
            static_cast<std::int64_t>(batch),
            static_cast<std::int64_t>(context)};
        result.push_back({microllm::Tensor::from_int32_vector(inputs, shape),
                          microllm::Tensor::from_int32_vector(targets, shape)});
    }
    return result;
}

void write_indices(const std::vector<std::size_t>& values) {
    std::cout << '[';
    for (std::size_t index = 0; index < values.size(); ++index) {
        if (index != 0) std::cout << ',';
        std::cout << values[index];
    }
    std::cout << ']';
}

void write_strings(const std::vector<std::string>& values) {
    std::cout << '[';
    for (std::size_t index = 0; index < values.size(); ++index) {
        if (index != 0) std::cout << ',';
        std::cout << '\"' << values[index] << '\"';
    }
    std::cout << ']';
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const auto options = parse(argc, argv);
        const auto model_config = config(options.model);
        const auto rank_batches = batches(
            model_config, options.batch, options.context);
        const microllm::training::AdamWConfig optimizer{
            .learning_rate = 0.005F,
            .beta1 = 0.9F,
            .beta2 = 0.99F,
            .epsilon = 1.0e-8F,
            .weight_decay = 0.0F,
        };
        microllm::multi_gpu::DataParallelTrainer trainer(
            model_config, options.seed,
            {.device_indices = {0, 1},
             .maximum_bucket_bytes = options.bucket_bytes,
             .parameter_check_interval = options.parameter_check_interval,
             .in_place_bucket_average = options.in_place_bucket_average,
             .persistent_gradient_buckets = options.persistent_gradient_buckets,
             .gradient_bucket_views = options.gradient_bucket_views,
             .record_gradient_ready_order = options.record_gradient_ready_order,
             .overlap_gradient_communication = options.overlap_gradient_communication,
             .optimizer = optimizer});
        const auto parameter_count = trainer.model(0).parameter_count();
        std::vector<std::string> parameter_names;
        std::vector<std::size_t> parameter_elements;
        for (const auto& [name, parameter] : trainer.model(0).named_parameters()) {
            parameter_names.push_back(name);
            parameter_elements.push_back(
                static_cast<std::size_t>(parameter->data().numel()));
        }
        for (const auto device : {0, 1}) {
            microllm::runtime::reset_allocation_peak(microllm::Device::hip(device));
        }
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
                const auto metrics = trainer.step(rank_batches, step);
                if (!std::isfinite(metrics.mean_loss) ||
                    (metrics.parameter_check_performed &&
                     metrics.maximum_parameter_difference != 0.0F)) {
                    throw std::runtime_error(
                        "distributed step failed its loss or rank-parameter gate");
                }
                std::size_t maximum_peak_bytes = 0;
                std::size_t maximum_current_bytes = 0;
                for (const auto device : {0, 1}) {
                    const auto allocation = microllm::runtime::allocation_stats(
                        microllm::Device::hip(device));
                    maximum_peak_bytes = std::max(
                        maximum_peak_bytes, allocation.peak_bytes);
                    maximum_current_bytes = std::max(
                        maximum_current_bytes, allocation.current_bytes);
                }
                std::cout << "{\"step\":" << metrics.step
                          << ",\"model\":\"" << options.model << "\""
                          << ",\"batch\":" << options.batch
                          << ",\"context\":" << options.context
                          << ",\"parameter_count\":" << parameter_count
                          << ",\"inplace_bucket_average\":"
                          << (options.in_place_bucket_average ? "true" : "false")
                          << ",\"persistent_gradient_buckets\":"
                          << (options.persistent_gradient_buckets ? "true" : "false")
                          << ",\"gradient_bucket_views\":"
                          << (options.gradient_bucket_views ? "true" : "false")
                          << ",\"record_gradient_ready_order\":"
                          << (options.record_gradient_ready_order ? "true" : "false")
                          << ",\"overlap_gradient_communication\":"
                          << (options.overlap_gradient_communication ? "true" : "false")
                          << ",\"mean_loss\":" << metrics.mean_loss
                          << ",\"bucket_count\":" << metrics.buckets.bucket_count
                          << ",\"bucket_parameter_count\":"
                          << metrics.buckets.parameter_count
                          << ",\"bucket_total_elements\":"
                          << metrics.buckets.total_elements
                          << ",\"bucket_tensor_count\":"
                          << metrics.buckets.bucket_tensor_count
                          << ",\"average_tensor_count\":"
                          << metrics.buckets.average_tensor_count
                          << ",\"unpacked_tensor_count\":"
                          << metrics.buckets.unpacked_tensor_count
                          << ",\"gradient_view_count\":"
                          << metrics.buckets.gradient_view_count
                          << ",\"pack_copy_calls\":"
                          << metrics.buckets.pack_copy_calls
                          << ",\"unpack_copy_calls\":"
                          << metrics.buckets.unpack_copy_calls
                          << ",\"bucket_temporary_elements\":"
                          << metrics.buckets.temporary_elements
                          << ",\"bucket_temporary_bytes\":"
                          << metrics.buckets.temporary_bytes
                          << ",\"bucket_persistent_storage\":"
                          << (metrics.buckets.persistent_storage ? "true" : "false")
                          << ",\"bucket_plan_reused\":"
                          << (metrics.buckets.plan_reused ? "true" : "false")
                          << ",\"bucket_overlap_enabled\":"
                          << (metrics.buckets.overlap_enabled ? "true" : "false")
                          << ",\"overlapped_bucket_count\":"
                          << metrics.buckets.overlapped_bucket_count
                          << ",\"bucket_plan_capacity_elements\":"
                          << metrics.buckets.plan_capacity_elements
                          << ",\"bucket_plan_capacity_bytes\":"
                          << metrics.buckets.plan_capacity_bytes
                          << ",\"parameter_check_performed\":"
                          << (metrics.parameter_check_performed ? "true" : "false")
                          << ",\"parameter_max_difference\":"
                          << metrics.maximum_parameter_difference
                          << ",\"forward_backward_ms\":" << metrics.forward_backward_ms
                          << ",\"communication_ms\":" << metrics.communication_ms
                          << ",\"communication_allocation_calls\":"
                          << metrics.communication_allocation_calls
                          << ",\"communication_backend_allocation_calls\":"
                          << metrics.communication_backend_allocation_calls
                          << ",\"communication_cache_reuse_calls\":"
                          << metrics.communication_cache_reuse_calls
                          << ",\"communication_total_allocated_bytes\":"
                          << metrics.communication_total_allocated_bytes
                          << ",\"optimizer_ms\":" << metrics.optimizer_ms
                          << ",\"verification_ms\":" << metrics.verification_ms
                          << ",\"maximum_engine_peak_bytes\":"
                          << maximum_peak_bytes
                          << ",\"maximum_engine_current_bytes\":"
                          << maximum_current_bytes
                          << ",\"gradient_ready_audit_performed\":"
                          << (metrics.gradient_ready_audit_performed ? "true" : "false")
                          << ",\"gradient_ready_orders_match\":"
                          << (metrics.gradient_ready_orders_match ? "true" : "false")
                          << ",\"overlap_communication_performed\":"
                          << (metrics.overlap_communication_performed ? "true" : "false")
                          << ",\"overlap_finish_ms\":"
                          << metrics.overlap_finish_ms
                          << ",\"parameter_names\":";
                write_strings(parameter_names);
                std::cout << ",\"parameter_elements\":";
                write_indices(parameter_elements);
                for (std::size_t rank = 0;
                     rank < metrics.rank_gradient_ready_order.size(); ++rank) {
                    std::cout << ",\"gradient_ready_order_rank" << rank << "\":";
                    write_indices(metrics.rank_gradient_ready_order[rank]);
                }
                std::cout << ",\"total_ms\":" << metrics.total_ms << "}\n";
            }
        }
        if (!options.trace.empty()) trace.write_jsonl(options.trace);
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "microllm_distributed_train: " << error.what() << '\n';
        return 1;
    }
}
