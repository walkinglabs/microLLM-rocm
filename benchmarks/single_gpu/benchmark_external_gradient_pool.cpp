#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <map>
#include <numeric>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <microllm/model/model.h>
#include <microllm/runtime/memory.h>
#include <microllm/runtime/runtime.h>
#include <microllm/training/optimizer.h>

namespace {

using Clock = std::chrono::steady_clock;

struct Options {
    std::string model = "tiny";
    std::string first = "baseline";
    int warmup = 1;
    int repetitions = 3;
    std::int64_t context = 8;
};

Options parse_options(int argc, char** argv) {
    Options result;
    for (int index = 1; index < argc; index += 2) {
        if (index + 1 >= argc) throw std::invalid_argument("missing option value");
        const std::string_view name(argv[index]);
        const std::string value(argv[index + 1]);
        if (name == "--model") result.model = value;
        else if (name == "--first") result.first = value;
        else if (name == "--warmup") result.warmup = std::stoi(value);
        else if (name == "--repetitions") result.repetitions = std::stoi(value);
        else if (name == "--context") result.context = std::stoll(value);
        else throw std::invalid_argument("unknown option: " + std::string(name));
    }
    if ((result.model != "tiny" && result.model != "model-s") ||
        (result.first != "baseline" && result.first != "external") ||
        result.warmup < 1 || result.repetitions < 1 || result.context < 2) {
        throw std::invalid_argument("external gradient pool options are invalid");
    }
    return result;
}

microllm::model::ModelConfig model_config(const Options& options) {
    if (options.model == "model-s") {
        auto config = microllm::model::ModelConfig::model_s();
        if (options.context > config.max_sequence_length) {
            throw std::invalid_argument("context exceeds Model-S maximum");
        }
        return config;
    }
    if (options.context > 32) {
        throw std::invalid_argument("context exceeds tiny maximum");
    }
    return {.vocabulary_size = 64,
            .dimension = 32,
            .layers = 2,
            .heads = 4,
            .kv_heads = 2,
            .ffn_dimension = 64,
            .max_sequence_length = 32,
            .rope_base = 10000.0F,
            .tie_embeddings = false};
}

std::vector<std::int32_t> token_values(std::int64_t context,
                                       std::int64_t vocabulary,
                                       std::int64_t offset) {
    std::vector<std::int32_t> result;
    result.reserve(static_cast<std::size_t>(context));
    for (std::int64_t position = 0; position < context; ++position) {
        result.push_back(static_cast<std::int32_t>(
            (offset + position * 17) % vocabulary));
    }
    return result;
}

double median(std::vector<double> values) {
    if (values.empty()) throw std::logic_error("median requires values");
    std::sort(values.begin(), values.end());
    const auto middle = values.size() / 2;
    return values.size() % 2 == 0
               ? (values[middle - 1] + values[middle]) / 2.0
               : values[middle];
}

struct Metrics {
    std::string policy;
    float loss = 0.0F;
    std::vector<double> event_ms;
    std::vector<double> wall_ms;
    std::size_t parameter_tensors = 0;
    std::uint64_t gradient_elements = 0;
    std::uint64_t pool_bytes = 0;
    std::size_t stable_addresses = 0;
    std::size_t allocation_calls = 0;
    std::size_t backend_allocation_calls = 0;
    std::size_t cache_reuse_calls = 0;
    std::size_t peak_extra_bytes = 0;
    std::map<std::string, std::vector<float>> gradients;
};

Metrics run_policy(const Options& options, const std::string& policy,
                   const microllm::Tensor& inputs,
                   const microllm::Tensor& targets) {
    const auto device = microllm::Device::hip(0);
    microllm::model::TransformerModel model(model_config(options), 1701);
    model.to(device);
    auto named = model.named_parameters();

    Metrics result;
    result.policy = policy;
    result.parameter_tensors = named.size();
    for (const auto& [name, parameter] : named) {
        (void)name;
        result.gradient_elements +=
            static_cast<std::uint64_t>(parameter->data().numel());
    }

    microllm::Tensor pool;
    std::map<std::string, const void*> expected_addresses;
    if (policy == "external") {
        pool = microllm::Tensor(
            {static_cast<std::int64_t>(result.gradient_elements)},
            microllm::DType::Float32, device);
        result.pool_bytes = pool.storage().num_bytes();
        auto* cursor = static_cast<std::byte*>(pool.storage().data());
        for (const auto& [name, parameter] : named) {
            const auto bytes = static_cast<std::size_t>(
                parameter->data().numel()) * sizeof(float);
            auto storage = microllm::Storage::from_external(cursor, bytes, device);
            auto view = microllm::Tensor::from_storage(
                std::move(storage), parameter->data().shape(),
                microllm::contiguous_strides(parameter->data().shape()), 0,
                microllm::DType::Float32);
            expected_addresses.emplace(name, cursor);
            parameter->bind_grad_buffer(std::move(view));
            cursor += bytes;
        }
    }

    const auto execute = [&]() {
        microllm::training::zero_grad(model.parameters());
        auto loss = model.loss(inputs, targets);
        loss.backward();
        return loss;
    };

    for (int iteration = 0; iteration < options.warmup; ++iteration) {
        auto loss = execute();
        (void)loss;
        microllm::runtime::synchronize(device);
    }
    microllm::runtime::enable_hip_caching_allocator(device);
    microllm::runtime::reset_allocation_peak(device);
    const auto before = microllm::runtime::allocation_stats(device);

    for (int iteration = 0; iteration < options.repetitions; ++iteration) {
        microllm::runtime::Event start(device);
        microllm::runtime::Event finish(device);
        const auto wall_start = Clock::now();
        start.record_default_stream();
        auto loss = execute();
        finish.record_default_stream();
        finish.synchronize();
        const auto wall_finish = Clock::now();
        result.event_ms.push_back(finish.elapsed_ms_since(start));
        result.wall_ms.push_back(std::chrono::duration<double, std::milli>(
                                     wall_finish - wall_start)
                                     .count());
        result.loss = loss.data().to_vector().front();
    }

    const auto after = microllm::runtime::allocation_stats(device);
    result.allocation_calls = after.allocation_calls - before.allocation_calls;
    result.backend_allocation_calls =
        after.backend_allocation_calls - before.backend_allocation_calls;
    result.cache_reuse_calls = after.cache_reuse_calls - before.cache_reuse_calls;
    result.peak_extra_bytes = after.peak_bytes >= before.current_bytes
                                  ? after.peak_bytes - before.current_bytes
                                  : 0;

    for (const auto& [name, parameter] : named) {
        if (!parameter->has_grad()) {
            throw std::runtime_error("missing gradient for " + name);
        }
        if (policy == "external" &&
            parameter->grad().storage().data() == expected_addresses.at(name)) {
            ++result.stable_addresses;
        }
        result.gradients.emplace(name, parameter->grad().to_vector());
    }
    return result;
}

void write_metric(const Metrics& metric) {
    std::cout << "{\"policy\":\"" << metric.policy << "\""
              << ",\"loss\":" << metric.loss
              << ",\"event_median_ms\":" << median(metric.event_ms)
              << ",\"wall_median_ms\":" << median(metric.wall_ms)
              << ",\"parameter_tensors\":" << metric.parameter_tensors
              << ",\"gradient_elements\":" << metric.gradient_elements
              << ",\"pool_bytes\":" << metric.pool_bytes
              << ",\"stable_addresses\":" << metric.stable_addresses
              << ",\"allocation_calls\":" << metric.allocation_calls
              << ",\"backend_allocation_calls\":"
              << metric.backend_allocation_calls
              << ",\"cache_reuse_calls\":" << metric.cache_reuse_calls
              << ",\"peak_extra_bytes\":" << metric.peak_extra_bytes << '}';
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const auto options = parse_options(argc, argv);
        if (!microllm::runtime::hip_compiled() ||
            microllm::runtime::hip_device_count() == 0) {
            throw std::runtime_error("external gradient pool benchmark requires HIP");
        }
        const auto config = model_config(options);
        const auto device = microllm::Device::hip(0);
        const auto inputs = microllm::Tensor::from_int32_vector(
                                token_values(options.context,
                                             config.vocabulary_size, 3),
                                {1, options.context})
                                .to(device);
        const auto targets = microllm::Tensor::from_int32_vector(
                                 token_values(options.context,
                                              config.vocabulary_size, 5),
                                 {1, options.context})
                                 .to(device);

        const auto second = options.first == "baseline" ? "external" : "baseline";
        auto first_metrics = run_policy(options, options.first, inputs, targets);
        auto second_metrics = run_policy(options, second, inputs, targets);
        const auto& baseline = options.first == "baseline" ? first_metrics : second_metrics;
        const auto& external = options.first == "external" ? first_metrics : second_metrics;

        if (baseline.gradients.size() != external.gradients.size()) {
            throw std::runtime_error("policy parameter counts differ");
        }
        double squared_error = 0.0;
        double maximum_error = 0.0;
        std::uint64_t compared = 0;
        std::size_t mismatched_tensors = 0;
        for (const auto& [name, expected] : baseline.gradients) {
            const auto found = external.gradients.find(name);
            if (found == external.gradients.end() ||
                found->second.size() != expected.size()) {
                throw std::runtime_error("missing external gradient for " + name);
            }
            double tensor_maximum = 0.0;
            for (std::size_t index = 0; index < expected.size(); ++index) {
                const auto error = std::abs(static_cast<double>(
                    found->second[index] - expected[index]));
                maximum_error = std::max(maximum_error, error);
                tensor_maximum = std::max(tensor_maximum, error);
                squared_error += error * error;
                ++compared;
            }
            if (tensor_maximum != 0.0) ++mismatched_tensors;
        }
        const auto rms_error = compared == 0
                                   ? 0.0
                                   : std::sqrt(squared_error /
                                               static_cast<double>(compared));
        const auto all_addresses_stable =
            external.stable_addresses == external.parameter_tensors;
        const auto exact = maximum_error == 0.0 &&
                           baseline.loss == external.loss;
        const auto device_info = microllm::runtime::device_info(device);

        std::cout << std::setprecision(12)
                  << "{\"schema_version\":1,\"status\":\""
                  << (exact && all_addresses_stable ? "pass" : "fail") << "\""
                  << ",\"record_type\":\"external_gradient_pool\""
                  << ",\"model\":\"" << options.model << "\""
                  << ",\"first\":\"" << options.first << "\""
                  << ",\"context\":" << options.context
                  << ",\"warmup\":" << options.warmup
                  << ",\"repetitions\":" << options.repetitions
                  << ",\"device_name\":\"" << device_info.name << "\""
                  << ",\"architecture\":\"" << device_info.architecture << "\""
                  << ",\"hip_runtime_version\":"
                  << microllm::runtime::hip_runtime_version()
                  << ",\"hip_driver_version\":"
                  << microllm::runtime::hip_driver_version()
                  << ",\"maximum_gradient_error\":" << maximum_error
                  << ",\"rms_gradient_error\":" << rms_error
                  << ",\"mismatched_gradient_tensors\":"
                  << mismatched_tensors
                  << ",\"compared_gradient_elements\":" << compared
                  << ",\"all_external_addresses_stable\":"
                  << (all_addresses_stable ? "true" : "false")
                  << ",\"baseline\":";
        write_metric(baseline);
        std::cout << ",\"external\":";
        write_metric(external);
        std::cout << "}\n";
        return exact && all_addresses_stable ? 0 : 2;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
