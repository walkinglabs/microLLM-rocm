#include <algorithm>
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

#include <microllm/ops/ops.h>
#include <microllm/runtime/runtime.h>

namespace {

struct Options {
    std::int64_t batch = 1;
    std::int64_t heads = 14;
    std::int64_t kv_heads = 2;
    std::int64_t sequence = 512;
    std::int64_t width = 64;
    int warmup = 5;
    int repetitions = 20;
};

std::int64_t integer(const char* value, const char* name) {
    char* end = nullptr;
    const auto result = std::strtoll(value, &end, 10);
    if (end == value || *end != '\0') {
        throw std::invalid_argument(std::string("invalid ") + name);
    }
    return result;
}

Options parse(int argc, char** argv) {
    Options result;
    for (int index = 1; index < argc; index += 2) {
        if (index + 1 >= argc) {
            throw std::invalid_argument("option is missing a value");
        }
        const std::string_view name(argv[index]);
        if (name == "--batch") result.batch = integer(argv[index + 1], "batch");
        else if (name == "--heads") result.heads = integer(argv[index + 1], "heads");
        else if (name == "--kv-heads") {
            result.kv_heads = integer(argv[index + 1], "kv-heads");
        } else if (name == "--sequence") {
            result.sequence = integer(argv[index + 1], "sequence");
        } else if (name == "--width") {
            result.width = integer(argv[index + 1], "width");
        } else if (name == "--warmup") {
            result.warmup = static_cast<int>(integer(argv[index + 1], "warmup"));
        } else if (name == "--repetitions") {
            result.repetitions =
                static_cast<int>(integer(argv[index + 1], "repetitions"));
        } else {
            throw std::invalid_argument("unknown option: " + std::string(name));
        }
    }
    if (result.batch <= 0 || result.batch > 8 || result.heads <= 0 ||
        result.kv_heads <= 0 || result.heads % result.kv_heads != 0 ||
        result.sequence <= 0 || result.sequence > 4096 ||
        result.width <= 0 || result.width > 256 || result.warmup < 0 ||
        result.repetitions <= 0) {
        throw std::invalid_argument("online Attention operator options are invalid");
    }
    return result;
}

double percentile(std::vector<double> values, double fraction) {
    std::sort(values.begin(), values.end());
    const auto position = fraction * static_cast<double>(values.size() - 1);
    const auto lower = static_cast<std::size_t>(std::floor(position));
    const auto upper = static_cast<std::size_t>(std::ceil(position));
    const auto weight = position - static_cast<double>(lower);
    return values[lower] * (1.0 - weight) + values[upper] * weight;
}

struct Error {
    float maximum = 0.0F;
    double rms = 0.0;
    bool finite = true;
};

Error compare(const std::vector<float>& actual,
              const std::vector<float>& expected) {
    Error result;
    double squared = 0.0;
    for (std::size_t index = 0; index < actual.size(); ++index) {
        result.finite = result.finite && std::isfinite(actual[index]);
        const auto difference = std::abs(actual[index] - expected[index]);
        result.maximum = std::max(result.maximum, difference);
        squared += static_cast<double>(difference) * difference;
    }
    result.rms = std::sqrt(squared / static_cast<double>(actual.size()));
    return result;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const auto options = parse(argc, argv);
        if (microllm::runtime::hip_device_count() == 0) {
            throw std::runtime_error("online Attention operator benchmark needs HIP");
        }
        const auto query_elements = static_cast<std::size_t>(
            options.batch * options.heads * options.sequence * options.width);
        const auto kv_elements = static_cast<std::size_t>(
            options.batch * options.kv_heads * options.sequence * options.width);
        std::vector<float> query_values(query_elements);
        std::vector<float> key_values(kv_elements);
        std::vector<float> value_values(kv_elements);
        for (std::size_t index = 0; index < query_values.size(); ++index) {
            query_values[index] =
                static_cast<float>(static_cast<int>(index % 29) - 14) / 32.0F;
        }
        for (std::size_t index = 0; index < key_values.size(); ++index) {
            key_values[index] =
                static_cast<float>(static_cast<int>(index % 31) - 15) / 31.0F;
            value_values[index] =
                static_cast<float>(static_cast<int>(index % 37) - 18) / 37.0F;
        }
        const auto query = microllm::Tensor::from_vector(
            query_values,
            {options.batch, options.heads, options.sequence, options.width},
            microllm::DType::BFloat16);
        const auto key = microllm::Tensor::from_vector(
            key_values,
            {options.batch, options.kv_heads, options.sequence, options.width},
            microllm::DType::BFloat16);
        std::vector<float> value_bthd_values(kv_elements);
        for (std::int64_t batch = 0; batch < options.batch; ++batch) {
            for (std::int64_t token = 0; token < options.sequence; ++token) {
                for (std::int64_t head = 0; head < options.kv_heads; ++head) {
                    for (std::int64_t column = 0; column < options.width; ++column) {
                        const auto source = static_cast<std::size_t>(
                            ((batch * options.kv_heads + head) * options.sequence + token) *
                                options.width + column);
                        const auto destination = static_cast<std::size_t>(
                            ((batch * options.sequence + token) * options.kv_heads + head) *
                                options.width + column);
                        value_bthd_values[destination] = value_values[source];
                    }
                }
            }
        }
        const auto value = microllm::Tensor::from_vector(
            value_bthd_values,
            {options.batch, options.sequence, options.kv_heads, options.width},
            microllm::DType::BFloat16);
        const auto scale = 1.0F / std::sqrt(static_cast<float>(options.width));
        const auto repeats = options.heads / options.kv_heads;
        const auto expected = microllm::ops::online_causal_gqa_attention_bthd(
            query, key, value, repeats, scale).to_vector();
        const auto device = microllm::Device::hip(0);
        const auto device_query = query.to(device);
        const auto device_key = key.to(device);
        const auto device_value = value.to(device);
        const auto current_query = device_query.cast(microllm::DType::Float32);
        const auto current_key = device_key.cast(microllm::DType::Float32);
        const auto current_value = device_value.cast(microllm::DType::Float32);
        microllm::Tensor candidate;
        microllm::Tensor current;
        candidate = microllm::ops::online_causal_gqa_attention_bthd(
            device_query, device_key, device_value, repeats, scale);
        current = microllm::ops::causal_gqa_attention_bthd(
            current_query, current_key, current_value, repeats, scale);
        microllm::runtime::synchronize(device);
        const auto candidate_error = compare(candidate.to_vector(), expected);
        const auto current_error = compare(current.to_vector(), expected);
        const bool accuracy_passed =
            candidate_error.finite && current_error.finite &&
            candidate_error.maximum <= 2.0e-3F && candidate_error.rms <= 2.0e-4 &&
            current_error.maximum <= 3.0e-4F && current_error.rms <= 3.0e-5;
        if (!accuracy_passed) {
            throw std::runtime_error("online Attention operator complete-output gate failed");
        }

        microllm::runtime::Event start(device);
        microllm::runtime::Event finish(device);
        const auto measure = [&](auto launch) {
            for (int iteration = 0; iteration < options.warmup; ++iteration) launch();
            microllm::runtime::synchronize(device);
            std::vector<double> event_times;
            std::vector<double> wall_times;
            for (int iteration = 0; iteration < options.repetitions; ++iteration) {
                const auto wall_start = std::chrono::steady_clock::now();
                start.record_default_stream();
                launch();
                finish.record_default_stream();
                finish.synchronize();
                const auto wall_finish = std::chrono::steady_clock::now();
                event_times.push_back(finish.elapsed_ms_since(start));
                wall_times.push_back(std::chrono::duration<double, std::milli>(
                    wall_finish - wall_start).count());
            }
            return std::pair{event_times, wall_times};
        };
        microllm::ops::clear_rocwmma_online_attention_stats();
        microllm::runtime::reset_transfer_stats();
        const auto candidate_times = measure([&] {
            candidate = microllm::ops::online_causal_gqa_attention_bthd(
                device_query, device_key, device_value, repeats, scale);
        });
        const auto candidate_transfers = microllm::runtime::transfer_stats();
        const auto native_calls = microllm::ops::rocwmma_online_attention_native_calls();
        const auto fallback_calls = microllm::ops::rocwmma_online_attention_fallback_calls();
        microllm::runtime::reset_transfer_stats();
        const auto current_times = measure([&] {
            current = microllm::ops::causal_gqa_attention_bthd(
                current_query, current_key, current_value, repeats, scale);
        });
        const auto current_transfers = microllm::runtime::transfer_stats();
        if (candidate_transfers.host_to_device_calls != 0 ||
            candidate_transfers.device_to_host_calls != 0 ||
            current_transfers.host_to_device_calls != 0 ||
            current_transfers.device_to_host_calls != 0) {
            throw std::runtime_error("online Attention timing contains payload transfer");
        }
        const auto candidate_event = percentile(candidate_times.first, 0.50);
        const auto current_event = percentile(current_times.first, 0.50);
        const auto native_expected =
            microllm::ops::rocwmma_online_attention_available() &&
            microllm::runtime::device_info(device).architecture.starts_with("gfx942") &&
            options.sequence >= 32 && options.sequence % 32 == 0 &&
            (options.width == 64 || options.width == 128);
        const auto current_score_bytes =
            static_cast<std::uint64_t>(options.batch) *
            static_cast<std::uint64_t>(options.heads) *
            static_cast<std::uint64_t>(options.sequence) *
            static_cast<std::uint64_t>(options.sequence) * sizeof(float);
        std::cout << std::setprecision(9)
                  << "{\"schema_version\":1,\"status\":\"pass\""
                  << ",\"record_type\":\"online_attention_operator\""
                  << ",\"architecture\":\""
                  << microllm::runtime::device_info(device).architecture << "\""
                  << ",\"batch\":" << options.batch
                  << ",\"heads\":" << options.heads
                  << ",\"kv_heads\":" << options.kv_heads
                  << ",\"sequence\":" << options.sequence
                  << ",\"width\":" << options.width
                  << ",\"native_expected\":" << (native_expected ? "true" : "false")
                  << ",\"native_calls\":" << native_calls
                  << ",\"fallback_calls\":" << fallback_calls
                  << ",\"candidate_global_score_bytes\":0"
                  << ",\"current_score_bytes\":" << current_score_bytes
                  << ",\"complete_output_elements\":" << expected.size()
                  << ",\"candidate_max_error\":" << candidate_error.maximum
                  << ",\"candidate_rms_error\":" << candidate_error.rms
                  << ",\"current_max_error\":" << current_error.maximum
                  << ",\"current_rms_error\":" << current_error.rms
                  << ",\"accuracy_passed\":" << (accuracy_passed ? "true" : "false")
                  << ",\"warmup\":" << options.warmup
                  << ",\"repetitions\":" << options.repetitions
                  << ",\"candidate_event_ms_p50\":" << candidate_event
                  << ",\"candidate_event_ms_p95\":"
                  << percentile(candidate_times.first, 0.95)
                  << ",\"candidate_wall_ms_p50\":"
                  << percentile(candidate_times.second, 0.50)
                  << ",\"current_event_ms_p50\":" << current_event
                  << ",\"current_event_ms_p95\":"
                  << percentile(current_times.first, 0.95)
                  << ",\"current_wall_ms_p50\":"
                  << percentile(current_times.second, 0.50)
                  << ",\"candidate_over_current\":"
                  << current_event / candidate_event
                  << ",\"candidate_h2d_calls\":"
                  << candidate_transfers.host_to_device_calls
                  << ",\"candidate_d2h_calls\":"
                  << candidate_transfers.device_to_host_calls
                  << "}\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "benchmark_online_attention_operator: "
                  << error.what() << '\n';
        return 2;
    }
}
