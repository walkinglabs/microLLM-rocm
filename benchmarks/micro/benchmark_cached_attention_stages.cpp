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
#include <utility>
#include <vector>

#include <microllm/ops/ops.h>
#include <microllm/runtime/memory.h>
#include <microllm/runtime/runtime.h>

namespace {

struct Options {
    std::int64_t batch = 1;
    std::int64_t heads = 12;
    std::int64_t kv_heads = 2;
    std::int64_t sequence = 512;
    std::int64_t width = 128;
    std::int64_t splits = 0;
    std::int64_t pv_splits = 0;
    std::int64_t gqa_tile_columns = 32;
    std::int64_t finalize_threads = 256;
    bool materialized = false;
    bool native128 = false;
    bool gqa_value_reuse = false;
    std::string cache_dtype = "bf16";
    std::string order = "forward";
    int warmup = 3;
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

bool boolean(const char* value, const char* name) {
    const std::string_view text(value);
    if (text == "true") return true;
    if (text == "false") return false;
    throw std::invalid_argument(std::string(name) + " must be true or false");
}

Options parse(int argc, char** argv) {
    Options result;
    for (int index = 1; index < argc; index += 2) {
        if (index + 1 >= argc) {
            throw std::invalid_argument("option is missing a value");
        }
        const std::string_view name(argv[index]);
        if (name == "--batch") {
            result.batch = integer(argv[index + 1], "batch");
        } else if (name == "--heads") {
            result.heads = integer(argv[index + 1], "heads");
        } else if (name == "--kv-heads") {
            result.kv_heads = integer(argv[index + 1], "kv-heads");
        } else if (name == "--sequence") {
            result.sequence = integer(argv[index + 1], "sequence");
        } else if (name == "--width") {
            result.width = integer(argv[index + 1], "width");
        } else if (name == "--splits") {
            result.splits = integer(argv[index + 1], "splits");
        } else if (name == "--pv-splits") {
            result.pv_splits = integer(argv[index + 1], "pv-splits");
        } else if (name == "--gqa-tile-columns") {
            result.gqa_tile_columns = integer(
                argv[index + 1], "gqa-tile-columns");
        } else if (name == "--materialized") {
            result.materialized = boolean(argv[index + 1], "materialized");
        } else if (name == "--native128") {
            result.native128 = boolean(argv[index + 1], "native128");
        } else if (name == "--gqa-value-reuse") {
            result.gqa_value_reuse = boolean(
                argv[index + 1], "gqa-value-reuse");
        } else if (name == "--finalize-threads") {
            result.finalize_threads = integer(
                argv[index + 1], "finalize-threads");
        } else if (name == "--cache-dtype") {
            result.cache_dtype = argv[index + 1];
        } else if (name == "--order") {
            result.order = argv[index + 1];
        } else if (name == "--warmup") {
            result.warmup = static_cast<int>(integer(argv[index + 1], "warmup"));
        } else if (name == "--repetitions") {
            result.repetitions = static_cast<int>(
                integer(argv[index + 1], "repetitions"));
        } else {
            throw std::invalid_argument("unknown option: " + std::string(name));
        }
    }
    if (result.batch <= 0 || result.batch > 8 || result.heads <= 0 ||
        result.kv_heads <= 0 || result.heads % result.kv_heads != 0 ||
        result.sequence <= 0 || result.sequence > 4096 ||
        result.width <= 0 || result.width > 256 ||
        result.splits < 0 || result.splits > 32 ||
        result.splits > result.sequence ||
        result.pv_splits < 0 || result.pv_splits > 32 ||
        result.pv_splits > result.sequence ||
        (result.gqa_tile_columns != 8 && result.gqa_tile_columns != 16 &&
         result.gqa_tile_columns != 32 && result.gqa_tile_columns != 64) ||
        (result.finalize_threads != 64 && result.finalize_threads != 128 &&
         result.finalize_threads != 256) ||
        (result.native128 && !result.materialized) ||
        (result.cache_dtype != "fp32" && result.cache_dtype != "bf16") ||
        (result.order != "forward" && result.order != "reverse") ||
        result.warmup < 3 || result.repetitions <= 0 ||
        result.repetitions > 10000) {
        throw std::invalid_argument(
            "cached Attention stage options are outside the measured contract");
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
    if (actual.size() != expected.size() || actual.empty()) {
        throw std::invalid_argument("comparison shapes are inconsistent");
    }
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

struct Timing {
    double event_p50 = 0.0;
    double event_p95 = 0.0;
    double wall_p50 = 0.0;
    double wall_p95 = 0.0;
    double allocation_calls_per_invocation = 0.0;
    double backend_allocation_calls_per_invocation = 0.0;
    double cache_reuse_calls_per_invocation = 0.0;
};

template <typename Operation>
Timing measure(Operation&& operation, const Options& options,
               microllm::Device device) {
    for (int iteration = 0; iteration < options.warmup; ++iteration) {
        operation();
    }
    microllm::runtime::synchronize(device);
    const auto allocation_before = microllm::runtime::allocation_stats(device);
    microllm::runtime::Event start(device);
    microllm::runtime::Event finish(device);
    std::vector<double> event_times;
    std::vector<double> wall_times;
    event_times.reserve(static_cast<std::size_t>(options.repetitions));
    wall_times.reserve(static_cast<std::size_t>(options.repetitions));
    for (int iteration = 0; iteration < options.repetitions; ++iteration) {
        const auto wall_start = std::chrono::steady_clock::now();
        start.record_default_stream();
        operation();
        finish.record_default_stream();
        finish.synchronize();
        const auto wall_finish = std::chrono::steady_clock::now();
        event_times.push_back(finish.elapsed_ms_since(start));
        wall_times.push_back(std::chrono::duration<double, std::milli>(
            wall_finish - wall_start).count());
    }
    const auto allocation_after = microllm::runtime::allocation_stats(device);
    const auto denominator = static_cast<double>(options.repetitions);
    return {
        percentile(event_times, 0.50), percentile(event_times, 0.95),
        percentile(wall_times, 0.50), percentile(wall_times, 0.95),
        static_cast<double>(allocation_after.allocation_calls -
                            allocation_before.allocation_calls) / denominator,
        static_cast<double>(allocation_after.backend_allocation_calls -
                            allocation_before.backend_allocation_calls) / denominator,
        static_cast<double>(allocation_after.cache_reuse_calls -
                            allocation_before.cache_reuse_calls) / denominator};
}

void print_timing(std::string_view name, const Timing& timing) {
    std::cout << ",\"" << name << "_event_ms_p50\":" << timing.event_p50
              << ",\"" << name << "_event_ms_p95\":" << timing.event_p95
              << ",\"" << name << "_wall_ms_p50\":" << timing.wall_p50
              << ",\"" << name << "_wall_ms_p95\":" << timing.wall_p95
              << ",\"" << name << "_allocation_calls_per_invocation\":"
              << timing.allocation_calls_per_invocation
              << ",\"" << name
              << "_backend_allocation_calls_per_invocation\":"
              << timing.backend_allocation_calls_per_invocation
              << ",\"" << name << "_cache_reuse_calls_per_invocation\":"
              << timing.cache_reuse_calls_per_invocation;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const auto options = parse(argc, argv);
        if (microllm::runtime::hip_device_count() == 0) {
            throw std::runtime_error(
                "cached Attention stage benchmark requires HIP");
        }
        const auto device = microllm::Device::hip(0);
        microllm::runtime::enable_hip_caching_allocator(device);
        const auto cache_dtype = options.cache_dtype == "bf16"
            ? microllm::DType::BFloat16 : microllm::DType::Float32;
        const auto repeats = options.heads / options.kv_heads;
        const auto scale = 1.0F / std::sqrt(static_cast<float>(options.width));
        const auto query_elements = static_cast<std::size_t>(
            options.batch * options.heads * options.width);
        const auto cache_elements = static_cast<std::size_t>(
            options.batch * options.kv_heads * options.sequence * options.width);
        std::vector<float> query_values(query_elements);
        std::vector<float> key_values(cache_elements);
        std::vector<float> value_values(cache_elements);
        for (std::size_t index = 0; index < query_values.size(); ++index) {
            query_values[index] =
                static_cast<float>(static_cast<int>(index % 37U) - 18) *
                0.015625F;
        }
        for (std::size_t index = 0; index < key_values.size(); ++index) {
            key_values[index] =
                static_cast<float>(static_cast<int>(index % 41U) - 20) *
                0.005859375F;
            value_values[index] =
                static_cast<float>(static_cast<int>(index % 43U) - 21) *
                0.0048828125F;
        }
        const auto query = microllm::Tensor::from_vector(
            query_values, {options.batch, options.heads, 1, options.width});
        const auto key = microllm::Tensor::from_vector(
            key_values,
            {options.batch, options.kv_heads, options.sequence, options.width},
            cache_dtype);
        const auto value = microllm::Tensor::from_vector(
            value_values,
            {options.batch, options.kv_heads, options.sequence, options.width},
            cache_dtype);
        const auto expected_scores = microllm::ops::cached_gqa_attention_scores(
            query, key, repeats, scale);
        const auto expected_probabilities = microllm::ops::softmax(
            expected_scores, -1);
        const auto expected_context = microllm::ops::cached_gqa_attention_context(
            expected_probabilities, value, repeats);
        const auto expected_score_values = expected_scores.to_vector();
        const auto expected_probability_values =
            expected_probabilities.to_vector();
        const auto expected_context_values = expected_context.to_vector();

        const auto device_query = query.to(device);
        const auto device_key = key.to(device);
        const auto device_value = value.to(device);
        auto device_scores = microllm::ops::cached_gqa_attention_scores(
            device_query, device_key, repeats, scale);
        auto device_probabilities = microllm::ops::softmax(device_scores, -1);
        auto device_context = microllm::ops::cached_gqa_attention_context(
            device_probabilities, device_value, repeats);
        auto device_pipeline = microllm::ops::cached_gqa_attention_context(
            microllm::ops::softmax(
                microllm::ops::cached_gqa_attention_scores(
                    device_query, device_key, repeats, scale), -1),
            device_value, repeats);
        auto device_fused = microllm::ops::cached_gqa_attention(
            device_query, device_key, device_value, repeats, scale);
        microllm::Tensor device_split;
        if (options.splits > 0) {
            device_split = microllm::ops::cached_gqa_attention_split_sequence(
                device_query, device_key, device_value, repeats, scale,
                options.splits);
        }
        microllm::Tensor device_materialized;
        if (options.materialized) {
            device_materialized =
                microllm::ops::cached_gqa_attention_materialized_scores(
                    device_query, device_key, device_value, repeats, scale,
                    options.finalize_threads);
        }
        microllm::Tensor device_native128;
        if (options.native128) {
            device_native128 =
                microllm::ops::cached_gqa_attention_materialized_scores_native128(
                    device_query, device_key, device_value, repeats, scale);
        }
        microllm::Tensor device_split_pv;
        if (options.pv_splits > 0) {
            device_split_pv =
                microllm::ops::cached_gqa_attention_split_pv_exact_softmax(
                    device_query, device_key, device_value, repeats, scale,
                    options.pv_splits);
        }
        microllm::Tensor device_gqa_value_reuse;
        if (options.gqa_value_reuse) {
            device_gqa_value_reuse =
                microllm::ops::cached_gqa_attention_gqa_value_reuse(
                    device_query, device_key, device_value, repeats, scale,
                    options.gqa_tile_columns);
        }
        microllm::runtime::synchronize(device);

        const auto score_error = compare(
            device_scores.to_vector(), expected_score_values);
        const auto probability_error = compare(
            device_probabilities.to_vector(), expected_probability_values);
        const auto context_error = compare(
            device_context.to_vector(), expected_context_values);
        const auto pipeline_error = compare(
            device_pipeline.to_vector(), expected_context_values);
        const auto fused_error = compare(
            device_fused.to_vector(), expected_context_values);
        Error split_error;
        if (options.splits > 0) {
            split_error = compare(
                device_split.to_vector(), expected_context_values);
        }
        Error materialized_error;
        bool materialized_bitwise_equal = true;
        if (options.materialized) {
            const auto materialized_values = device_materialized.to_vector();
            const auto fused_values = device_fused.to_vector();
            materialized_error = compare(
                materialized_values, expected_context_values);
            materialized_bitwise_equal = materialized_values == fused_values;
        }
        Error native128_error;
        bool native128_bitwise_equal_materialized = true;
        if (options.native128) {
            const auto native_values = device_native128.to_vector();
            const auto materialized_values = device_materialized.to_vector();
            native128_error = compare(native_values, expected_context_values);
            native128_bitwise_equal_materialized =
                native_values == materialized_values;
        }
        Error split_pv_error;
        bool split_pv_bitwise_equal_materialized = true;
        if (options.pv_splits > 0) {
            const auto split_pv_values = device_split_pv.to_vector();
            const auto reference_values = options.materialized
                ? device_materialized.to_vector() : device_fused.to_vector();
            split_pv_error = compare(
                split_pv_values, expected_context_values);
            split_pv_bitwise_equal_materialized =
                split_pv_values == reference_values;
        }
        Error gqa_value_reuse_error;
        bool gqa_value_reuse_bitwise_equal_materialized = true;
        if (options.gqa_value_reuse) {
            const auto candidate_values = device_gqa_value_reuse.to_vector();
            const auto reference_values = options.materialized
                ? device_materialized.to_vector() : device_fused.to_vector();
            gqa_value_reuse_error = compare(
                candidate_values, expected_context_values);
            gqa_value_reuse_bitwise_equal_materialized =
                candidate_values == reference_values;
        }
        const auto accuracy_passed =
            score_error.finite && probability_error.finite &&
            context_error.finite && pipeline_error.finite && fused_error.finite &&
            score_error.maximum <= 2.0e-4F && score_error.rms <= 3.0e-5 &&
            probability_error.maximum <= 3.0e-4F &&
            probability_error.rms <= 3.0e-5 &&
            context_error.maximum <= 8.0e-4F && context_error.rms <= 8.0e-5 &&
            pipeline_error.maximum <= 8.0e-4F && pipeline_error.rms <= 8.0e-5 &&
            fused_error.maximum <= 8.0e-4F && fused_error.rms <= 8.0e-5 &&
            (options.splits == 0 ||
             (split_error.finite && split_error.maximum <= 8.0e-4F &&
              split_error.rms <= 8.0e-5)) &&
            (!options.materialized ||
             (materialized_error.finite && materialized_bitwise_equal)) &&
            (!options.native128 ||
             (native128_error.finite &&
              native128_error.maximum <= 8.0e-4F &&
              native128_error.rms <= 8.0e-5)) &&
            (options.pv_splits == 0 ||
             (split_pv_error.finite && split_pv_error.maximum <= 8.0e-4F &&
              split_pv_error.rms <= 8.0e-5 &&
              (options.pv_splits != 1 ||
               split_pv_bitwise_equal_materialized)));
        const auto value_reuse_accuracy =
            !options.gqa_value_reuse ||
            (gqa_value_reuse_error.finite &&
             gqa_value_reuse_bitwise_equal_materialized);
        if (!accuracy_passed || !value_reuse_accuracy) {
            throw std::runtime_error(
                "cached Attention stage complete-output gate failed");
        }

        const auto score_operation = [&] {
            device_scores = microllm::ops::cached_gqa_attention_scores(
                device_query, device_key, repeats, scale);
        };
        const auto softmax_operation = [&] {
            device_probabilities = microllm::ops::softmax(device_scores, -1);
        };
        const auto context_operation = [&] {
            device_context = microllm::ops::cached_gqa_attention_context(
                device_probabilities, device_value, repeats);
        };
        const auto pipeline_operation = [&] {
            device_pipeline = microllm::ops::cached_gqa_attention_context(
                microllm::ops::softmax(
                    microllm::ops::cached_gqa_attention_scores(
                        device_query, device_key, repeats, scale), -1),
                device_value, repeats);
        };
        const auto fused_operation = [&] {
            device_fused = microllm::ops::cached_gqa_attention(
                device_query, device_key, device_value, repeats, scale);
        };
        const auto split_operation = [&] {
            device_split = microllm::ops::cached_gqa_attention_split_sequence(
                device_query, device_key, device_value, repeats, scale,
                options.splits);
        };
        const auto materialized_operation = [&] {
            device_materialized =
                microllm::ops::cached_gqa_attention_materialized_scores(
                    device_query, device_key, device_value, repeats, scale,
                    options.finalize_threads);
        };
        const auto native128_operation = [&] {
            device_native128 =
                microllm::ops::cached_gqa_attention_materialized_scores_native128(
                    device_query, device_key, device_value, repeats, scale);
        };
        const auto split_pv_operation = [&] {
            device_split_pv =
                microllm::ops::cached_gqa_attention_split_pv_exact_softmax(
                    device_query, device_key, device_value, repeats, scale,
                    options.pv_splits);
        };
        const auto gqa_value_reuse_operation = [&] {
            device_gqa_value_reuse =
                microllm::ops::cached_gqa_attention_gqa_value_reuse(
                    device_query, device_key, device_value, repeats, scale,
                    options.gqa_tile_columns);
        };

        Timing score_timing;
        Timing softmax_timing;
        Timing context_timing;
        Timing pipeline_timing;
        Timing fused_timing;
        Timing split_timing;
        Timing materialized_timing;
        Timing native128_timing;
        Timing split_pv_timing;
        Timing gqa_value_reuse_timing;
        microllm::runtime::reset_transfer_stats();
        if (options.order == "forward") {
            score_timing = measure(score_operation, options, device);
            softmax_timing = measure(softmax_operation, options, device);
            context_timing = measure(context_operation, options, device);
            pipeline_timing = measure(pipeline_operation, options, device);
            fused_timing = measure(fused_operation, options, device);
            if (options.splits > 0) {
                split_timing = measure(split_operation, options, device);
            }
            if (options.materialized) {
                materialized_timing = measure(
                    materialized_operation, options, device);
            }
            if (options.native128) {
                native128_timing = measure(
                    native128_operation, options, device);
            }
            if (options.pv_splits > 0) {
                split_pv_timing = measure(
                    split_pv_operation, options, device);
            }
            if (options.gqa_value_reuse) {
                gqa_value_reuse_timing = measure(
                    gqa_value_reuse_operation, options, device);
            }
        } else {
            if (options.native128) {
                native128_timing = measure(
                    native128_operation, options, device);
            }
            if (options.gqa_value_reuse) {
                gqa_value_reuse_timing = measure(
                    gqa_value_reuse_operation, options, device);
            }
            if (options.pv_splits > 0) {
                split_pv_timing = measure(
                    split_pv_operation, options, device);
            }
            if (options.materialized) {
                materialized_timing = measure(
                    materialized_operation, options, device);
            }
            if (options.splits > 0) {
                split_timing = measure(split_operation, options, device);
            }
            fused_timing = measure(fused_operation, options, device);
            pipeline_timing = measure(pipeline_operation, options, device);
            context_timing = measure(context_operation, options, device);
            softmax_timing = measure(softmax_operation, options, device);
            score_timing = measure(score_operation, options, device);
        }
        const auto transfers = microllm::runtime::transfer_stats();
        if (transfers.host_to_device_calls != 0 ||
            transfers.device_to_host_calls != 0) {
            throw std::runtime_error(
                "cached Attention stage timing contains a payload transfer");
        }
        if (score_timing.backend_allocation_calls_per_invocation != 0.0 ||
            softmax_timing.backend_allocation_calls_per_invocation != 0.0 ||
            context_timing.backend_allocation_calls_per_invocation != 0.0 ||
            pipeline_timing.backend_allocation_calls_per_invocation != 0.0 ||
            fused_timing.backend_allocation_calls_per_invocation != 0.0 ||
            (options.splits > 0 &&
             split_timing.backend_allocation_calls_per_invocation != 0.0) ||
            (options.materialized &&
             materialized_timing.backend_allocation_calls_per_invocation != 0.0) ||
            (options.native128 &&
             native128_timing.backend_allocation_calls_per_invocation != 0.0) ||
            (options.pv_splits > 0 &&
             split_pv_timing.backend_allocation_calls_per_invocation != 0.0) ||
            (options.gqa_value_reuse &&
             gqa_value_reuse_timing.backend_allocation_calls_per_invocation !=
                 0.0)) {
            throw std::runtime_error(
                "warm stage measurement reached the backend allocator");
        }

        const auto info = microllm::runtime::device_info(device);
        const auto score_elements = static_cast<std::uint64_t>(
            options.batch * options.heads * options.sequence);
        const auto context_elements = static_cast<std::uint64_t>(
            options.batch * options.heads * options.width);
        const auto stage_sum = score_timing.event_p50 +
            softmax_timing.event_p50 + context_timing.event_p50;
        std::cout << std::setprecision(12)
                  << "{\"schema_version\":1,\"status\":\"pass\""
                  << ",\"record_type\":\"cached_attention_stage_probe\""
                  << ",\"device_name\":\"" << info.name << "\""
                  << ",\"architecture\":\"" << info.architecture << "\""
                  << ",\"batch\":" << options.batch
                  << ",\"heads\":" << options.heads
                  << ",\"kv_heads\":" << options.kv_heads
                  << ",\"sequence\":" << options.sequence
                  << ",\"width\":" << options.width
                  << ",\"repeats\":" << repeats
                  << ",\"cache_dtype\":\"" << options.cache_dtype << "\""
                  << ",\"order\":\"" << options.order << "\""
                  << ",\"warmup\":" << options.warmup
                  << ",\"repetitions\":" << options.repetitions
                  << ",\"score_elements\":" << score_elements
                  << ",\"context_elements\":" << context_elements
                  << ",\"global_score_bytes\":" << score_elements * sizeof(float)
                  << ",\"complete_output_accuracy_passed\":true"
                  << ",\"score_max_error\":" << score_error.maximum
                  << ",\"score_rms_error\":" << score_error.rms
                  << ",\"probability_max_error\":"
                  << probability_error.maximum
                  << ",\"probability_rms_error\":" << probability_error.rms
                  << ",\"context_max_error\":" << context_error.maximum
                  << ",\"context_rms_error\":" << context_error.rms
                  << ",\"pipeline_max_error\":" << pipeline_error.maximum
                  << ",\"pipeline_rms_error\":" << pipeline_error.rms
                  << ",\"fused_max_error\":" << fused_error.maximum
                  << ",\"fused_rms_error\":" << fused_error.rms;
        if (options.splits > 0) {
            const auto partial_elements = static_cast<std::uint64_t>(
                options.batch * options.heads * options.splits *
                (options.width + 2));
            std::cout << ",\"splits\":" << options.splits
                      << ",\"split_partial_blocks\":"
                      << options.batch * options.heads * options.splits
                      << ",\"split_combine_blocks\":"
                      << options.batch * options.heads
                      << ",\"split_partial_bytes\":"
                      << partial_elements * sizeof(float)
                      << ",\"split_max_error\":" << split_error.maximum
                      << ",\"split_rms_error\":" << split_error.rms;
        }
        if (options.materialized) {
            std::cout << ",\"materialized_score_bytes\":"
                      << score_elements * sizeof(float)
                      << ",\"materialized_finalize_threads\":"
                      << options.finalize_threads
                      << ",\"materialized_max_error\":"
                      << materialized_error.maximum
                      << ",\"materialized_rms_error\":"
                      << materialized_error.rms
                      << ",\"materialized_bitwise_equal_current\":"
                      << (materialized_bitwise_equal ? "true" : "false");
        }
        if (options.native128) {
            std::cout << ",\"native128_max_error\":"
                      << native128_error.maximum
                      << ",\"native128_rms_error\":"
                      << native128_error.rms
                      << ",\"native128_bitwise_equal_materialized\":"
                      << (native128_bitwise_equal_materialized
                              ? "true" : "false");
        }
        if (options.pv_splits > 0) {
            const auto probability_bytes = score_elements * sizeof(float);
            const auto partial_elements = static_cast<std::uint64_t>(
                options.batch * options.heads * options.pv_splits *
                options.width);
            std::cout << ",\"pv_splits\":" << options.pv_splits
                      << ",\"split_pv_probability_bytes\":"
                      << probability_bytes
                      << ",\"split_pv_partial_bytes\":"
                      << partial_elements * sizeof(float)
                      << ",\"split_pv_max_error\":"
                      << split_pv_error.maximum
                      << ",\"split_pv_rms_error\":"
                      << split_pv_error.rms
                      << ",\"split_pv_bitwise_equal_materialized\":"
                      << (split_pv_bitwise_equal_materialized ? "true" : "false");
        }
        if (options.gqa_value_reuse) {
            std::cout << ",\"gqa_value_reuse_tile_columns\":"
                      << options.gqa_tile_columns
                      << ",\"gqa_value_reuse_probability_bytes\":"
                      << score_elements * sizeof(float)
                      << ",\"gqa_value_reuse_max_error\":"
                      << gqa_value_reuse_error.maximum
                      << ",\"gqa_value_reuse_rms_error\":"
                      << gqa_value_reuse_error.rms
                      << ",\"gqa_value_reuse_bitwise_equal_materialized\":"
                      << (gqa_value_reuse_bitwise_equal_materialized
                              ? "true" : "false");
        }
        print_timing("score", score_timing);
        print_timing("softmax", softmax_timing);
        print_timing("context", context_timing);
        print_timing("pipeline", pipeline_timing);
        print_timing("fused", fused_timing);
        if (options.splits > 0) {
            print_timing("split", split_timing);
        }
        if (options.materialized) {
            print_timing("materialized", materialized_timing);
        }
        if (options.native128) {
            print_timing("native128", native128_timing);
        }
        if (options.pv_splits > 0) {
            print_timing("split_pv", split_pv_timing);
        }
        if (options.gqa_value_reuse) {
            print_timing("gqa_value_reuse", gqa_value_reuse_timing);
        }
        std::cout << ",\"stage_sum_event_ms_p50\":" << stage_sum
                  << ",\"stage_sum_over_pipeline\":"
                  << stage_sum / pipeline_timing.event_p50
                  << ",\"fused_speedup_over_pipeline\":"
                  << pipeline_timing.event_p50 / fused_timing.event_p50
                  ;
        if (options.splits > 0) {
            std::cout << ",\"split_speedup_over_fused\":"
                      << fused_timing.event_p50 / split_timing.event_p50;
        }
        if (options.materialized) {
            std::cout << ",\"materialized_speedup_over_fused\":"
                      << fused_timing.event_p50 /
                             materialized_timing.event_p50;
        }
        if (options.pv_splits > 0 && options.materialized) {
            std::cout << ",\"split_pv_speedup_over_materialized\":"
                      << materialized_timing.event_p50 /
                             split_pv_timing.event_p50;
        }
        if (options.gqa_value_reuse && options.materialized) {
            std::cout << ",\"gqa_value_reuse_speedup_over_materialized\":"
                      << materialized_timing.event_p50 /
                             gqa_value_reuse_timing.event_p50;
        }
        std::cout << ",\"host_to_device_calls\":"
                  << transfers.host_to_device_calls
                  << ",\"device_to_host_calls\":"
                  << transfers.device_to_host_calls << "}\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "benchmark_cached_attention_stages: "
                  << error.what() << '\n';
        return 2;
    }
}
