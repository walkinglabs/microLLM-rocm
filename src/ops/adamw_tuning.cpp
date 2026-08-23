#include <microllm/ops/tuning.h>

#include <algorithm>
#include <atomic>
#include <charconv>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <map>
#include <mutex>
#include <set>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <microllm/runtime/memory.h>
#include <microllm/runtime/runtime.h>

namespace microllm::ops {
namespace {

constexpr int kAdamWTuningCacheSchema = 1;
constexpr std::int64_t kComparisonChunkElements = 1 << 20;

std::mutex adamw_registry_mutex;
std::map<AdamWTuningKey, AdamWImplementation> adamw_registry;
std::atomic<std::size_t> adamw_registry_entries{0};

struct TuningEnvironment {
    std::string architecture;
    int runtime_version = 0;
    int driver_version = 0;
};

TuningEnvironment tuning_environment(Device device) {
    if (device.is_cpu()) return {"host", 0, 0};
    static std::mutex mutex;
    static std::map<int, TuningEnvironment> environments;
    const std::lock_guard<std::mutex> lock(mutex);
    const auto found = environments.find(device.index());
    if (found != environments.end()) return found->second;
    const auto inserted = environments.emplace(
        device.index(),
        TuningEnvironment{runtime::device_info(device).architecture,
                          runtime::hip_runtime_version(),
                          runtime::hip_driver_version()});
    return inserted.first->second;
}

bool aligned16(const void* pointer) {
    return reinterpret_cast<std::uintptr_t>(pointer) % 16U == 0U;
}

void validate_adamw_tensors(
    const Tensor& parameter, const Tensor& gradient,
    const Tensor& first_moment, const Tensor& second_moment,
    const Tensor* mirror) {
    const auto valid_state = parameter.defined() && parameter.numel() > 0 &&
        parameter.dtype() == DType::Float32 &&
        gradient.dtype() == DType::Float32 &&
        first_moment.dtype() == DType::Float32 &&
        second_moment.dtype() == DType::Float32 &&
        gradient.shape() == parameter.shape() &&
        first_moment.shape() == parameter.shape() &&
        second_moment.shape() == parameter.shape() &&
        gradient.device() == parameter.device() &&
        first_moment.device() == parameter.device() &&
        second_moment.device() == parameter.device() &&
        parameter.is_contiguous() && gradient.is_contiguous() &&
        first_moment.is_contiguous() && second_moment.is_contiguous();
    if (!valid_state) {
        throw std::invalid_argument(
            "AdamW tuning requires matching contiguous FP32 state tensors");
    }
    if (mirror != nullptr &&
        (!mirror->defined() || mirror->dtype() != DType::BFloat16 ||
         mirror->shape() != parameter.shape() ||
         mirror->device() != parameter.device() || !mirror->is_contiguous())) {
        throw std::invalid_argument(
            "AdamW tuning BF16 mirror must match parameter shape and device");
    }
}

void validate_adamw_key(const AdamWTuningKey& key) {
    if (key.elements <= 0 || key.parameter_dtype != DType::Float32 ||
        key.gradient_dtype != DType::Float32 ||
        key.first_moment_dtype != DType::Float32 ||
        key.second_moment_dtype != DType::Float32 || key.architecture.empty() ||
        key.hip_runtime_version < 0 || key.hip_driver_version < 0) {
        throw std::invalid_argument("registered AdamW tuning key is incomplete");
    }
}

void validate_adamw_choice(const AdamWTuningKey& key,
                           AdamWImplementation implementation) {
    validate_adamw_key(key);
    if (implementation == AdamWImplementation::Auto) {
        throw std::invalid_argument(
            "AdamW registry choice must name a concrete implementation");
    }
    if (implementation == AdamWImplementation::Vectorized &&
        (!key.parameter_aligned16 || !key.gradient_aligned16 ||
         !key.first_moment_aligned16 || !key.second_moment_aligned16)) {
        throw std::invalid_argument(
            "vectorized AdamW cannot be registered for unaligned state");
    }
}

void validate_options(const AdamWAutotuneOptions& options) {
    const auto valid_hyperparameters = options.learning_rate > 0.0F &&
        options.beta1 >= 0.0F && options.beta1 < 1.0F &&
        options.beta2 >= 0.0F && options.beta2 < 1.0F &&
        options.epsilon > 0.0F && options.weight_decay >= 0.0F &&
        options.first_correction > 0.0F && options.second_correction > 0.0F;
    if (options.warmup < 0 || options.repetitions <= 0 ||
        !std::isfinite(options.maximum_absolute_tolerance) ||
        !std::isfinite(options.rms_tolerance) || !valid_hyperparameters ||
        options.candidates.empty()) {
        throw std::invalid_argument("AdamW autotune options are invalid");
    }
    std::set<AdamWImplementation> unique;
    for (const auto candidate : options.candidates) {
        if (candidate == AdamWImplementation::Auto ||
            !unique.insert(candidate).second) {
            throw std::invalid_argument(
                "AdamW autotune candidates must be unique concrete implementations");
        }
    }
}

std::string json_string(std::string_view value) {
    std::string output;
    output.reserve(value.size() + 2);
    output.push_back('"');
    for (const auto character : value) {
        if (character == '"' || character == '\\') output.push_back('\\');
        output.push_back(character);
    }
    output.push_back('"');
    return output;
}

std::size_t field_start(std::string_view line, std::string_view name) {
    const auto needle = "\"" + std::string(name) + "\":";
    auto position = line.find(needle);
    if (position == std::string_view::npos) {
        throw std::runtime_error("AdamW tuning cache field is missing: " +
                                 std::string(name));
    }
    position += needle.size();
    while (position < line.size() && line[position] == ' ') ++position;
    return position;
}

void require_delimiter(std::string_view line, std::size_t position,
                       std::string_view name) {
    while (position < line.size() && line[position] == ' ') ++position;
    if (position >= line.size() ||
        (line[position] != ',' && line[position] != '}')) {
        throw std::runtime_error("AdamW tuning cache value is invalid: " +
                                 std::string(name));
    }
}

template <typename Integer>
Integer integer_field(std::string_view line, std::string_view name) {
    const auto position = field_start(line, name);
    Integer value{};
    const auto parsed = std::from_chars(
        line.data() + position, line.data() + line.size(), value);
    if (parsed.ec != std::errc{} || parsed.ptr == line.data() + position) {
        throw std::runtime_error("AdamW tuning cache integer is invalid: " +
                                 std::string(name));
    }
    require_delimiter(line,
                      static_cast<std::size_t>(parsed.ptr - line.data()), name);
    return value;
}

std::string string_field(std::string_view line, std::string_view name) {
    auto position = field_start(line, name);
    if (position >= line.size() || line[position] != '"') {
        throw std::runtime_error("AdamW tuning cache string is invalid: " +
                                 std::string(name));
    }
    ++position;
    std::string output;
    while (position < line.size()) {
        const auto character = line[position++];
        if (character == '"') {
            require_delimiter(line, position, name);
            return output;
        }
        if (character == '\\') {
            if (position >= line.size() ||
                (line[position] != '\\' && line[position] != '"')) {
                throw std::runtime_error("AdamW tuning cache escape is invalid");
            }
            output.push_back(line[position++]);
        } else {
            output.push_back(character);
        }
    }
    throw std::runtime_error("AdamW tuning cache string is unterminated");
}

bool bool_field(std::string_view line, std::string_view name) {
    const auto position = field_start(line, name);
    const auto value = line.substr(position);
    if (value.starts_with("true")) {
        require_delimiter(line, position + 4, name);
        return true;
    }
    if (value.starts_with("false")) {
        require_delimiter(line, position + 5, name);
        return false;
    }
    throw std::runtime_error("AdamW tuning cache bool is invalid: " +
                             std::string(name));
}

const char* mode_name(OpMode mode) {
    switch (mode) {
        case OpMode::Unspecified: return "unspecified";
        case OpMode::Inference: return "inference";
        case OpMode::Training: return "training";
    }
    throw std::invalid_argument("unknown AdamW mode");
}

OpMode mode_from_name(const std::string& name) {
    if (name == "unspecified") return OpMode::Unspecified;
    if (name == "inference") return OpMode::Inference;
    if (name == "training") return OpMode::Training;
    throw std::runtime_error("AdamW tuning cache mode is unsupported: " + name);
}

const char* implementation_name(AdamWImplementation implementation) {
    switch (implementation) {
        case AdamWImplementation::Scalar: return "scalar";
        case AdamWImplementation::Vectorized: return "vectorized";
        case AdamWImplementation::Auto: break;
    }
    throw std::invalid_argument("automatic AdamW choice cannot be serialized");
}

AdamWImplementation implementation_from_name(const std::string& name) {
    if (name == "scalar") return AdamWImplementation::Scalar;
    if (name == "vectorized") return AdamWImplementation::Vectorized;
    throw std::runtime_error(
        "AdamW tuning cache implementation is unsupported: " + name);
}

std::pair<AdamWTuningKey, AdamWImplementation> parse_cache_entry(
    std::string_view line) {
    if (integer_field<int>(line, "schema_version") !=
            kAdamWTuningCacheSchema || string_field(line, "kind") != "entry") {
        throw std::runtime_error("AdamW tuning cache entry schema is invalid");
    }
    AdamWTuningKey key;
    key.elements = integer_field<std::int64_t>(line, "elements");
    if (string_field(line, "parameter_dtype") != "float32" ||
        string_field(line, "gradient_dtype") != "float32" ||
        string_field(line, "first_moment_dtype") != "float32" ||
        string_field(line, "second_moment_dtype") != "float32") {
        throw std::runtime_error("AdamW tuning cache state dtype is unsupported");
    }
    key.bf16_mirror = bool_field(line, "bf16_mirror");
    key.parameter_aligned16 = bool_field(line, "parameter_aligned16");
    key.gradient_aligned16 = bool_field(line, "gradient_aligned16");
    key.first_moment_aligned16 = bool_field(line, "first_moment_aligned16");
    key.second_moment_aligned16 = bool_field(line, "second_moment_aligned16");
    key.architecture = string_field(line, "architecture");
    key.hip_runtime_version = integer_field<int>(line, "hip_runtime_version");
    key.hip_driver_version = integer_field<int>(line, "hip_driver_version");
    key.mode = mode_from_name(string_field(line, "mode"));
    validate_adamw_key(key);
    return {std::move(key), implementation_from_name(
        string_field(line, "implementation"))};
}

Tensor clone_preserving_alignment(const Tensor& source) {
    const auto source_aligned = aligned16(source.data());
    if (source_aligned) {
        Tensor result(source.shape(), source.dtype(), source.device());
        runtime::copy_bytes(result.data(), result.device(), source.data(), source.device(),
                            static_cast<std::size_t>(source.numel()) *
                                dtype_size(source.dtype()));
        return result;
    }
    Tensor storage({source.numel() + 1}, source.dtype(), source.device());
    auto result = storage.slice(0, 1, source.numel() + 1).reshape(source.shape());
    runtime::copy_bytes(result.data(), result.device(), source.data(), source.device(),
                        static_cast<std::size_t>(source.numel()) *
                            dtype_size(source.dtype()));
    return result;
}

struct AdamWState {
    Tensor parameter;
    Tensor gradient;
    Tensor first;
    Tensor second;
    Tensor mirror;
};

AdamWState clone_state(const Tensor& parameter, const Tensor& gradient,
                       const Tensor& first, const Tensor& second,
                       const Tensor* mirror) {
    AdamWState result{clone_preserving_alignment(parameter),
                      clone_preserving_alignment(gradient),
                      clone_preserving_alignment(first),
                      clone_preserving_alignment(second), {}};
    if (mirror != nullptr) result.mirror = clone_preserving_alignment(*mirror);
    return result;
}

void update_state(AdamWState& state, AdamWImplementation implementation,
                  const AdamWAutotuneOptions& options,
                  const OpContext& context) {
    if (state.mirror.defined()) {
        adamw_update_bf16_mirror_(
            state.parameter, state.gradient, state.first, state.second,
            state.mirror, options.learning_rate, options.beta1, options.beta2,
            options.epsilon, options.weight_decay, options.first_correction,
            options.second_correction, context, implementation);
    } else {
        adamw_update_(state.parameter, state.gradient, state.first, state.second,
                      options.learning_rate, options.beta1, options.beta2,
                      options.epsilon, options.weight_decay,
                      options.first_correction, options.second_correction,
                      context, implementation);
    }
}

AdamWStateError compare_complete(const Tensor& actual, const Tensor& reference,
                                 bool& finite) {
    if (actual.shape() != reference.shape() || actual.dtype() != reference.dtype()) {
        throw std::runtime_error("AdamW autotune state shape or dtype changed");
    }
    const auto actual_flat = actual.reshape({actual.numel()});
    const auto reference_flat = reference.reshape({reference.numel()});
    float maximum = 0.0F;
    double squared = 0.0;
    finite = true;
    for (std::int64_t begin = 0; begin < actual.numel();
         begin += kComparisonChunkElements) {
        const auto end = std::min(actual.numel(), begin + kComparisonChunkElements);
        const auto actual_values = actual_flat.slice(0, begin, end).to_vector();
        const auto reference_values = reference_flat.slice(0, begin, end).to_vector();
        for (std::size_t index = 0; index < actual_values.size(); ++index) {
            if (!std::isfinite(actual_values[index]) ||
                !std::isfinite(reference_values[index])) {
                finite = false;
                continue;
            }
            const auto difference =
                std::abs(actual_values[index] - reference_values[index]);
            maximum = std::max(maximum, difference);
            squared += static_cast<double>(difference) * difference;
        }
    }
    return {maximum,
            std::sqrt(squared / static_cast<double>(actual.numel()))};
}

double percentile(std::vector<double> values, double fraction) {
    if (values.empty()) throw std::invalid_argument("cannot summarize empty timings");
    std::sort(values.begin(), values.end());
    const auto index = static_cast<std::size_t>(
        std::ceil(fraction * static_cast<double>(values.size()))) - 1U;
    return values[std::min(index, values.size() - 1U)];
}

bool error_passed(const AdamWStateError& error, float maximum_tolerance,
                  double rms_tolerance) {
    return error.maximum_absolute_error <= maximum_tolerance &&
           error.rms_error <= rms_tolerance;
}

}  // namespace

AdamWTuningKey make_adamw_tuning_key(
    const Tensor& parameter, const Tensor& gradient,
    const Tensor& first_moment, const Tensor& second_moment,
    const Tensor* bf16_mirror, const OpContext& context) {
    validate_adamw_tensors(parameter, gradient, first_moment, second_moment,
                           bf16_mirror);
    const auto environment = tuning_environment(parameter.device());
    return {.elements = parameter.numel(),
            .parameter_dtype = parameter.dtype(),
            .gradient_dtype = gradient.dtype(),
            .first_moment_dtype = first_moment.dtype(),
            .second_moment_dtype = second_moment.dtype(),
            .bf16_mirror = bf16_mirror != nullptr,
            .parameter_aligned16 = aligned16(parameter.data()),
            .gradient_aligned16 = aligned16(gradient.data()),
            .first_moment_aligned16 = aligned16(first_moment.data()),
            .second_moment_aligned16 = aligned16(second_moment.data()),
            .architecture = environment.architecture,
            .hip_runtime_version = environment.runtime_version,
            .hip_driver_version = environment.driver_version,
            .mode = context.mode};
}

AdamWImplementation choose_adamw_implementation(
    const Tensor& parameter, const Tensor& gradient,
    const Tensor& first_moment, const Tensor& second_moment,
    const Tensor* bf16_mirror, const OpContext& context) {
    if (!parameter.device().is_hip()) return AdamWImplementation::Scalar;
    const auto key = make_adamw_tuning_key(
        parameter, gradient, first_moment, second_moment, bf16_mirror, context);
    if (adamw_registry_entries.load(std::memory_order_acquire) != 0) {
        const std::lock_guard<std::mutex> lock(adamw_registry_mutex);
        const auto found = adamw_registry.find(key);
        if (found != adamw_registry.end()) return found->second;
    }
    return AdamWImplementation::Scalar;
}

void register_adamw_implementation(const AdamWTuningKey& key,
                                   AdamWImplementation implementation) {
    validate_adamw_choice(key, implementation);
    const std::lock_guard<std::mutex> lock(adamw_registry_mutex);
    const auto [unused, inserted] =
        adamw_registry.insert_or_assign(key, implementation);
    (void)unused;
    if (inserted) adamw_registry_entries.fetch_add(1, std::memory_order_release);
}

void clear_adamw_implementation_registry() {
    const std::lock_guard<std::mutex> lock(adamw_registry_mutex);
    adamw_registry.clear();
    adamw_registry_entries.store(0, std::memory_order_release);
}

std::size_t adamw_registered_implementation_count() noexcept {
    return adamw_registry_entries.load(std::memory_order_acquire);
}

void save_adamw_tuning_cache(const std::filesystem::path& path) {
    if (path.empty() || !path.has_filename()) {
        throw std::invalid_argument("AdamW tuning cache path must name a file");
    }
    std::vector<std::pair<AdamWTuningKey, AdamWImplementation>> entries;
    {
        const std::lock_guard<std::mutex> lock(adamw_registry_mutex);
        entries.assign(adamw_registry.begin(), adamw_registry.end());
    }
    auto temporary = path;
    temporary += ".tmp";
    try {
        std::ofstream output(temporary, std::ios::trunc);
        if (!output) {
            throw std::runtime_error("cannot open temporary AdamW tuning cache");
        }
        output << "{\"schema_version\":" << kAdamWTuningCacheSchema
               << ",\"kind\":\"microllm_adamw_tuning_cache\"}\n";
        for (const auto& [key, implementation] : entries) {
            output << "{\"schema_version\":" << kAdamWTuningCacheSchema
                   << ",\"kind\":\"entry\""
                   << ",\"elements\":" << key.elements
                   << ",\"parameter_dtype\":\"float32\""
                   << ",\"gradient_dtype\":\"float32\""
                   << ",\"first_moment_dtype\":\"float32\""
                   << ",\"second_moment_dtype\":\"float32\""
                   << ",\"bf16_mirror\":"
                   << (key.bf16_mirror ? "true" : "false")
                   << ",\"parameter_aligned16\":"
                   << (key.parameter_aligned16 ? "true" : "false")
                   << ",\"gradient_aligned16\":"
                   << (key.gradient_aligned16 ? "true" : "false")
                   << ",\"first_moment_aligned16\":"
                   << (key.first_moment_aligned16 ? "true" : "false")
                   << ",\"second_moment_aligned16\":"
                   << (key.second_moment_aligned16 ? "true" : "false")
                   << ",\"architecture\":" << json_string(key.architecture)
                   << ",\"hip_runtime_version\":" << key.hip_runtime_version
                   << ",\"hip_driver_version\":" << key.hip_driver_version
                   << ",\"mode\":" << json_string(mode_name(key.mode))
                   << ",\"implementation\":"
                   << json_string(implementation_name(implementation)) << "}\n";
        }
        output.flush();
        if (!output) throw std::runtime_error("cannot write AdamW tuning cache");
        output.close();
        std::error_code error;
        std::filesystem::rename(temporary, path, error);
        if (error) {
            throw std::runtime_error(
                "cannot atomically replace AdamW tuning cache: " + error.message());
        }
    } catch (...) {
        std::error_code ignored;
        std::filesystem::remove(temporary, ignored);
        throw;
    }
}

AdamWTuningCacheLoadReport load_adamw_tuning_cache(
    const std::filesystem::path& path, Device device, bool replace_existing) {
    std::ifstream input(path);
    if (!input) throw std::runtime_error("cannot open AdamW tuning cache");
    std::string line;
    if (!std::getline(input, line) || line.size() > 65536 ||
        integer_field<int>(line, "schema_version") != kAdamWTuningCacheSchema ||
        string_field(line, "kind") != "microllm_adamw_tuning_cache") {
        throw std::runtime_error("AdamW tuning cache header is invalid");
    }
    std::vector<std::pair<AdamWTuningKey, AdamWImplementation>> accepted;
    std::set<AdamWTuningKey> keys;
    AdamWTuningCacheLoadReport report;
    const auto environment = tuning_environment(device);
    while (std::getline(input, line)) {
        if (line.empty()) continue;
        if (line.size() > 65536 || report.parsed_entries >= 100000) {
            throw std::runtime_error("AdamW tuning cache exceeds safety limits");
        }
        auto entry = parse_cache_entry(line);
        ++report.parsed_entries;
        if (!keys.insert(entry.first).second) {
            throw std::runtime_error("AdamW tuning cache contains a duplicate key");
        }
        const auto& key = entry.first;
        if (key.architecture != environment.architecture ||
            key.hip_runtime_version != environment.runtime_version ||
            key.hip_driver_version != environment.driver_version) {
            ++report.stale_entries;
            continue;
        }
        validate_adamw_choice(key, entry.second);
        accepted.push_back(std::move(entry));
    }
    if (!input.eof()) throw std::runtime_error("cannot read AdamW tuning cache");
    {
        const std::lock_guard<std::mutex> lock(adamw_registry_mutex);
        auto updated = replace_existing
                           ? std::map<AdamWTuningKey, AdamWImplementation>{}
                           : adamw_registry;
        for (auto& [key, implementation] : accepted) {
            updated.insert_or_assign(std::move(key), implementation);
        }
        adamw_registry.swap(updated);
        adamw_registry_entries.store(adamw_registry.size(),
                                     std::memory_order_release);
        report.loaded_entries = accepted.size();
    }
    return report;
}

AdamWAutotuneReport autotune_adamw(
    const Tensor& parameter, const Tensor& gradient,
    const Tensor& first_moment, const Tensor& second_moment,
    const Tensor* bf16_mirror, const AdamWAutotuneOptions& options) {
    validate_options(options);
    validate_adamw_tensors(parameter, gradient, first_moment, second_moment,
                           bf16_mirror);
    if (!parameter.device().is_hip()) {
        throw std::invalid_argument("AdamW autotune requires state on one HIP device");
    }
    const auto maximum_tolerance = options.maximum_absolute_tolerance < 0.0F
                                       ? 2.0e-6F
                                       : options.maximum_absolute_tolerance;
    const auto rms_tolerance = options.rms_tolerance < 0.0F
                                   ? 5.0e-7
                                   : static_cast<double>(options.rms_tolerance);
    OpContext context;
    context.mode = options.mode;
    runtime::synchronize(parameter.device());

    auto reference = clone_state(parameter, gradient, first_moment, second_moment,
                                 bf16_mirror);
    update_state(reference, AdamWImplementation::Scalar, options, context);
    runtime::synchronize(parameter.device());

    AdamWAutotuneReport report;
    report.key = make_adamw_tuning_key(parameter, gradient, first_moment,
                                       second_moment, bf16_mirror, context);
    report.reference_elements = parameter.numel();
    report.maximum_absolute_tolerance = maximum_tolerance;
    report.rms_tolerance = rms_tolerance;

    runtime::Event start(parameter.device());
    runtime::Event finish(parameter.device());
    for (const auto implementation : options.candidates) {
        AdamWAutotuneCandidate candidate;
        candidate.implementation = implementation;
        try {
            auto checked = clone_state(parameter, gradient, first_moment,
                                       second_moment, bf16_mirror);
            update_state(checked, implementation, options, context);
            runtime::synchronize(parameter.device());
            bool parameter_finite = true;
            bool first_finite = true;
            bool second_finite = true;
            bool mirror_finite = true;
            candidate.parameter = compare_complete(
                checked.parameter, reference.parameter, parameter_finite);
            candidate.first_moment = compare_complete(
                checked.first, reference.first, first_finite);
            candidate.second_moment = compare_complete(
                checked.second, reference.second, second_finite);
            if (bf16_mirror != nullptr) {
                candidate.bf16_mirror = compare_complete(
                    checked.mirror, reference.mirror, mirror_finite);
            }
            candidate.supported = true;
            candidate.finite = parameter_finite && first_finite && second_finite &&
                               mirror_finite;
            candidate.correctness_passed = candidate.finite &&
                error_passed(candidate.parameter, maximum_tolerance, rms_tolerance) &&
                error_passed(candidate.first_moment, maximum_tolerance, rms_tolerance) &&
                error_passed(candidate.second_moment, maximum_tolerance, rms_tolerance) &&
                error_passed(candidate.bf16_mirror, maximum_tolerance, rms_tolerance);
            if (!candidate.correctness_passed) {
                candidate.failure = "complete-state correctness gate failed";
                report.candidates.push_back(std::move(candidate));
                continue;
            }

            auto timed = clone_state(parameter, gradient, first_moment,
                                     second_moment, bf16_mirror);
            for (int iteration = 0; iteration < options.warmup; ++iteration) {
                update_state(timed, implementation, options, context);
            }
            runtime::synchronize(parameter.device());
            std::vector<double> event_times;
            std::vector<double> wall_times;
            event_times.reserve(static_cast<std::size_t>(options.repetitions));
            wall_times.reserve(static_cast<std::size_t>(options.repetitions));
            for (int iteration = 0; iteration < options.repetitions; ++iteration) {
                const auto wall_start = std::chrono::steady_clock::now();
                start.record_default_stream();
                update_state(timed, implementation, options, context);
                finish.record_default_stream();
                finish.synchronize();
                const auto wall_finish = std::chrono::steady_clock::now();
                event_times.push_back(finish.elapsed_ms_since(start));
                wall_times.push_back(std::chrono::duration<double, std::milli>(
                    wall_finish - wall_start).count());
            }
            candidate.event_ms_p50 = percentile(event_times, 0.50);
            candidate.event_ms_p95 = percentile(event_times, 0.95);
            candidate.wall_ms_p50 = percentile(wall_times, 0.50);
            candidate.wall_ms_p95 = percentile(wall_times, 0.95);
        } catch (const std::exception& error) {
            candidate.failure = error.what();
        }
        report.candidates.push_back(std::move(candidate));
    }

    const AdamWAutotuneCandidate* best = nullptr;
    for (const auto& candidate : report.candidates) {
        if (!candidate.supported || !candidate.correctness_passed ||
            !(candidate.event_ms_p50 > 0.0) || !(candidate.event_ms_p95 > 0.0)) {
            continue;
        }
        if (best == nullptr ||
            std::pair(candidate.event_ms_p50, candidate.event_ms_p95) <
                std::pair(best->event_ms_p50, best->event_ms_p95)) {
            best = &candidate;
        }
    }
    if (best == nullptr) {
        throw std::runtime_error("no AdamW candidate passed complete-state correctness");
    }
    report.recommended = best->implementation;
    return report;
}

void register_adamw_autotune_winner(const AdamWAutotuneReport& report) {
    if (report.recommended == AdamWImplementation::Auto) {
        throw std::invalid_argument("AdamW autotune report has no recommendation");
    }
    const auto found = std::find_if(
        report.candidates.begin(), report.candidates.end(),
        [&](const auto& candidate) {
            return candidate.implementation == report.recommended;
        });
    if (found == report.candidates.end() || !found->supported ||
        !found->correctness_passed || !found->finite ||
        !(found->event_ms_p50 > 0.0) || !(found->event_ms_p95 > 0.0)) {
        throw std::invalid_argument(
            "AdamW recommendation lacks correctness and timing evidence");
    }
    register_adamw_implementation(report.key, report.recommended);
}

}  // namespace microllm::ops
