#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include <microllm/ops/tuning.h>
#include <microllm/runtime/memory.h>
#include <microllm/runtime/runtime.h>

namespace {

struct Options {
    std::int64_t elements = 1 << 20;
    bool mirror = true;
    bool aligned = true;
    int warmup = 3;
    int repetitions = 10;
    float maximum_tolerance = -1.0F;
    float rms_tolerance = -1.0F;
    std::string mode = "unspecified";
    bool accept = false;
    std::filesystem::path cache_output;
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
        if (name == "--elements") {
            result.elements = integer(argv[index + 1], "elements");
        } else if (name == "--mirror") {
            result.mirror = boolean(argv[index + 1], "mirror");
        } else if (name == "--aligned") {
            result.aligned = boolean(argv[index + 1], "aligned");
        } else if (name == "--warmup") {
            result.warmup = static_cast<int>(integer(argv[index + 1], "warmup"));
        } else if (name == "--repetitions") {
            result.repetitions =
                static_cast<int>(integer(argv[index + 1], "repetitions"));
        } else if (name == "--maximum-absolute-tolerance") {
            result.maximum_tolerance = std::stof(argv[index + 1]);
        } else if (name == "--rms-tolerance") {
            result.rms_tolerance = std::stof(argv[index + 1]);
        } else if (name == "--mode") {
            result.mode = argv[index + 1];
        } else if (name == "--accept") {
            result.accept = boolean(argv[index + 1], "accept");
        } else if (name == "--cache-output") {
            result.cache_output = argv[index + 1];
        } else {
            throw std::invalid_argument("unknown option: " + std::string(name));
        }
    }
    if (result.elements <= 0 || result.elements > 300000000 ||
        result.warmup < 0 || result.repetitions <= 0 ||
        (result.mode != "unspecified" && result.mode != "training") ||
        (!result.accept && !result.cache_output.empty())) {
        throw std::invalid_argument(
            "AdamW tune options are outside the supported contract");
    }
    return result;
}

microllm::ops::OpMode mode(const std::string& name) {
    return name == "training" ? microllm::ops::OpMode::Training
                               : microllm::ops::OpMode::Unspecified;
}

const char* implementation(microllm::ops::AdamWImplementation value) {
    switch (value) {
        case microllm::ops::AdamWImplementation::Scalar: return "scalar";
        case microllm::ops::AdamWImplementation::Vectorized: return "vectorized";
        case microllm::ops::AdamWImplementation::Auto: return "auto";
    }
    return "unknown";
}

std::string escaped(const std::string& value) {
    std::string result;
    for (const auto character : value) {
        if (character == '"' || character == '\\') result.push_back('\\');
        if (character == '\n') result += "\\n";
        else result.push_back(character);
    }
    return result;
}

microllm::Tensor state_tensor(const std::vector<float>& values,
                              microllm::DType dtype,
                              microllm::Device device, bool aligned) {
    const auto host = microllm::Tensor::from_vector(
        values, {static_cast<std::int64_t>(values.size())}, dtype);
    if (aligned) return host.to(device);
    microllm::Tensor storage(
        {static_cast<std::int64_t>(values.size()) + 1}, dtype, device);
    auto result = storage.slice(
        0, 1, static_cast<std::int64_t>(values.size()) + 1);
    microllm::runtime::copy_bytes(
        result.data(), device, host.data(), host.device(),
        values.size() * microllm::dtype_size(dtype));
    return result;
}

void print_error(const char* name,
                 const microllm::ops::AdamWStateError& error) {
    std::cout << ",\"" << name << "_maximum_absolute_error\":"
              << error.maximum_absolute_error
              << ",\"" << name << "_rms_error\":" << error.rms_error;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const auto options = parse(argc, argv);
        if (microllm::runtime::hip_device_count() == 0) {
            throw std::runtime_error("AdamW tune requires a visible HIP device");
        }
        std::vector<float> parameter_values(
            static_cast<std::size_t>(options.elements));
        std::vector<float> gradient_values(
            static_cast<std::size_t>(options.elements));
        std::vector<float> first_values(
            static_cast<std::size_t>(options.elements));
        std::vector<float> second_values(
            static_cast<std::size_t>(options.elements));
        for (std::size_t index = 0; index < parameter_values.size(); ++index) {
            parameter_values[index] =
                static_cast<float>(static_cast<int>(index % 31) - 15) / 17.0F;
            gradient_values[index] =
                static_cast<float>(static_cast<int>(index % 13) - 6) / 19.0F;
            first_values[index] =
                static_cast<float>(static_cast<int>(index % 7) - 3) / 101.0F;
            second_values[index] = static_cast<float>(index % 11) / 97.0F;
        }
        const auto device = microllm::Device::hip(0);
        const auto parameter = state_tensor(
            parameter_values, microllm::DType::Float32, device, options.aligned);
        const auto gradient = state_tensor(
            gradient_values, microllm::DType::Float32, device, options.aligned);
        const auto first = state_tensor(
            first_values, microllm::DType::Float32, device, options.aligned);
        const auto second = state_tensor(
            second_values, microllm::DType::Float32, device, options.aligned);
        microllm::Tensor mirror;
        if (options.mirror) {
            mirror = parameter.cast(microllm::DType::BFloat16);
        }
        microllm::ops::AdamWAutotuneOptions tuning;
        tuning.warmup = options.warmup;
        tuning.repetitions = options.repetitions;
        tuning.maximum_absolute_tolerance = options.maximum_tolerance;
        tuning.rms_tolerance = options.rms_tolerance;
        tuning.mode = mode(options.mode);
        const auto report = microllm::ops::autotune_adamw(
            parameter, gradient, first, second,
            options.mirror ? &mirror : nullptr, tuning);
        if (options.accept) {
            microllm::ops::clear_adamw_implementation_registry();
            microllm::ops::register_adamw_autotune_winner(report);
            if (!options.cache_output.empty()) {
                microllm::ops::save_adamw_tuning_cache(options.cache_output);
            }
        }
        const auto info = microllm::runtime::device_info(device);
        std::cout << std::setprecision(9)
                  << "{\"schema_version\":1,\"status\":\"pass\""
                  << ",\"track\":\"adamw_correctness_before_timing\""
                  << ",\"architecture\":\"" << escaped(info.architecture) << "\""
                  << ",\"elements\":" << options.elements
                  << ",\"bf16_mirror\":" << (options.mirror ? "true" : "false")
                  << ",\"aligned16\":" << (options.aligned ? "true" : "false")
                  << ",\"mode\":\"" << options.mode << "\""
                  << ",\"warmup\":" << options.warmup
                  << ",\"repetitions\":" << options.repetitions
                  << ",\"reference_elements\":" << report.reference_elements
                  << ",\"maximum_absolute_tolerance\":"
                  << report.maximum_absolute_tolerance
                  << ",\"rms_tolerance\":" << report.rms_tolerance
                  << ",\"recommended\":\"" << implementation(report.recommended) << "\""
                  << ",\"accepted\":" << (options.accept ? "true" : "false")
                  << ",\"cache_output\":\""
                  << escaped(options.cache_output.string()) << "\""
                  << ",\"candidates\":[";
        for (std::size_t index = 0; index < report.candidates.size(); ++index) {
            if (index != 0) std::cout << ',';
            const auto& candidate = report.candidates[index];
            std::cout << "{\"implementation\":\""
                      << implementation(candidate.implementation) << "\""
                      << ",\"supported\":"
                      << (candidate.supported ? "true" : "false")
                      << ",\"correctness_passed\":"
                      << (candidate.correctness_passed ? "true" : "false")
                      << ",\"finite\":" << (candidate.finite ? "true" : "false");
            print_error("parameter", candidate.parameter);
            print_error("first_moment", candidate.first_moment);
            print_error("second_moment", candidate.second_moment);
            print_error("bf16_mirror", candidate.bf16_mirror);
            std::cout << ",\"event_ms_p50\":" << candidate.event_ms_p50
                      << ",\"event_ms_p95\":" << candidate.event_ms_p95
                      << ",\"wall_ms_p50\":" << candidate.wall_ms_p50
                      << ",\"wall_ms_p95\":" << candidate.wall_ms_p95
                      << ",\"failure\":\"" << escaped(candidate.failure) << "\"}";
        }
        std::cout << "],\"boundary\":\"operator state and timing only; "
                     "end-to-end acceptance remains external\"}\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "tune_adamw: " << error.what() << '\n';
        return 2;
    }
}
