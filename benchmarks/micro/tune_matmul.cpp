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
#include <microllm/runtime/runtime.h>

namespace {

struct Options {
    std::int64_t rows = 128;
    std::int64_t inner = 128;
    std::int64_t columns = 128;
    std::string dtype = "fp32";
    bool transpose_left = false;
    bool transpose_right = false;
    int warmup = 3;
    int repetitions = 10;
    float maximum_tolerance = -1.0F;
    float rms_tolerance = -1.0F;
    std::size_t workspace_limit = 0;
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
        if (index + 1 >= argc) throw std::invalid_argument("option is missing a value");
        const std::string_view name(argv[index]);
        if (name == "--m") result.rows = integer(argv[index + 1], "m");
        else if (name == "--k") result.inner = integer(argv[index + 1], "k");
        else if (name == "--n") result.columns = integer(argv[index + 1], "n");
        else if (name == "--dtype") result.dtype = argv[index + 1];
        else if (name == "--transpose-left") {
            result.transpose_left = boolean(argv[index + 1], "transpose-left");
        } else if (name == "--transpose-right") {
            result.transpose_right = boolean(argv[index + 1], "transpose-right");
        } else if (name == "--warmup") {
            result.warmup = static_cast<int>(integer(argv[index + 1], "warmup"));
        } else if (name == "--repetitions") {
            result.repetitions = static_cast<int>(integer(argv[index + 1], "repetitions"));
        } else if (name == "--maximum-absolute-tolerance") {
            result.maximum_tolerance = std::stof(argv[index + 1]);
        } else if (name == "--rms-tolerance") {
            result.rms_tolerance = std::stof(argv[index + 1]);
        } else if (name == "--workspace-limit") {
            const auto parsed = integer(argv[index + 1], "workspace-limit");
            if (parsed < 0) throw std::invalid_argument("workspace-limit must be nonnegative");
            result.workspace_limit = static_cast<std::size_t>(parsed);
        } else if (name == "--mode") result.mode = argv[index + 1];
        else if (name == "--accept") result.accept = boolean(argv[index + 1], "accept");
        else if (name == "--cache-output") result.cache_output = argv[index + 1];
        else throw std::invalid_argument("unknown option: " + std::string(name));
    }
    if (result.rows <= 0 || result.inner <= 0 || result.columns <= 0 ||
        result.rows > 16384 || result.inner > 16384 || result.columns > 16384 ||
        result.warmup < 0 || result.repetitions <= 0 ||
        (result.dtype != "fp32" && result.dtype != "fp16" && result.dtype != "bf16") ||
        (result.mode != "unspecified" && result.mode != "inference" &&
         result.mode != "training") || (!result.accept && !result.cache_output.empty())) {
        throw std::invalid_argument("matmul tune options are outside the supported contract");
    }
    return result;
}

microllm::DType dtype(const std::string& name) {
    if (name == "fp16") return microllm::DType::Float16;
    if (name == "bf16") return microllm::DType::BFloat16;
    return microllm::DType::Float32;
}

microllm::ops::OpMode mode(const std::string& name) {
    if (name == "inference") return microllm::ops::OpMode::Inference;
    if (name == "training") return microllm::ops::OpMode::Training;
    return microllm::ops::OpMode::Unspecified;
}

const char* implementation(microllm::ops::MatmulImplementation value) {
    switch (value) {
        case microllm::ops::MatmulImplementation::Readable: return "readable";
        case microllm::ops::MatmulImplementation::HipBLASLt: return "hipblaslt";
        case microllm::ops::MatmulImplementation::Auto: return "auto";
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

}  // namespace

int main(int argc, char** argv) {
    try {
        const auto options = parse(argc, argv);
        if (microllm::runtime::hip_device_count() == 0) {
            throw std::runtime_error("matmul tune requires a visible HIP device");
        }
        const auto left_rows = options.transpose_left ? options.inner : options.rows;
        const auto left_columns = options.transpose_left ? options.rows : options.inner;
        const auto right_rows = options.transpose_right ? options.columns : options.inner;
        const auto right_columns = options.transpose_right ? options.inner : options.columns;
        std::vector<float> left_values(
            static_cast<std::size_t>(left_rows * left_columns));
        std::vector<float> right_values(
            static_cast<std::size_t>(right_rows * right_columns));
        for (std::size_t index = 0; index < left_values.size(); ++index) {
            left_values[index] =
                static_cast<float>(static_cast<int>(index % 29) - 14) / 29.0F;
        }
        for (std::size_t index = 0; index < right_values.size(); ++index) {
            right_values[index] =
                static_cast<float>(static_cast<int>(index % 17) - 8) / 17.0F;
        }
        const auto device = microllm::Device::hip(0);
        const auto operand_dtype = dtype(options.dtype);
        const auto left = microllm::Tensor::from_vector(
            left_values, {left_rows, left_columns}, operand_dtype).to(device);
        const auto right = microllm::Tensor::from_vector(
            right_values, {right_rows, right_columns}, operand_dtype).to(device);
        microllm::ops::MatmulAutotuneOptions tuning;
        tuning.warmup = options.warmup;
        tuning.repetitions = options.repetitions;
        tuning.maximum_absolute_tolerance = options.maximum_tolerance;
        tuning.rms_tolerance = options.rms_tolerance;
        tuning.workspace_limit = options.workspace_limit;
        tuning.mode = mode(options.mode);
        const auto report = microllm::ops::autotune_matmul(
            left, right, options.transpose_left, options.transpose_right, tuning);
        if (options.accept) {
            microllm::ops::clear_matmul_implementation_registry();
            microllm::ops::register_matmul_autotune_winner(report);
            if (!options.cache_output.empty()) {
                microllm::ops::save_matmul_tuning_cache(options.cache_output);
            }
        }
        const auto info = microllm::runtime::device_info(device);
        std::cout << std::setprecision(9)
                  << "{\"schema_version\":1,\"status\":\"pass\""
                  << ",\"track\":\"matmul_correctness_before_timing\""
                  << ",\"architecture\":\"" << escaped(info.architecture) << "\""
                  << ",\"dtype\":\"" << options.dtype << "\""
                  << ",\"m\":" << options.rows
                  << ",\"k\":" << options.inner
                  << ",\"n\":" << options.columns
                  << ",\"warmup\":" << options.warmup
                  << ",\"repetitions\":" << options.repetitions
                  << ",\"reference_elements\":" << report.reference_elements
                  << ",\"maximum_absolute_tolerance\":"
                  << report.maximum_absolute_tolerance
                  << ",\"rms_tolerance\":" << report.rms_tolerance
                  << ",\"recommended\":\"" << implementation(report.recommended) << "\""
                  << ",\"accepted\":" << (options.accept ? "true" : "false")
                  << ",\"cache_output\":\"" << escaped(options.cache_output.string()) << "\""
                  << ",\"candidates\":[";
        for (std::size_t index = 0; index < report.candidates.size(); ++index) {
            if (index != 0) std::cout << ',';
            const auto& candidate = report.candidates[index];
            std::cout << "{\"implementation\":\""
                      << implementation(candidate.implementation) << "\""
                      << ",\"supported\":" << (candidate.supported ? "true" : "false")
                      << ",\"correctness_passed\":"
                      << (candidate.correctness_passed ? "true" : "false")
                      << ",\"finite\":" << (candidate.finite ? "true" : "false")
                      << ",\"maximum_absolute_error\":"
                      << candidate.maximum_absolute_error
                      << ",\"rms_error\":" << candidate.rms_error
                      << ",\"event_ms_p50\":" << candidate.event_ms_p50
                      << ",\"event_ms_p95\":" << candidate.event_ms_p95
                      << ",\"wall_ms_p50\":" << candidate.wall_ms_p50
                      << ",\"wall_ms_p95\":" << candidate.wall_ms_p95
                      << ",\"failure\":\"" << escaped(candidate.failure) << "\"}";
        }
        std::cout << "],\"boundary\":\"operator correctness and timing only; "
                     "end-to-end acceptance remains external\"}\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "tune_matmul: " << error.what() << '\n';
        return 2;
    }
}
