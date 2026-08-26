#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <microllm/ops/ops.h>
#include <microllm/runtime/runtime.h>

namespace {

struct Options {
    std::vector<std::int64_t> rows{1, 2, 4, 8};
    std::int64_t inner = 1536;
    std::int64_t columns = 8960;
    std::vector<int> candidates;
    int warmup = 1;
    int repetitions = 3;
};

template <typename Integer>
std::vector<Integer> integer_list(const std::string& text, const char* name) {
    std::vector<Integer> result;
    std::stringstream stream(text);
    std::string item;
    while (std::getline(stream, item, ',')) {
        if (item.empty()) throw std::invalid_argument(std::string(name) + " is empty");
        const auto parsed = std::stoll(item);
        if (parsed <= 0) throw std::invalid_argument(std::string(name) + " must be positive");
        result.push_back(static_cast<Integer>(parsed));
    }
    if (result.empty()) throw std::invalid_argument(std::string(name) + " is empty");
    return result;
}

Options options(int argc, char** argv) {
    Options result;
    for (int index = 1; index < argc; index += 2) {
        if (index + 1 >= argc) throw std::invalid_argument("missing CLI value");
        const std::string name = argv[index];
        if (name == "--rows") {
            result.rows = integer_list<std::int64_t>(argv[index + 1], "rows");
        } else if (name == "--inner") {
            result.inner = std::stoll(argv[index + 1]);
        } else if (name == "--columns") {
            result.columns = std::stoll(argv[index + 1]);
        } else if (name == "--candidates") {
            result.candidates = integer_list<int>(argv[index + 1], "candidates");
        } else if (name == "--warmup") {
            result.warmup = std::stoi(argv[index + 1]);
        } else if (name == "--repetitions") {
            result.repetitions = std::stoi(argv[index + 1]);
        } else {
            throw std::invalid_argument("unknown CLI option: " + name);
        }
    }
    if (result.inner <= 0 || result.columns <= 0 || result.inner > 16384 ||
        result.columns > 200000 || result.candidates.empty() ||
        result.warmup < 0 || result.repetitions <= 0 ||
        result.rows != std::vector<std::int64_t>({1, 2, 4, 8}) ||
        std::adjacent_find(result.candidates.begin(), result.candidates.end()) !=
            result.candidates.end()) {
        throw std::invalid_argument("BF16 row-invariance options are outside the contract");
    }
    return result;
}

double median(std::vector<double> values) {
    std::sort(values.begin(), values.end());
    const auto middle = values.size() / 2U;
    return values.size() % 2U == 0U
               ? (values[middle - 1U] + values[middle]) / 2.0
               : values[middle];
}

struct Error {
    float maximum = 0.0F;
    double rms = 0.0;
    bool exact = true;
};

Error error(const std::vector<float>& left, const std::vector<float>& right) {
    if (left.empty() || left.size() != right.size()) {
        throw std::invalid_argument("row comparison sizes changed");
    }
    double square = 0.0;
    Error result;
    for (std::size_t index = 0; index < left.size(); ++index) {
        const auto delta = std::abs(left[index] - right[index]);
        result.maximum = std::max(result.maximum, delta);
        square += static_cast<double>(delta) * delta;
        result.exact = result.exact && left[index] == right[index];
    }
    result.rms = std::sqrt(square / static_cast<double>(left.size()));
    return result;
}

std::vector<float> cpu_reference(const std::vector<float>& input,
                                 const std::vector<float>& weight,
                                 std::int64_t inner, std::int64_t columns) {
    std::vector<float> output(static_cast<std::size_t>(columns));
    for (std::int64_t column = 0; column < columns; ++column) {
        float sum = 0.0F;
        for (std::int64_t index = 0; index < inner; ++index) {
            sum += input[static_cast<std::size_t>(index)] *
                   weight[static_cast<std::size_t>(index * columns + column)];
        }
        output[static_cast<std::size_t>(column)] = sum;
    }
    return microllm::Tensor::from_vector(
               output, {1, columns}, microllm::DType::BFloat16)
        .to_vector();
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const auto command = options(argc, argv);
        if (microllm::runtime::hip_device_count() == 0) {
            throw std::runtime_error("BF16 row-invariance search needs a HIP device");
        }
        const auto device = microllm::Device::hip(0);
        std::vector<float> input_values(static_cast<std::size_t>(command.inner));
        for (std::size_t index = 0; index < input_values.size(); ++index) {
            input_values[index] =
                static_cast<float>(static_cast<int>(index % 29U) - 14) / 29.0F;
        }
        const auto input_cpu = microllm::Tensor::from_vector(
            input_values, {1, command.inner}, microllm::DType::BFloat16);
        const auto rounded_input = input_cpu.to_vector();
        std::vector<float> weight_values(
            static_cast<std::size_t>(command.inner * command.columns));
        for (std::size_t index = 0; index < weight_values.size(); ++index) {
            weight_values[index] =
                static_cast<float>(static_cast<int>(index % 17U) - 8) / 17.0F;
        }
        const auto weight_cpu = microllm::Tensor::from_vector(
            weight_values, {command.inner, command.columns},
            microllm::DType::BFloat16);
        weight_values.clear();
        weight_values.shrink_to_fit();
        const auto rounded_weight = weight_cpu.to_vector();
        const auto reference = cpu_reference(
            rounded_input, rounded_weight, command.inner, command.columns);
        const auto weight = weight_cpu.to(device);
        std::vector<microllm::Tensor> inputs;
        inputs.reserve(command.rows.size());
        for (const auto rows : command.rows) {
            std::vector<float> repeated(
                static_cast<std::size_t>(rows * command.inner));
            for (std::int64_t row = 0; row < rows; ++row) {
                std::copy(rounded_input.begin(), rounded_input.end(),
                          repeated.begin() + row * command.inner);
            }
            inputs.push_back(microllm::Tensor::from_vector(
                repeated, {rows, command.inner}, microllm::DType::BFloat16)
                                 .to(device));
        }

        microllm::runtime::Event start(device);
        microllm::runtime::Event finish(device);
        std::size_t supported_count = 0;
        std::size_t reference_pass_count = 0;
        std::size_t invariant_count = 0;
        bool first = true;
        std::cout << std::setprecision(9)
                  << "{\"schema_version\":1,\"status\":\"pass\""
                  << ",\"record_type\":\"bf16_operator_row_invariance\""
                  << ",\"rows\":[1,2,4,8]"
                  << ",\"inner\":" << command.inner
                  << ",\"columns\":" << command.columns
                  << ",\"output_dtype\":\"bfloat16\""
                  << ",\"candidate_count\":" << command.candidates.size()
                  << ",\"complete_row_elements\":" << command.columns
                  << ",\"candidates\":[";
        for (const auto candidate : command.candidates) {
            bool supported = false;
            bool reference_passed = false;
            bool invariant = false;
            float reference_maximum = 0.0F;
            double reference_rms = 0.0;
            float row_maximum = 0.0F;
            double row_rms = 0.0;
            std::vector<double> event_p50;
            std::string failure;
            try {
                microllm::ops::clear_bf16_algorithm_registry();
                for (const auto rows : command.rows) {
                    microllm::ops::register_bf16_algorithm(
                        rows, command.inner, command.columns,
                        microllm::DType::BFloat16, candidate);
                }
                std::vector<std::vector<float>> outputs;
                outputs.reserve(command.rows.size());
                for (std::size_t shape = 0; shape < command.rows.size(); ++shape) {
                    auto output = microllm::ops::bf16_matmul_output(
                        inputs[shape], weight, microllm::DType::BFloat16);
                    microllm::runtime::synchronize(device);
                    const auto values = output.to_vector();
                    const auto rows = command.rows[shape];
                    std::vector<float> first_row(
                        values.begin(), values.begin() + command.columns);
                    const auto cpu_error = error(reference, first_row);
                    reference_maximum = std::max(
                        reference_maximum, cpu_error.maximum);
                    reference_rms = std::max(reference_rms, cpu_error.rms);
                    for (std::int64_t row = 1; row < rows; ++row) {
                        const std::vector<float> other(
                            values.begin() + row * command.columns,
                            values.begin() + (row + 1) * command.columns);
                        const auto within = error(first_row, other);
                        row_maximum = std::max(row_maximum, within.maximum);
                        row_rms = std::max(row_rms, within.rms);
                    }
                    outputs.push_back(std::move(first_row));

                    for (int iteration = 0; iteration < command.warmup; ++iteration) {
                        output = microllm::ops::bf16_matmul_output(
                            inputs[shape], weight, microllm::DType::BFloat16);
                    }
                    microllm::runtime::synchronize(device);
                    std::vector<double> times;
                    for (int iteration = 0; iteration < command.repetitions;
                         ++iteration) {
                        start.record_default_stream();
                        output = microllm::ops::bf16_matmul_output(
                            inputs[shape], weight, microllm::DType::BFloat16);
                        finish.record_default_stream();
                        finish.synchronize();
                        times.push_back(finish.elapsed_ms_since(start));
                    }
                    event_p50.push_back(median(std::move(times)));
                }
                for (std::size_t shape = 1; shape < outputs.size(); ++shape) {
                    const auto across = error(outputs.front(), outputs[shape]);
                    row_maximum = std::max(row_maximum, across.maximum);
                    row_rms = std::max(row_rms, across.rms);
                }
                supported = true;
                reference_passed = reference_maximum <= 0.03125F &&
                                   reference_rms <= 0.005;
                invariant = reference_passed && row_maximum == 0.0F;
            } catch (const std::exception& error) {
                failure = error.what();
            }
            supported_count += supported;
            reference_pass_count += reference_passed;
            invariant_count += invariant;
            if (!first) std::cout << ',';
            first = false;
            std::cout << "{\"index\":" << candidate
                      << ",\"supported\":" << (supported ? "true" : "false")
                      << ",\"reference_passed\":"
                      << (reference_passed ? "true" : "false")
                      << ",\"row_invariant\":"
                      << (invariant ? "true" : "false")
                      << ",\"reference_maximum_error\":" << reference_maximum
                      << ",\"reference_rms_error\":" << reference_rms
                      << ",\"row_maximum_error\":" << row_maximum
                      << ",\"row_rms_error\":" << row_rms
                      << ",\"event_ms_p50\":[";
            for (std::size_t index = 0; index < event_p50.size(); ++index) {
                if (index != 0) std::cout << ',';
                std::cout << event_p50[index];
            }
            std::cout << "]";
            if (!failure.empty()) std::cout << ",\"failure\":\"candidate_failed\"";
            std::cout << '}';
        }
        std::cout << "]"
                  << ",\"supported_count\":" << supported_count
                  << ",\"reference_pass_count\":" << reference_pass_count
                  << ",\"row_invariant_count\":" << invariant_count
                  << "}\n";
        if (supported_count == 0 || reference_pass_count == 0) {
            throw std::runtime_error("no BF16 candidate passed support/reference gates");
        }
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "benchmark_bf16_row_invariance: " << error.what() << '\n';
        return 2;
    }
}
