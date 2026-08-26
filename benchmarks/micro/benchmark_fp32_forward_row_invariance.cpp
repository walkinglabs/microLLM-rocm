#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <hipblaslt/hipblaslt-ext.hpp>
#include <hipblaslt/hipblaslt.h>
#include <microllm/core/storage.h>
#include <microllm/ops/ops.h>
#include <microllm/runtime/runtime.h>

namespace {

void check(hipblasStatus_t status, const char* operation) {
    if (status != HIPBLAS_STATUS_SUCCESS) {
        throw std::runtime_error(std::string(operation) + " failed: " +
                                 std::to_string(static_cast<int>(status)));
    }
}

class Handle {
public:
    Handle() { check(hipblasLtCreate(&value_), "hipblasLtCreate"); }
    ~Handle() { (void)hipblasLtDestroy(value_); }
    hipblasLtHandle_t get() const noexcept { return value_; }
private:
    hipblasLtHandle_t value_ = nullptr;
};

class Description {
public:
    Description() {
        check(hipblasLtMatmulDescCreate(&value_, HIPBLAS_COMPUTE_32F,
                                        HIP_R_32F),
              "hipblasLtMatmulDescCreate");
    }
    ~Description() { (void)hipblasLtMatmulDescDestroy(value_); }
    hipblasLtMatmulDesc_t get() const noexcept { return value_; }
private:
    hipblasLtMatmulDesc_t value_ = nullptr;
};

class Layout {
public:
    Layout(std::uint64_t rows, std::uint64_t columns,
           std::int64_t leading) {
        check(hipblasLtMatrixLayoutCreate(
                  &value_, HIP_R_32F, rows, columns, leading),
              "hipblasLtMatrixLayoutCreate");
    }
    ~Layout() { (void)hipblasLtMatrixLayoutDestroy(value_); }
    hipblasLtMatrixLayout_t get() const noexcept { return value_; }
private:
    hipblasLtMatrixLayout_t value_ = nullptr;
};

class Preference {
public:
    explicit Preference(std::uint64_t bytes) {
        check(hipblasLtMatmulPreferenceCreate(&value_),
              "hipblasLtMatmulPreferenceCreate");
        check(hipblasLtMatmulPreferenceSetAttribute(
                  value_, HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                  &bytes, sizeof(bytes)), "set workspace preference");
    }
    ~Preference() { (void)hipblasLtMatmulPreferenceDestroy(value_); }
    hipblasLtMatmulPreference_t get() const noexcept { return value_; }
private:
    hipblasLtMatmulPreference_t value_ = nullptr;
};

struct Options {
    std::int64_t block_rows = 2048;
    std::vector<std::int64_t> multipliers{1, 2, 4, 8};
    std::int64_t inner = 1536;
    std::int64_t columns = 1536;
    int maximum_algorithms = 64;
    std::uint64_t workspace_bytes = 32U * 1024U * 1024U;
    int warmup = 1;
    int repetitions = 3;
};

std::vector<std::int64_t> positive_list(const std::string& text) {
    std::vector<std::int64_t> result;
    std::stringstream stream(text);
    std::string item;
    while (std::getline(stream, item, ',')) {
        const auto value = std::stoll(item);
        if (value <= 0) throw std::invalid_argument("multipliers must be positive");
        result.push_back(value);
    }
    if (result.empty()) throw std::invalid_argument("multipliers cannot be empty");
    return result;
}

Options options(int argc, char** argv) {
    Options result;
    for (int index = 1; index < argc; index += 2) {
        if (index + 1 >= argc) throw std::invalid_argument("missing CLI value");
        const std::string name = argv[index];
        if (name == "--block-rows") result.block_rows = std::stoll(argv[index + 1]);
        else if (name == "--multipliers") {
            result.multipliers = positive_list(argv[index + 1]);
        } else if (name == "--inner") result.inner = std::stoll(argv[index + 1]);
        else if (name == "--columns") result.columns = std::stoll(argv[index + 1]);
        else if (name == "--maximum-algorithms") {
            result.maximum_algorithms = std::stoi(argv[index + 1]);
        } else if (name == "--workspace-bytes") {
            result.workspace_bytes = std::stoull(argv[index + 1]);
        } else if (name == "--warmup") result.warmup = std::stoi(argv[index + 1]);
        else if (name == "--repetitions") {
            result.repetitions = std::stoi(argv[index + 1]);
        } else {
            throw std::invalid_argument("unknown CLI option: " + name);
        }
    }
    if (result.block_rows <= 0 || result.inner <= 0 || result.columns <= 0 ||
        result.block_rows * result.multipliers.back() > 16384 ||
        result.inner > 16384 || result.columns > 16384 ||
        result.maximum_algorithms <= 0 || result.maximum_algorithms > 128 ||
        result.warmup < 0 || result.repetitions <= 0 ||
        !std::is_sorted(result.multipliers.begin(), result.multipliers.end()) ||
        std::adjacent_find(result.multipliers.begin(), result.multipliers.end()) !=
            result.multipliers.end()) {
        throw std::invalid_argument("FP32 row-invariance options are outside the contract");
    }
    return result;
}

struct Candidate {
    int index = -1;
    std::size_t workspace_bytes = 0;
};

std::vector<Candidate> inventory(Handle& handle, std::int64_t rows,
                                 const Options& options) {
    Description operation;
    // row-major C=left*right is submitted as C^T=right^T*left^T.
    Layout matrix_b(static_cast<std::uint64_t>(options.columns),
                    static_cast<std::uint64_t>(options.inner), options.columns);
    Layout matrix_a(static_cast<std::uint64_t>(options.inner),
                    static_cast<std::uint64_t>(rows), options.inner);
    Layout matrix_c(static_cast<std::uint64_t>(options.columns),
                    static_cast<std::uint64_t>(rows), options.columns);
    Preference preference(options.workspace_bytes);
    std::vector<hipblasLtMatmulHeuristicResult_t> results(
        static_cast<std::size_t>(options.maximum_algorithms));
    int returned = 0;
    check(hipblasLtMatmulAlgoGetHeuristic(
              handle.get(), operation.get(), matrix_b.get(), matrix_a.get(),
              matrix_c.get(), matrix_c.get(), preference.get(),
              options.maximum_algorithms, results.data(), &returned),
          "hipblasLtMatmulAlgoGetHeuristic(FP32 forward)");
    std::vector<Candidate> output;
    for (int position = 0; position < returned; ++position) {
        auto& result = results[static_cast<std::size_t>(position)];
        if (result.state != HIPBLAS_STATUS_SUCCESS) continue;
        output.push_back({hipblaslt_ext::getIndexFromAlgo(result.algo),
                          result.workspaceSize});
    }
    return output;
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

Error compare(const std::vector<float>& left,
              const std::vector<float>& right) {
    if (left.empty() || left.size() != right.size()) {
        throw std::invalid_argument("FP32 block comparison changed size");
    }
    Error result;
    double square = 0.0;
    for (std::size_t index = 0; index < left.size(); ++index) {
        const auto delta = std::abs(left[index] - right[index]);
        result.maximum = std::max(result.maximum, delta);
        square += static_cast<double>(delta) * delta;
        result.exact = result.exact && left[index] == right[index];
    }
    result.rms = std::sqrt(square / static_cast<double>(left.size()));
    return result;
}

std::vector<float> sentinel_reference(
    const std::vector<float>& block, const std::vector<float>& weight,
    const Options& options, std::int64_t sample_rows,
    std::int64_t sample_columns) {
    std::vector<float> output(
        static_cast<std::size_t>(sample_rows * sample_columns));
    for (std::int64_t row = 0; row < sample_rows; ++row) {
        for (std::int64_t column = 0; column < sample_columns; ++column) {
            float sum = 0.0F;
            for (std::int64_t inner = 0; inner < options.inner; ++inner) {
                sum += block[static_cast<std::size_t>(row * options.inner + inner)] *
                       weight[static_cast<std::size_t>(inner * options.columns + column)];
            }
            output[static_cast<std::size_t>(row * sample_columns + column)] = sum;
        }
    }
    return output;
}

std::vector<float> sentinel_values(const microllm::Tensor& output,
                                   std::int64_t sample_rows,
                                   std::int64_t sample_columns) {
    const auto host = output.slice(0, 0, sample_rows).contiguous().to_vector();
    std::vector<float> selected;
    selected.reserve(static_cast<std::size_t>(sample_rows * sample_columns));
    const auto columns = output.shape()[1];
    for (std::int64_t row = 0; row < sample_rows; ++row) {
        selected.insert(selected.end(),
                        host.begin() + row * columns,
                        host.begin() + row * columns + sample_columns);
    }
    return selected;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const auto command = options(argc, argv);
        if (microllm::runtime::hip_device_count() == 0) {
            throw std::runtime_error("FP32 row-invariance search needs HIP");
        }
        const auto device = microllm::Device::hip(0);
        Handle handle;
        std::vector<std::vector<Candidate>> inventories;
        std::set<int> common;
        for (std::size_t shape = 0; shape < command.multipliers.size(); ++shape) {
            const auto rows = command.block_rows * command.multipliers[shape];
            inventories.push_back(inventory(handle, rows, command));
            std::set<int> current;
            for (const auto& candidate : inventories.back()) {
                current.insert(candidate.index);
            }
            if (shape == 0) common = std::move(current);
            else {
                std::set<int> intersection;
                std::set_intersection(common.begin(), common.end(),
                                      current.begin(), current.end(),
                                      std::inserter(intersection,
                                                    intersection.begin()));
                common = std::move(intersection);
            }
        }
        if (common.empty()) {
            throw std::runtime_error("FP32 forward shapes have no common solution");
        }

        std::vector<float> block_values(
            static_cast<std::size_t>(command.block_rows * command.inner));
        for (std::size_t index = 0; index < block_values.size(); ++index) {
            block_values[index] =
                static_cast<float>(static_cast<int>(index % 29U) - 14) / 127.0F;
        }
        std::vector<float> weight_values(
            static_cast<std::size_t>(command.inner * command.columns));
        for (std::size_t index = 0; index < weight_values.size(); ++index) {
            weight_values[index] =
                static_cast<float>(static_cast<int>(index % 31U) - 15) / 131.0F;
        }
        const auto sample_rows = std::min<std::int64_t>(8, command.block_rows);
        const auto sample_columns = std::min<std::int64_t>(16, command.columns);
        const auto cpu_sentinel = sentinel_reference(
            block_values, weight_values, command, sample_rows, sample_columns);
        const auto weight = microllm::Tensor::from_vector(
            weight_values, {command.inner, command.columns}).to(device);
        std::vector<microllm::Tensor> inputs;
        inputs.reserve(command.multipliers.size());
        for (const auto multiplier : command.multipliers) {
            std::vector<float> values(
                static_cast<std::size_t>(multiplier) * block_values.size());
            for (std::int64_t block = 0; block < multiplier; ++block) {
                std::copy(block_values.begin(), block_values.end(),
                          values.begin() + block * block_values.size());
            }
            inputs.push_back(microllm::Tensor::from_vector(
                values, {command.block_rows * multiplier, command.inner})
                                 .to(device));
        }
        microllm::Storage workspace(command.workspace_bytes, device);
        microllm::ops::OpContext context;
        context.workspace = workspace.data();
        context.workspace_bytes = command.workspace_bytes;
        context.mode = microllm::ops::OpMode::Inference;
        microllm::runtime::Event start(device);
        microllm::runtime::Event finish(device);

        std::size_t supported_count = 0;
        std::size_t sentinel_pass_count = 0;
        std::size_t invariant_count = 0;
        bool first = true;
        std::cout << std::setprecision(9)
                  << "{\"schema_version\":1,\"status\":\"pass\""
                  << ",\"record_type\":\"fp32_forward_row_invariance\""
                  << ",\"block_rows\":" << command.block_rows
                  << ",\"multipliers\":[";
        for (std::size_t index = 0; index < command.multipliers.size(); ++index) {
            if (index != 0) std::cout << ',';
            std::cout << command.multipliers[index];
        }
        std::cout << "]"
                  << ",\"inner\":" << command.inner
                  << ",\"columns\":" << command.columns
                  << ",\"workspace_limit_bytes\":" << command.workspace_bytes
                  << ",\"requested_algorithms\":" << command.maximum_algorithms
                  << ",\"shape_candidate_counts\":[";
        for (std::size_t index = 0; index < inventories.size(); ++index) {
            if (index != 0) std::cout << ',';
            std::cout << inventories[index].size();
        }
        std::cout << "]"
                  << ",\"common_candidate_count\":" << common.size()
                  << ",\"sentinel_elements\":" << cpu_sentinel.size()
                  << ",\"candidates\":[";

        for (const auto candidate : common) {
            bool supported = false;
            bool sentinel_passed = false;
            bool invariant = false;
            float sentinel_maximum = 0.0F;
            double sentinel_rms = 0.0;
            float block_maximum = 0.0F;
            double block_rms = 0.0;
            std::vector<double> event_p50;
            std::string failure;
            try {
                microllm::ops::clear_fp32_matmul_solution_registry();
                for (const auto& input : inputs) {
                    const auto key = microllm::ops::make_fp32_matmul_solution_key(
                        input.shape(), weight.shape(), device, false, false,
                        context);
                    microllm::ops::register_fp32_matmul_solution(key, candidate);
                }
                std::vector<float> reference_block;
                for (std::size_t shape = 0; shape < inputs.size(); ++shape) {
                    auto output = microllm::ops::matmul_with_implementation(
                        inputs[shape], weight,
                        microllm::ops::MatmulImplementation::HipBLASLt,
                        false, false, context);
                    microllm::runtime::synchronize(device);
                    const auto sentinel = sentinel_values(
                        output, sample_rows, sample_columns);
                    const auto sentinel_error = compare(cpu_sentinel, sentinel);
                    sentinel_maximum = std::max(
                        sentinel_maximum, sentinel_error.maximum);
                    sentinel_rms = std::max(sentinel_rms, sentinel_error.rms);
                    for (std::int64_t block = 0;
                         block < command.multipliers[shape]; ++block) {
                        const auto values = output.slice(
                            0, block * command.block_rows,
                            (block + 1) * command.block_rows).contiguous().to_vector();
                        if (shape == 0 && block == 0) {
                            reference_block = values;
                        } else {
                            const auto block_error = compare(reference_block, values);
                            block_maximum = std::max(
                                block_maximum, block_error.maximum);
                            block_rms = std::max(block_rms, block_error.rms);
                        }
                    }
                    for (int iteration = 0; iteration < command.warmup; ++iteration) {
                        output = microllm::ops::matmul_with_implementation(
                            inputs[shape], weight,
                            microllm::ops::MatmulImplementation::HipBLASLt,
                            false, false, context);
                    }
                    microllm::runtime::synchronize(device);
                    std::vector<double> times;
                    for (int iteration = 0; iteration < command.repetitions;
                         ++iteration) {
                        start.record_default_stream();
                        output = microllm::ops::matmul_with_implementation(
                            inputs[shape], weight,
                            microllm::ops::MatmulImplementation::HipBLASLt,
                            false, false, context);
                        finish.record_default_stream();
                        finish.synchronize();
                        times.push_back(finish.elapsed_ms_since(start));
                    }
                    event_p50.push_back(median(std::move(times)));
                }
                supported = true;
                sentinel_passed = sentinel_maximum <= 1.0e-4F &&
                                  sentinel_rms <= 1.0e-5;
                invariant = sentinel_passed && block_maximum == 0.0F;
            } catch (const std::exception& error) {
                failure = error.what();
            }
            supported_count += supported;
            sentinel_pass_count += sentinel_passed;
            invariant_count += invariant;
            std::size_t maximum_workspace = 0;
            for (const auto& inventory : inventories) {
                const auto found = std::find_if(
                    inventory.begin(), inventory.end(),
                    [candidate](const Candidate& item) {
                        return item.index == candidate;
                    });
                if (found != inventory.end()) {
                    maximum_workspace = std::max(
                        maximum_workspace, found->workspace_bytes);
                }
            }
            if (!first) std::cout << ',';
            first = false;
            std::cout << "{\"index\":" << candidate
                      << ",\"maximum_workspace_bytes\":" << maximum_workspace
                      << ",\"supported\":" << (supported ? "true" : "false")
                      << ",\"sentinel_passed\":"
                      << (sentinel_passed ? "true" : "false")
                      << ",\"block_invariant\":"
                      << (invariant ? "true" : "false")
                      << ",\"sentinel_maximum_error\":" << sentinel_maximum
                      << ",\"sentinel_rms_error\":" << sentinel_rms
                      << ",\"block_maximum_error\":" << block_maximum
                      << ",\"block_rms_error\":" << block_rms
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
                  << ",\"sentinel_pass_count\":" << sentinel_pass_count
                  << ",\"block_invariant_count\":" << invariant_count
                  << "}\n";
        if (supported_count == 0 || sentinel_pass_count == 0) {
            throw std::runtime_error("no FP32 forward candidate passed base gates");
        }
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "benchmark_fp32_forward_row_invariance: "
                  << error.what() << '\n';
        return 2;
    }
}
