#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <map>
#include <memory>
#include <set>
#include <span>
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
    explicit Description(bool transpose_right) {
        check(hipblasLtMatmulDescCreate(
                  &value_, HIPBLAS_COMPUTE_32F, HIP_R_32F),
              "hipblasLtMatmulDescCreate");
        const auto operation_a =
            transpose_right ? HIPBLAS_OP_T : HIPBLAS_OP_N;
        const auto operation_b = HIPBLAS_OP_N;
        check(hipblasLtMatmulDescSetAttribute(
                  value_, HIPBLASLT_MATMUL_DESC_TRANSA,
                  &operation_a, sizeof(operation_a)),
              "set TRANSA");
        check(hipblasLtMatmulDescSetAttribute(
                  value_, HIPBLASLT_MATMUL_DESC_TRANSB,
                  &operation_b, sizeof(operation_b)),
              "set TRANSB");
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
    void set_batch(std::int32_t count, std::int64_t stride) {
        check(hipblasLtMatrixLayoutSetAttribute(
                  value_, HIPBLASLT_MATRIX_LAYOUT_BATCH_COUNT,
                  &count, sizeof(count)), "set batch count");
        check(hipblasLtMatrixLayoutSetAttribute(
                  value_, HIPBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET,
                  &stride, sizeof(stride)), "set batch stride");
    }
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

std::vector<std::int64_t> positive_list(const std::string& text) {
    std::vector<std::int64_t> result;
    std::stringstream stream(text);
    std::string item;
    while (std::getline(stream, item, ',')) {
        const auto value = std::stoll(item);
        if (value <= 0) throw std::invalid_argument("list values must be positive");
        result.push_back(value);
    }
    if (result.empty() || !std::is_sorted(result.begin(), result.end()) ||
        std::adjacent_find(result.begin(), result.end()) != result.end()) {
        throw std::invalid_argument("list values must be sorted and unique");
    }
    return result;
}

std::vector<int> index_list(const std::string& text) {
    std::vector<int> result;
    if (text.empty()) return result;
    std::stringstream stream(text);
    std::string item;
    while (std::getline(stream, item, ',')) {
        const auto value = std::stoi(item);
        if (value < 0) throw std::invalid_argument("candidate indices must be nonnegative");
        result.push_back(value);
    }
    std::sort(result.begin(), result.end());
    result.erase(std::unique(result.begin(), result.end()), result.end());
    return result;
}

struct Options {
    std::string operation = "qk";
    std::int64_t sequence = 2048;
    std::int64_t heads = 12;
    std::int64_t kv_heads = 2;
    std::int64_t width = 128;
    std::vector<std::int64_t> request_batches{1, 2, 4, 8};
    std::vector<int> candidates;
    int maximum_algorithms = 64;
    std::uint64_t workspace_bytes = 32U * 1024U * 1024U;
    int warmup = 1;
    int repetitions = 3;
    bool inventory_only = false;
};

Options options(int argc, char** argv) {
    Options result;
    for (int index = 1; index < argc; index += 2) {
        if (index + 1 >= argc) throw std::invalid_argument("missing option value");
        const std::string name = argv[index];
        if (name == "--operation") result.operation = argv[index + 1];
        else if (name == "--sequence") result.sequence = std::stoll(argv[index + 1]);
        else if (name == "--heads") result.heads = std::stoll(argv[index + 1]);
        else if (name == "--kv-heads") result.kv_heads = std::stoll(argv[index + 1]);
        else if (name == "--width") result.width = std::stoll(argv[index + 1]);
        else if (name == "--request-batches") {
            result.request_batches = positive_list(argv[index + 1]);
        } else if (name == "--candidates") {
            result.candidates = index_list(argv[index + 1]);
        } else if (name == "--maximum-algorithms") {
            result.maximum_algorithms = std::stoi(argv[index + 1]);
        } else if (name == "--workspace-bytes") {
            result.workspace_bytes = std::stoull(argv[index + 1]);
        } else if (name == "--warmup") {
            result.warmup = std::stoi(argv[index + 1]);
        } else if (name == "--repetitions") {
            result.repetitions = std::stoi(argv[index + 1]);
        } else if (name == "--inventory-only") {
            const std::string value = argv[index + 1];
            if (value != "true" && value != "false") {
                throw std::invalid_argument(
                    "inventory-only must be true or false");
            }
            result.inventory_only = value == "true";
        } else {
            throw std::invalid_argument("unknown option: " + name);
        }
    }
    if ((result.operation != "qk" && result.operation != "pv") ||
        result.sequence <= 0 || result.sequence > 4096 ||
        result.heads <= 0 || result.kv_heads <= 0 ||
        result.heads % result.kv_heads != 0 || result.width <= 0 ||
        result.width > 256 || result.request_batches.back() > 8 ||
        result.maximum_algorithms <= 0 || result.maximum_algorithms > 256 ||
        result.warmup < 0 || result.repetitions <= 0) {
        throw std::invalid_argument(
            "FP32 Attention batch-invariance options are outside the contract");
    }
    return result;
}

struct Problem {
    std::int64_t batch_count = 0;
    std::int64_t left_rows = 0;
    std::int64_t left_columns = 0;
    std::int64_t right_rows = 0;
    std::int64_t right_columns = 0;
    std::int64_t output_columns = 0;
    bool transpose_right = false;
};

Problem problem(const Options& command, std::int64_t request_batch) {
    if (command.operation == "qk") {
        return {request_batch * command.heads, command.sequence, command.width,
                command.sequence, command.width, command.sequence, true};
    }
    return {request_batch * command.heads, command.sequence, command.sequence,
            command.sequence, command.width, command.width, false};
}

struct Candidate {
    int index = -1;
    std::size_t workspace_bytes = 0;
    hipblasLtMatmulAlgo_t algorithm{};
};

class Plan {
public:
    Plan(const Problem& shape, std::uint64_t workspace_bytes)
        : shape_(shape), operation_(shape.transpose_right),
          matrix_b_(static_cast<std::uint64_t>(shape.right_columns),
                    static_cast<std::uint64_t>(shape.right_rows),
                    shape.right_columns),
          matrix_a_(static_cast<std::uint64_t>(shape.left_columns),
                    static_cast<std::uint64_t>(shape.left_rows),
                    shape.left_columns),
          matrix_c_(static_cast<std::uint64_t>(shape.output_columns),
                    static_cast<std::uint64_t>(shape.left_rows),
                    shape.output_columns), preference_(workspace_bytes) {
        const auto count = static_cast<std::int32_t>(shape.batch_count);
        matrix_b_.set_batch(
            count, shape.right_rows * shape.right_columns);
        matrix_a_.set_batch(
            count, shape.left_rows * shape.left_columns);
        matrix_c_.set_batch(
            count, shape.left_rows * shape.output_columns);
    }

    const Problem& shape() const noexcept { return shape_; }
    hipblasLtMatmulDesc_t operation() const noexcept { return operation_.get(); }
    hipblasLtMatrixLayout_t matrix_b() const noexcept { return matrix_b_.get(); }
    hipblasLtMatrixLayout_t matrix_a() const noexcept { return matrix_a_.get(); }
    hipblasLtMatrixLayout_t matrix_c() const noexcept { return matrix_c_.get(); }
    hipblasLtMatmulPreference_t preference() const noexcept {
        return preference_.get();
    }
private:
    Problem shape_;
    Description operation_;
    Layout matrix_b_;
    Layout matrix_a_;
    Layout matrix_c_;
    Preference preference_;
};

std::map<int, Candidate> inventory(
    Handle& handle, Plan& plan, int maximum) {
    std::vector<hipblasLtMatmulHeuristicResult_t> heuristic(
        static_cast<std::size_t>(maximum));
    int returned = 0;
    check(hipblasLtMatmulAlgoGetHeuristic(
              handle.get(), plan.operation(), plan.matrix_b(), plan.matrix_a(),
              plan.matrix_c(), plan.matrix_c(), plan.preference(), maximum,
              heuristic.data(), &returned),
          "hipblasLtMatmulAlgoGetHeuristic(Attention batch invariance)");
    std::map<int, Candidate> output;
    for (int position = 0; position < returned; ++position) {
        auto& result = heuristic[static_cast<std::size_t>(position)];
        if (result.state != HIPBLAS_STATUS_SUCCESS) continue;
        const auto index = hipblaslt_ext::getIndexFromAlgo(result.algo);
        output.try_emplace(
            index, Candidate{index, result.workspaceSize, result.algo});
    }
    return output;
}

struct Error {
    float maximum = 0.0F;
    double rms = 0.0;
    bool exact = true;
};

Error compare(std::span<const float> left, std::span<const float> right) {
    if (left.empty() || left.size() != right.size()) {
        throw std::invalid_argument("Attention comparison size changed");
    }
    Error result;
    double squared = 0.0;
    for (std::size_t index = 0; index < left.size(); ++index) {
        if (!std::isfinite(left[index]) || !std::isfinite(right[index])) {
            throw std::runtime_error("Attention output contains non-finite values");
        }
        const auto delta = std::abs(left[index] - right[index]);
        result.maximum = std::max(result.maximum, delta);
        squared += static_cast<double>(delta) * delta;
        result.exact = result.exact && left[index] == right[index];
    }
    result.rms = std::sqrt(squared / static_cast<double>(left.size()));
    return result;
}

double median(std::vector<double> values) {
    std::sort(values.begin(), values.end());
    const auto middle = values.size() / 2U;
    return values.size() % 2U == 0U
               ? (values[middle - 1U] + values[middle]) / 2.0
               : values[middle];
}

struct Timing {
    double event_ms = 0.0;
    double wall_ms = 0.0;
};

std::vector<float> qk_left_values(const Options& command) {
    std::vector<float> values(static_cast<std::size_t>(
        command.heads * command.sequence * command.width));
    const auto scale = 1.0F / std::sqrt(static_cast<float>(command.width));
    for (std::int64_t head = 0; head < command.heads; ++head) {
        for (std::int64_t row = 0; row < command.sequence; ++row) {
            for (std::int64_t column = 0; column < command.width; ++column) {
                const auto index = static_cast<std::size_t>(
                    (head * command.sequence + row) * command.width + column);
                values[index] = scale * static_cast<float>(
                    ((head * 13 + row * 7 + column * 3) % 61) - 30) / 31.0F;
            }
        }
    }
    return values;
}

std::vector<float> repeated_kv_values(const Options& command) {
    std::vector<float> values(static_cast<std::size_t>(
        command.heads * command.sequence * command.width));
    const auto repeats = command.heads / command.kv_heads;
    for (std::int64_t head = 0; head < command.heads; ++head) {
        const auto kv_head = head / repeats;
        for (std::int64_t row = 0; row < command.sequence; ++row) {
            for (std::int64_t column = 0; column < command.width; ++column) {
                const auto index = static_cast<std::size_t>(
                    (head * command.sequence + row) * command.width + column);
                values[index] = static_cast<float>(
                    ((kv_head * 17 + row * 5 + column * 11) % 67) - 33) /
                    37.0F;
            }
        }
    }
    return values;
}

std::vector<float> probability_values(const Options& command) {
    std::vector<float> values(static_cast<std::size_t>(
        command.heads * command.sequence * command.sequence), 0.0F);
    for (std::int64_t head = 0; head < command.heads; ++head) {
        for (std::int64_t row = 0; row < command.sequence; ++row) {
            float denominator = 0.0F;
            for (std::int64_t source = 0; source <= row; ++source) {
                denominator += static_cast<float>(
                    1 + ((head * 7 + row * 3 + source * 5) % 19));
            }
            for (std::int64_t source = 0; source <= row; ++source) {
                const auto index = static_cast<std::size_t>(
                    (head * command.sequence + row) * command.sequence + source);
                values[index] = static_cast<float>(
                    1 + ((head * 7 + row * 3 + source * 5) % 19)) /
                    denominator;
            }
        }
    }
    return values;
}

std::vector<float> sentinel_reference(
    const Problem& shape, const std::vector<float>& left,
    const std::vector<float>& right,
    std::int64_t sample_heads, std::int64_t sample_rows,
    std::int64_t sample_columns) {
    std::vector<float> output(static_cast<std::size_t>(
        sample_heads * sample_rows * sample_columns));
    for (std::int64_t head = 0; head < sample_heads; ++head) {
        for (std::int64_t row = 0; row < sample_rows; ++row) {
            for (std::int64_t column = 0; column < sample_columns; ++column) {
                float total = 0.0F;
                for (std::int64_t inner = 0; inner < shape.left_columns; ++inner) {
                    const auto left_index = static_cast<std::size_t>(
                        (head * shape.left_rows + row) * shape.left_columns + inner);
                    const auto right_index = static_cast<std::size_t>(
                        shape.transpose_right
                            ? (head * shape.right_rows + column) *
                                  shape.right_columns + inner
                            : (head * shape.right_rows + inner) *
                                  shape.right_columns + column);
                    total += left[left_index] * right[right_index];
                }
                output[static_cast<std::size_t>(
                    (head * sample_rows + row) * sample_columns + column)] = total;
            }
        }
    }
    return output;
}

std::vector<float> sentinel_values(
    std::span<const float> output, const Problem& shape,
    std::int64_t sample_heads, std::int64_t sample_rows,
    std::int64_t sample_columns) {
    std::vector<float> selected;
    selected.reserve(static_cast<std::size_t>(
        sample_heads * sample_rows * sample_columns));
    for (std::int64_t head = 0; head < sample_heads; ++head) {
        for (std::int64_t row = 0; row < sample_rows; ++row) {
            const auto offset = static_cast<std::size_t>(
                (head * shape.left_rows + row) * shape.output_columns);
            selected.insert(selected.end(), output.begin() + offset,
                            output.begin() + offset + sample_columns);
        }
    }
    return selected;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const auto command = options(argc, argv);
        if (microllm::runtime::hip_device_count() == 0) {
            throw std::runtime_error(
                "FP32 Attention batch invariance requires a visible HIP GPU");
        }
        const auto device = microllm::Device::hip(0);
        Handle handle;
        std::vector<std::unique_ptr<Plan>> plans;
        std::vector<std::map<int, Candidate>> inventories;
        std::set<int> common;
        for (std::size_t index = 0; index < command.request_batches.size(); ++index) {
            plans.push_back(std::make_unique<Plan>(
                problem(command, command.request_batches[index]),
                command.workspace_bytes));
            inventories.push_back(inventory(
                handle, *plans.back(), command.maximum_algorithms));
            std::set<int> current;
            for (const auto& [candidate, unused] : inventories.back()) {
                (void)unused;
                current.insert(candidate);
            }
            if (index == 0) common = std::move(current);
            else {
                std::set<int> intersection;
                std::set_intersection(
                    common.begin(), common.end(), current.begin(), current.end(),
                    std::inserter(intersection, intersection.begin()));
                common = std::move(intersection);
            }
        }
        if (!command.candidates.empty()) {
            std::set<int> requested(
                command.candidates.begin(), command.candidates.end());
            std::set<int> intersection;
            std::set_intersection(
                common.begin(), common.end(), requested.begin(), requested.end(),
                std::inserter(intersection, intersection.begin()));
            common = std::move(intersection);
        }
        if (common.empty()) {
            throw std::runtime_error(
                "FP32 Attention request batches have no common solution");
        }

        if (command.inventory_only) {
            const auto shape = plans.front()->shape();
            std::cout << "{\"schema_version\":1,\"status\":\"pass\""
                      << ",\"record_type\":\"fp32_attention_batch_inventory\""
                      << ",\"operation\":\"" << command.operation << "\""
                      << ",\"sequence\":" << command.sequence
                      << ",\"heads\":" << command.heads
                      << ",\"kv_heads\":" << command.kv_heads
                      << ",\"width\":" << command.width
                      << ",\"m\":" << shape.left_rows
                      << ",\"k\":" << shape.left_columns
                      << ",\"n\":" << shape.output_columns
                      << ",\"transpose_right\":"
                      << (shape.transpose_right ? "true" : "false")
                      << ",\"request_batches\":[";
            for (std::size_t index = 0;
                 index < command.request_batches.size(); ++index) {
                if (index != 0) std::cout << ',';
                std::cout << command.request_batches[index];
            }
            std::cout << "],\"shape_candidate_counts\":[";
            for (std::size_t index = 0; index < inventories.size(); ++index) {
                if (index != 0) std::cout << ',';
                std::cout << inventories[index].size();
            }
            std::cout << "],\"common_candidate_indices\":[";
            bool first = true;
            for (const auto candidate : common) {
                if (!first) std::cout << ',';
                first = false;
                std::cout << candidate;
            }
            std::cout << "]}\n";
            return 0;
        }

        const auto b1_shape = plans.front()->shape();
        const auto left_host = command.operation == "qk"
                                   ? qk_left_values(command)
                                   : probability_values(command);
        const auto right_host = repeated_kv_values(command);
        const auto left_b1 = microllm::Tensor::from_vector(
            left_host,
            {1, command.heads, b1_shape.left_rows, b1_shape.left_columns})
                                 .to(device);
        const auto right_b1 = microllm::Tensor::from_vector(
            right_host,
            {1, command.heads, b1_shape.right_rows, b1_shape.right_columns})
                                  .to(device);
        const auto maximum_requests = command.request_batches.back();
        const auto left = microllm::ops::repeat_interleave(
                              left_b1, 0, maximum_requests)
                              .reshape({maximum_requests * command.heads,
                                        b1_shape.left_rows,
                                        b1_shape.left_columns});
        const auto right = microllm::ops::repeat_interleave(
                               right_b1, 0, maximum_requests)
                               .reshape({maximum_requests * command.heads,
                                         b1_shape.right_rows,
                                         b1_shape.right_columns});
        microllm::Tensor output(
            {maximum_requests * command.heads, b1_shape.left_rows,
             b1_shape.output_columns},
            microllm::DType::Float32, device);
        microllm::Storage workspace(command.workspace_bytes, device);
        const float alpha = 1.0F;
        const float beta = 0.0F;
        const auto submit = [&](Plan& plan,
                                const hipblasLtMatmulAlgo_t* algorithm) {
            check(hipblasLtMatmul(
                      handle.get(), plan.operation(), &alpha,
                      right.data(), plan.matrix_b(), left.data(), plan.matrix_a(),
                      &beta, output.data(), plan.matrix_c(), output.data(),
                      plan.matrix_c(), algorithm, workspace.data(),
                      workspace.num_bytes(), nullptr),
                  "hipblasLtMatmul(Attention batch invariance)");
        };
        microllm::runtime::Event start(device);
        microllm::runtime::Event finish(device);
        const auto time = [&](Plan& plan,
                              const hipblasLtMatmulAlgo_t* algorithm) {
            for (int iteration = 0; iteration < command.warmup; ++iteration) {
                submit(plan, algorithm);
            }
            microllm::runtime::synchronize(device);
            std::vector<double> event;
            std::vector<double> wall;
            for (int iteration = 0; iteration < command.repetitions; ++iteration) {
                const auto wall_start = std::chrono::steady_clock::now();
                start.record_default_stream();
                submit(plan, algorithm);
                finish.record_default_stream();
                finish.synchronize();
                const auto wall_finish = std::chrono::steady_clock::now();
                event.push_back(finish.elapsed_ms_since(start));
                wall.push_back(std::chrono::duration<double, std::milli>(
                    wall_finish - wall_start).count());
            }
            return Timing{median(event), median(wall)};
        };

        std::vector<Timing> default_timings;
        std::vector<float> default_b1;
        float default_block_maximum = 0.0F;
        double default_block_rms = 0.0;
        bool default_block_invariant = true;
        for (std::size_t shape = 0; shape < plans.size(); ++shape) {
            auto& plan = *plans[shape];
            submit(plan, nullptr);
            microllm::runtime::synchronize(device);
            const auto request_batch = command.request_batches[shape];
            const auto actual = output.slice(
                0, 0, request_batch * command.heads)
                                    .contiguous().to_vector();
            if (shape == 0) {
                const auto elements = static_cast<std::size_t>(
                    command.heads * b1_shape.left_rows *
                    b1_shape.output_columns);
                default_b1.assign(
                    actual.begin(), actual.begin() +
                        static_cast<std::ptrdiff_t>(elements));
                if (default_b1.size() != elements) {
                    throw std::logic_error("default B1 output size changed");
                }
            }
            const auto block_elements = default_b1.size();
            if (actual.size() != block_elements *
                                     static_cast<std::size_t>(request_batch)) {
                throw std::logic_error("default output size changed");
            }
            for (std::int64_t request = 0; request < request_batch; ++request) {
                const auto offset = static_cast<std::size_t>(request) *
                                    block_elements;
                const auto error = compare(
                    default_b1, std::span<const float>(actual).subspan(
                                    offset, block_elements));
                default_block_maximum = std::max(
                    default_block_maximum, error.maximum);
                default_block_rms = std::max(default_block_rms, error.rms);
                default_block_invariant = default_block_invariant && error.exact;
            }
            default_timings.push_back(time(plan, nullptr));
        }
        const auto sample_heads = std::min<std::int64_t>(2, command.heads);
        const auto sample_rows = std::min<std::int64_t>(2, command.sequence);
        const auto sample_columns = std::min<std::int64_t>(8, b1_shape.output_columns);
        const auto cpu_sentinel = sentinel_reference(
            b1_shape, left_host, right_host,
            sample_heads, sample_rows, sample_columns);

        std::size_t correctness_count = 0;
        std::size_t invariant_count = 0;
        bool first_candidate = true;
        std::cout << std::setprecision(12)
                  << "{\"schema_version\":1,\"status\":\"pass\""
                  << ",\"record_type\":\"fp32_attention_batch_invariance\""
                  << ",\"operation\":\"" << command.operation << "\""
                  << ",\"sequence\":" << command.sequence
                  << ",\"heads\":" << command.heads
                  << ",\"kv_heads\":" << command.kv_heads
                  << ",\"width\":" << command.width
                  << ",\"m\":" << b1_shape.left_rows
                  << ",\"k\":" << b1_shape.left_columns
                  << ",\"n\":" << b1_shape.output_columns
                  << ",\"transpose_right\":"
                  << (b1_shape.transpose_right ? "true" : "false")
                  << ",\"request_batches\":[";
        for (std::size_t index = 0; index < command.request_batches.size(); ++index) {
            if (index != 0) std::cout << ',';
            std::cout << command.request_batches[index];
        }
        std::cout << "],\"backend_batch_counts\":[";
        for (std::size_t index = 0; index < plans.size(); ++index) {
            if (index != 0) std::cout << ',';
            std::cout << plans[index]->shape().batch_count;
        }
        std::cout << "],\"shape_candidate_counts\":[";
        for (std::size_t index = 0; index < inventories.size(); ++index) {
            if (index != 0) std::cout << ',';
            std::cout << inventories[index].size();
        }
        std::cout << "],\"common_candidate_count\":" << common.size()
                  << ",\"workspace_limit_bytes\":" << command.workspace_bytes
                  << ",\"complete_b1_output_elements\":" << default_b1.size()
                  << ",\"cpu_sentinel_elements\":" << cpu_sentinel.size()
                  << ",\"default_block_invariant\":"
                  << (default_block_invariant ? "true" : "false")
                  << ",\"default_block_maximum_error\":"
                  << default_block_maximum
                  << ",\"default_block_rms_error\":" << default_block_rms
                  << ",\"default_event_ms_p50\":[";
        for (std::size_t index = 0; index < default_timings.size(); ++index) {
            if (index != 0) std::cout << ',';
            std::cout << default_timings[index].event_ms;
        }
        std::cout << "],\"default_wall_ms_p50\":[";
        for (std::size_t index = 0; index < default_timings.size(); ++index) {
            if (index != 0) std::cout << ',';
            std::cout << default_timings[index].wall_ms;
        }
        std::cout << "],\"candidates\":[";

        for (const auto candidate_index : common) {
            bool supported = false;
            bool finite = true;
            bool correctness = false;
            bool invariant = false;
            std::size_t maximum_workspace = 0;
            float default_maximum = 0.0F;
            double default_rms = 0.0;
            float sentinel_maximum = 0.0F;
            double sentinel_rms = 0.0;
            float block_maximum = 0.0F;
            double block_rms = 0.0;
            std::vector<Timing> timings;
            std::vector<double> event_speedups;
            std::string failure;
            try {
                std::vector<float> candidate_b1;
                for (std::size_t shape = 0; shape < plans.size(); ++shape) {
                    const auto found = inventories[shape].find(candidate_index);
                    if (found == inventories[shape].end()) {
                        throw std::runtime_error("candidate missing from a shape");
                    }
                    maximum_workspace = std::max(
                        maximum_workspace, found->second.workspace_bytes);
                    submit(*plans[shape], &found->second.algorithm);
                    microllm::runtime::synchronize(device);
                    const auto request_batch = command.request_batches[shape];
                    const auto actual = output.slice(
                        0, 0, request_batch * command.heads)
                                            .contiguous().to_vector();
                    const auto block_elements = default_b1.size();
                    if (actual.size() != block_elements *
                                             static_cast<std::size_t>(request_batch)) {
                        throw std::logic_error("candidate output size changed");
                    }
                    if (shape == 0) {
                        candidate_b1.assign(
                            actual.begin(), actual.begin() +
                                static_cast<std::ptrdiff_t>(block_elements));
                        const auto default_error = compare(default_b1, candidate_b1);
                        default_maximum = default_error.maximum;
                        default_rms = default_error.rms;
                        const auto candidate_sentinel = sentinel_values(
                            candidate_b1, b1_shape, sample_heads,
                            sample_rows, sample_columns);
                        const auto sentinel_error = compare(
                            cpu_sentinel, candidate_sentinel);
                        sentinel_maximum = sentinel_error.maximum;
                        sentinel_rms = sentinel_error.rms;
                    }
                    for (std::int64_t request = 0;
                         request < request_batch; ++request) {
                        const auto offset = static_cast<std::size_t>(request) *
                                            block_elements;
                        const auto block_error = compare(
                            candidate_b1,
                            std::span<const float>(actual).subspan(
                                offset, block_elements));
                        block_maximum = std::max(
                            block_maximum, block_error.maximum);
                        block_rms = std::max(block_rms, block_error.rms);
                    }
                }
                supported = true;
                correctness = finite && default_maximum <= 0.05F &&
                              default_rms <= 0.005 &&
                              sentinel_maximum <= 0.05F &&
                              sentinel_rms <= 0.005;
                invariant = correctness && block_maximum == 0.0F &&
                            block_rms == 0.0;
                if (correctness) ++correctness_count;
                if (invariant) {
                    ++invariant_count;
                    for (std::size_t shape = 0; shape < plans.size(); ++shape) {
                        const auto& algorithm =
                            inventories[shape].at(candidate_index).algorithm;
                        timings.push_back(time(*plans[shape], &algorithm));
                        event_speedups.push_back(
                            default_timings[shape].event_ms /
                            timings.back().event_ms);
                    }
                } else {
                    failure = correctness
                                  ? "cross-batch block invariance failed"
                                  : "complete correctness gate failed";
                }
            } catch (const std::exception& error) {
                finite = false;
                failure = error.what();
            }
            if (!first_candidate) std::cout << ',';
            first_candidate = false;
            std::cout << "{\"index\":" << candidate_index
                      << ",\"maximum_workspace_bytes\":" << maximum_workspace
                      << ",\"supported\":" << (supported ? "true" : "false")
                      << ",\"finite\":" << (finite ? "true" : "false")
                      << ",\"correctness_passed\":"
                      << (correctness ? "true" : "false")
                      << ",\"block_invariant\":"
                      << (invariant ? "true" : "false")
                      << ",\"default_maximum_error\":" << default_maximum
                      << ",\"default_rms_error\":" << default_rms
                      << ",\"sentinel_maximum_error\":" << sentinel_maximum
                      << ",\"sentinel_rms_error\":" << sentinel_rms
                      << ",\"block_maximum_error\":" << block_maximum
                      << ",\"block_rms_error\":" << block_rms
                      << ",\"event_ms_p50\":[";
            for (std::size_t index = 0; index < timings.size(); ++index) {
                if (index != 0) std::cout << ',';
                std::cout << timings[index].event_ms;
            }
            std::cout << "],\"wall_ms_p50\":[";
            for (std::size_t index = 0; index < timings.size(); ++index) {
                if (index != 0) std::cout << ',';
                std::cout << timings[index].wall_ms;
            }
            std::cout << "],\"event_speedup_vs_default\":[";
            for (std::size_t index = 0; index < event_speedups.size(); ++index) {
                if (index != 0) std::cout << ',';
                std::cout << event_speedups[index];
            }
            std::cout << "],\"failure\":\"" << failure << "\"}";
        }
        std::cout << "],\"correctness_passed_count\":" << correctness_count
                  << ",\"block_invariant_count\":" << invariant_count
                  << "}\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "benchmark_fp32_attention_batch_invariance: "
                  << error.what() << '\n';
        return 1;
    }
}
