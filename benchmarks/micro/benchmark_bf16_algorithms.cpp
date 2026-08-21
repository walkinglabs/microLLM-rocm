#include <algorithm>
#include <cstdint>
#include <iostream>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <hipblaslt/hipblaslt-ext.hpp>
#include <hipblaslt/hipblaslt.h>

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
    Layout(hipDataType dtype, std::uint64_t rows, std::uint64_t columns,
           std::int64_t leading) {
        check(hipblasLtMatrixLayoutCreate(&value_, dtype, rows, columns,
                                          leading),
              "hipblasLtMatrixLayoutCreate");
    }
    ~Layout() { (void)hipblasLtMatrixLayoutDestroy(value_); }
    hipblasLtMatrixLayout_t get() const noexcept { return value_; }
private:
    hipblasLtMatrixLayout_t value_ = nullptr;
};

class Preference {
public:
    explicit Preference(std::uint64_t workspace_bytes) {
        check(hipblasLtMatmulPreferenceCreate(&value_),
              "hipblasLtMatmulPreferenceCreate");
        check(hipblasLtMatmulPreferenceSetAttribute(
                  value_, HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                  &workspace_bytes, sizeof(workspace_bytes)),
              "hipblasLtMatmulPreferenceSetAttribute");
    }
    ~Preference() { (void)hipblasLtMatmulPreferenceDestroy(value_); }
    hipblasLtMatmulPreference_t get() const noexcept { return value_; }
private:
    hipblasLtMatmulPreference_t value_ = nullptr;
};

struct Options {
    std::vector<std::int64_t> rows{32, 64};
    std::int64_t inner = 1536;
    std::int64_t columns = 8960;
    int max_algorithms = 64;
    std::uint64_t workspace_bytes = 32U * 1024U * 1024U;
};

std::vector<std::int64_t> positive_list(const std::string& text) {
    std::vector<std::int64_t> result;
    std::stringstream stream(text);
    std::string part;
    while (std::getline(stream, part, ',')) {
        const auto value = std::stoll(part);
        if (value <= 0) throw std::invalid_argument("rows must be positive");
        result.push_back(value);
    }
    if (result.empty()) throw std::invalid_argument("rows cannot be empty");
    return result;
}

Options options(int argc, char** argv) {
    Options result;
    for (int index = 1; index < argc; index += 2) {
        if (index + 1 >= argc) throw std::invalid_argument("missing CLI value");
        const std::string name = argv[index];
        if (name == "--rows") result.rows = positive_list(argv[index + 1]);
        else if (name == "--inner") result.inner = std::stoll(argv[index + 1]);
        else if (name == "--columns") result.columns = std::stoll(argv[index + 1]);
        else if (name == "--max-algorithms") {
            result.max_algorithms = std::stoi(argv[index + 1]);
        } else if (name == "--workspace-bytes") {
            result.workspace_bytes = std::stoull(argv[index + 1]);
        } else {
            throw std::invalid_argument("unknown CLI option: " + name);
        }
    }
    if (result.inner <= 0 || result.columns <= 0 || result.max_algorithms <= 0) {
        throw std::invalid_argument("shape and algorithm count must be positive");
    }
    return result;
}

struct Candidate {
    int index = -1;
    std::size_t workspace_bytes = 0;
    float waves = 0.0F;
};

std::vector<Candidate> query(Handle& handle, std::int64_t rows,
                             std::int64_t inner, std::int64_t columns,
                             int maximum, std::uint64_t workspace_bytes) {
    Description operation;
    Layout matrix_a(HIP_R_16BF, static_cast<std::uint64_t>(columns),
                    static_cast<std::uint64_t>(inner), columns);
    Layout matrix_b(HIP_R_16BF, static_cast<std::uint64_t>(inner),
                    static_cast<std::uint64_t>(rows), inner);
    Layout matrix_c(HIP_R_16BF, static_cast<std::uint64_t>(columns),
                    static_cast<std::uint64_t>(rows), columns);
    Preference preference(workspace_bytes);
    std::vector<hipblasLtMatmulHeuristicResult_t> results(
        static_cast<std::size_t>(maximum));
    int returned = 0;
    check(hipblasLtMatmulAlgoGetHeuristic(
              handle.get(), operation.get(), matrix_a.get(), matrix_b.get(),
              matrix_c.get(), matrix_c.get(), preference.get(), maximum,
              results.data(), &returned),
          "hipblasLtMatmulAlgoGetHeuristic");
    std::vector<Candidate> candidates;
    for (int index = 0; index < returned; ++index) {
        auto& result = results[static_cast<std::size_t>(index)];
        if (result.state != HIPBLAS_STATUS_SUCCESS) continue;
        candidates.push_back({hipblaslt_ext::getIndexFromAlgo(result.algo),
                              result.workspaceSize, result.wavesCount});
    }
    return candidates;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const auto command = options(argc, argv);
        Handle handle;
        std::vector<std::vector<Candidate>> all;
        for (const auto rows : command.rows) {
            all.push_back(query(handle, rows, command.inner, command.columns,
                                command.max_algorithms,
                                command.workspace_bytes));
        }
        std::set<int> common;
        if (!all.empty()) {
            for (const auto& candidate : all.front()) common.insert(candidate.index);
            for (std::size_t shape = 1; shape < all.size(); ++shape) {
                std::set<int> current;
                for (const auto& candidate : all[shape]) current.insert(candidate.index);
                std::set<int> intersection;
                std::set_intersection(common.begin(), common.end(), current.begin(),
                                      current.end(),
                                      std::inserter(intersection, intersection.begin()));
                common = std::move(intersection);
            }
        }
        std::cout << "{\"schema_version\":1,\"status\":\"pass\""
                  << ",\"record_type\":\"bf16_algorithm_inventory\""
                  << ",\"inner\":" << command.inner
                  << ",\"columns\":" << command.columns
                  << ",\"workspace_limit_bytes\":" << command.workspace_bytes
                  << ",\"requested_algorithms\":" << command.max_algorithms
                  << ",\"shapes\":[";
        for (std::size_t shape = 0; shape < all.size(); ++shape) {
            if (shape != 0) std::cout << ',';
            std::cout << "{\"rows\":" << command.rows[shape]
                      << ",\"candidate_count\":" << all[shape].size()
                      << ",\"candidates\":[";
            for (std::size_t index = 0; index < all[shape].size(); ++index) {
                if (index != 0) std::cout << ',';
                const auto& candidate = all[shape][index];
                std::cout << "{\"index\":" << candidate.index
                          << ",\"workspace_bytes\":"
                          << candidate.workspace_bytes
                          << ",\"waves\":" << candidate.waves << '}';
            }
            std::cout << "]}";
        }
        std::cout << "],\"common_indices\":[";
        std::size_t position = 0;
        for (const auto index : common) {
            if (position++ != 0) std::cout << ',';
            std::cout << index;
        }
        std::cout << "],\"common_candidate_count\":" << common.size()
                  << "}\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "microllm_bench_bf16_algorithms: " << error.what() << '\n';
        return 1;
    }
}
