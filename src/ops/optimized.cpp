#include <microllm/ops/ops.h>
#include <microllm/runtime/memory.h>

#include <algorithm>
#include <atomic>
#include <bit>
#include <charconv>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <set>
#include <tuple>
#include <vector>

#if MICROLLM_HAS_HIP
#include "hip/kernels.h"
#endif

#if MICROLLM_HAS_HIPBLASLT
#include <hipblaslt/hipblaslt-ext.hpp>
#include <hipblaslt/hipblaslt.h>
#include <hipblaslt/hipblaslt-version.h>
#endif

namespace microllm::ops {

namespace {
using MatmulShapeKey = std::tuple<std::int64_t, std::int64_t, std::int64_t>;
std::mutex registry_mutex;
std::map<MatmulTuningKey, MatmulImplementation> registry;
std::atomic<std::size_t> registry_entries{0};
thread_local std::map<Fp32MatmulSolutionKey, int> fp32_solution_registry;
thread_local std::size_t fp32_solution_registry_hits = 0;
thread_local std::size_t fp32_solution_registry_misses = 0;
thread_local std::size_t fp32_solution_cache_hits = 0;
thread_local std::size_t fp32_solution_cache_misses = 0;
thread_local std::size_t fp32_solution_dispatches = 0;
thread_local std::map<Bf16GroupedQkvKey, int> bf16_grouped_qkv_registry;
thread_local std::size_t bf16_grouped_qkv_plan_hits = 0;
thread_local std::size_t bf16_grouped_qkv_plan_misses = 0;
thread_local std::size_t bf16_grouped_qkv_dispatches = 0;
thread_local std::size_t bf16_grouped_qkv_retained_query_key_dispatches = 0;
thread_local std::size_t bf16_grouped_qkv_algorithm_hits = 0;
thread_local std::size_t bf16_grouped_qkv_algorithm_misses = 0;
thread_local std::size_t bf16_grouped_qkv_kernel_hits = 0;
thread_local std::size_t bf16_grouped_qkv_kernel_misses = 0;
thread_local double bf16_grouped_qkv_kernel_setup_ms = 0.0;
thread_local double bf16_grouped_qkv_argument_setup_ms = 0.0;
thread_local std::map<Bf16GroupedGateUpKey, int>
    bf16_grouped_gate_up_registry;
thread_local std::size_t bf16_grouped_gate_up_plan_hits = 0;
thread_local std::size_t bf16_grouped_gate_up_plan_misses = 0;
thread_local std::size_t bf16_grouped_gate_up_dispatches = 0;
thread_local std::size_t bf16_grouped_gate_up_algorithm_hits = 0;
thread_local std::size_t bf16_grouped_gate_up_algorithm_misses = 0;
thread_local std::size_t bf16_grouped_gate_up_kernel_hits = 0;
thread_local std::size_t bf16_grouped_gate_up_kernel_misses = 0;
thread_local double bf16_grouped_gate_up_kernel_setup_ms = 0.0;
thread_local double bf16_grouped_gate_up_argument_setup_ms = 0.0;
thread_local bool bf16_grouped_gate_up_swish = false;

struct TuningEnvironment {
    std::string architecture;
    int runtime_version = 0;
    int driver_version = 0;
};

Shape matmul_output_shape(const Tensor& left, const Tensor& right,
                          bool transpose_left, bool transpose_right) {
    if (left.ndim() < 2 || right.ndim() != left.ndim() ||
        !is_floating_point(left.dtype()) || right.dtype() != left.dtype() ||
        left.device() != right.device() || !left.is_contiguous() ||
        !right.is_contiguous()) {
        throw std::invalid_argument(
            "matmul output requires matching-rank contiguous floating tensors");
    }
    const auto rank = static_cast<std::size_t>(left.ndim());
    Shape output_shape(left.shape().begin(), left.shape().end() - 2);
    for (std::size_t dimension = 0; dimension + 2 < rank; ++dimension) {
        if (left.shape()[dimension] != right.shape()[dimension]) {
            throw std::invalid_argument("matmul batch dimensions must match");
        }
    }
    const auto left_rows = left.shape()[rank - 2];
    const auto left_columns = left.shape()[rank - 1];
    const auto right_rows = right.shape()[rank - 2];
    const auto right_columns = right.shape()[rank - 1];
    const auto rows = transpose_left ? left_columns : left_rows;
    const auto inner = transpose_left ? left_rows : left_columns;
    const auto right_inner = transpose_right ? right_columns : right_rows;
    const auto columns = transpose_right ? right_rows : right_columns;
    if (right_inner != inner) throw std::invalid_argument("matmul inner dimensions mismatch");
    output_shape.push_back(rows);
    output_shape.push_back(columns);
    return output_shape;
}

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

constexpr int kMatmulTuningCacheSchema = 1;

std::int64_t checked_positive_product(
    std::int64_t left, std::int64_t right, const char* name) {
    if (left <= 0 || right <= 0 ||
        left > std::numeric_limits<std::int64_t>::max() / right) {
        throw std::invalid_argument(std::string(name) +
                                    " must be a positive int64 product");
    }
    return left * right;
}

void validate_matmul_tuning_key(const MatmulTuningKey& key) {
    if (key.rows <= 0 || key.inner <= 0 || key.columns <= 0) {
        throw std::invalid_argument("registered matmul dimensions must be positive");
    }
    if (!is_floating_point(key.dtype) || key.left_strides.size() != 2 ||
        key.right_strides.size() != 2 ||
        std::any_of(key.left_strides.begin(), key.left_strides.end(),
                    [](std::int64_t value) { return value <= 0; }) ||
        std::any_of(key.right_strides.begin(), key.right_strides.end(),
                    [](std::int64_t value) { return value <= 0; }) ||
        key.architecture.empty() || key.hip_runtime_version < 0 ||
        key.hip_driver_version < 0 || key.hipblaslt_version < 0) {
        throw std::invalid_argument("registered matmul tuning key is incomplete");
    }
}

void validate_fp32_solution_key(const Fp32MatmulSolutionKey& key) {
    const auto alpha = std::bit_cast<float>(key.alpha_bits);
    if (key.batches <= 0 ||
        key.batches > std::numeric_limits<std::int32_t>::max() ||
        key.left_rows <= 0 || key.left_columns <= 0 ||
        key.right_rows <= 0 || key.right_columns <= 0 ||
        key.output_rows <= 0 || key.output_columns <= 0 ||
        !std::isfinite(alpha) || key.architecture.empty() ||
        key.hip_runtime_version < 0 || key.hip_driver_version < 0 ||
        key.hipblaslt_version <= 0) {
        throw std::invalid_argument("FP32 solution key is incomplete");
    }
    const auto rows = key.transpose_left ? key.left_columns : key.left_rows;
    const auto inner = key.transpose_left ? key.left_rows : key.left_columns;
    const auto right_inner =
        key.transpose_right ? key.right_columns : key.right_rows;
    const auto columns =
        key.transpose_right ? key.right_rows : key.right_columns;
    if (inner != right_inner || key.output_rows != rows ||
        key.output_columns != columns ||
        key.left_batch_stride != checked_positive_product(
                                     key.left_rows, key.left_columns,
                                     "FP32 left batch stride") ||
        key.right_batch_stride != checked_positive_product(
                                      key.right_rows, key.right_columns,
                                      "FP32 right batch stride") ||
        key.output_batch_stride != checked_positive_product(
                                       key.output_rows, key.output_columns,
                                       "FP32 output batch stride")) {
        throw std::invalid_argument(
            "FP32 solution key does not describe one exact contiguous GEMM");
    }
}

void validate_bf16_grouped_qkv_key(const Bf16GroupedQkvKey& key) {
    if (key.rows <= 0 || key.inner <= 0 || key.query_columns <= 0 ||
        key.key_columns <= 0 || key.value_columns <= 0 ||
        key.architecture.empty() || key.hip_runtime_version < 0 ||
        key.hip_driver_version < 0 || key.hipblaslt_version <= 0) {
        throw std::invalid_argument("BF16 grouped QKV key is incomplete");
    }
}

void validate_bf16_grouped_gate_up_key(
    const Bf16GroupedGateUpKey& key) {
    if (key.rows <= 0 || key.inner <= 0 || key.columns <= 0 ||
        key.architecture.empty() || key.hip_runtime_version < 0 ||
        key.hip_driver_version < 0 || key.hipblaslt_version <= 0) {
        throw std::invalid_argument(
            "BF16 grouped gate/up key is incomplete");
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
        throw std::runtime_error("matmul tuning cache field is missing: " +
                                 std::string(name));
    }
    position += needle.size();
    while (position < line.size() && line[position] == ' ') ++position;
    return position;
}

void require_value_delimiter(
    std::string_view line, std::size_t position, std::string_view name) {
    while (position < line.size() && line[position] == ' ') ++position;
    if (position >= line.size() ||
        (line[position] != ',' && line[position] != '}')) {
        throw std::runtime_error("matmul tuning cache value delimiter is invalid: " +
                                 std::string(name));
    }
}

std::string string_field(std::string_view line, std::string_view name) {
    auto position = field_start(line, name);
    if (position >= line.size() || line[position] != '"') {
        throw std::runtime_error("matmul tuning cache string is invalid: " +
                                 std::string(name));
    }
    ++position;
    std::string output;
    while (position < line.size()) {
        const auto character = line[position++];
        if (character == '"') {
            require_value_delimiter(line, position, name);
            return output;
        }
        if (character == '\\') {
            if (position >= line.size() ||
                (line[position] != '\\' && line[position] != '"')) {
                throw std::runtime_error("matmul tuning cache escape is invalid");
            }
            output.push_back(line[position++]);
        } else {
            output.push_back(character);
        }
    }
    throw std::runtime_error("matmul tuning cache string is unterminated");
}

template <typename Integer>
Integer integer_field(std::string_view line, std::string_view name) {
    const auto position = field_start(line, name);
    Integer value{};
    const auto parsed = std::from_chars(
        line.data() + position, line.data() + line.size(), value);
    if (parsed.ec != std::errc{} || parsed.ptr == line.data() + position) {
        throw std::runtime_error("matmul tuning cache integer is invalid: " +
                                 std::string(name));
    }
    require_value_delimiter(
        line, static_cast<std::size_t>(parsed.ptr - line.data()), name);
    return value;
}

bool bool_field(std::string_view line, std::string_view name) {
    const auto position = field_start(line, name);
    const auto value = line.substr(position);
    if (value.starts_with("true")) {
        require_value_delimiter(line, position + 4, name);
        return true;
    }
    if (value.starts_with("false")) {
        require_value_delimiter(line, position + 5, name);
        return false;
    }
    throw std::runtime_error("matmul tuning cache bool is invalid: " +
                             std::string(name));
}

std::vector<std::int64_t> strides_field(
    std::string_view line, std::string_view name) {
    auto position = field_start(line, name);
    if (position >= line.size() || line[position] != '[') {
        throw std::runtime_error("matmul tuning cache strides are invalid");
    }
    ++position;
    std::vector<std::int64_t> output;
    while (position < line.size()) {
        while (position < line.size() && line[position] == ' ') ++position;
        if (position < line.size() && line[position] == ']') return output;
        std::int64_t value = 0;
        const auto parsed = std::from_chars(
            line.data() + position, line.data() + line.size(), value);
        if (parsed.ec != std::errc{} || parsed.ptr == line.data() + position) {
            throw std::runtime_error("matmul tuning cache stride is invalid");
        }
        output.push_back(value);
        position = static_cast<std::size_t>(parsed.ptr - line.data());
        while (position < line.size() && line[position] == ' ') ++position;
        if (position < line.size() && line[position] == ',') {
            ++position;
            continue;
        }
        if (position < line.size() && line[position] == ']') return output;
        throw std::runtime_error("matmul tuning cache stride separator is invalid");
    }
    throw std::runtime_error("matmul tuning cache strides are unterminated");
}

DType dtype_from_name(const std::string& name) {
    for (const auto dtype : {DType::Float32, DType::Float16, DType::BFloat16,
                             DType::Float8E4M3FNUZ,
                             DType::Float8E5M2FNUZ}) {
        if (dtype_name(dtype) == name) return dtype;
    }
    throw std::runtime_error("matmul tuning cache dtype is unsupported: " + name);
}

OpMode mode_from_name(const std::string& name) {
    if (name == "unspecified") return OpMode::Unspecified;
    if (name == "inference") return OpMode::Inference;
    if (name == "training") return OpMode::Training;
    throw std::runtime_error("matmul tuning cache mode is unsupported: " + name);
}

const char* mode_name(OpMode mode) {
    switch (mode) {
        case OpMode::Unspecified: return "unspecified";
        case OpMode::Inference: return "inference";
        case OpMode::Training: return "training";
    }
    throw std::invalid_argument("unknown operator mode");
}

MatmulImplementation implementation_from_name(const std::string& name) {
    if (name == "readable") return MatmulImplementation::Readable;
    if (name == "hipblaslt") return MatmulImplementation::HipBLASLt;
    throw std::runtime_error(
        "matmul tuning cache implementation is unsupported: " + name);
}

const char* implementation_name(MatmulImplementation implementation) {
    switch (implementation) {
        case MatmulImplementation::Readable: return "readable";
        case MatmulImplementation::HipBLASLt: return "hipblaslt";
        case MatmulImplementation::Auto: break;
    }
    throw std::invalid_argument("automatic matmul choice cannot be serialized");
}

std::pair<MatmulTuningKey, MatmulImplementation> parse_cache_entry(
    std::string_view line) {
    if (integer_field<int>(line, "schema_version") !=
            kMatmulTuningCacheSchema ||
        string_field(line, "kind") != "entry") {
        throw std::runtime_error("matmul tuning cache entry schema is invalid");
    }
    MatmulTuningKey key;
    key.rows = integer_field<std::int64_t>(line, "rows");
    key.inner = integer_field<std::int64_t>(line, "inner");
    key.columns = integer_field<std::int64_t>(line, "columns");
    key.dtype = dtype_from_name(string_field(line, "dtype"));
    key.transpose_left = bool_field(line, "transpose_left");
    key.transpose_right = bool_field(line, "transpose_right");
    key.left_strides = strides_field(line, "left_strides");
    key.right_strides = strides_field(line, "right_strides");
    key.architecture = string_field(line, "architecture");
    key.hip_runtime_version = integer_field<int>(line, "hip_runtime_version");
    key.hip_driver_version = integer_field<int>(line, "hip_driver_version");
    key.hipblaslt_version = integer_field<int>(line, "hipblaslt_version");
    key.mode = mode_from_name(string_field(line, "mode"));
    key.workspace_limit = integer_field<std::size_t>(line, "workspace_limit");
    validate_matmul_tuning_key(key);
    return {std::move(key), implementation_from_name(
        string_field(line, "implementation"))};
}
#if MICROLLM_HAS_HIPBLASLT
// A few gfx942 decode shapes cannot write BF16-input GEMM results directly to
// FP32 even though the same problem can write BF16. Remember the observed
// capability per shape so repeated transformer layers do not retry a rejected
// library path on every token.
thread_local std::map<MatmulShapeKey, bool> bf16_fp32_direct_registry;
thread_local std::map<MatmulShapeKey, bool> fp8_fp32_direct_registry;
thread_local std::map<MatmulShapeKey, bool> fp8_native_matrix_registry;
thread_local std::size_t fp8_software_fallback_calls = 0;
thread_local std::optional<bool> fp8_outer_row_native;
thread_local std::size_t fp8_outer_row_fallback_calls = 0;
thread_local std::size_t fp8_output_column_scale_calls = 0;
thread_local std::optional<bool> fp8_output_column_native;
#endif
}  // namespace

bool hipblaslt_available() noexcept { return MICROLLM_HAS_HIPBLASLT != 0; }

int hipblaslt_version() noexcept {
#if MICROLLM_HAS_HIPBLASLT
    return HIPBLASLT_VERSION_MAJOR * 10000 + HIPBLASLT_VERSION_MINOR * 100 +
           HIPBLASLT_VERSION_PATCH;
#else
    return 0;
#endif
}

Bf16GroupedQkvKey make_bf16_grouped_qkv_key(
    std::int64_t rows, std::int64_t inner,
    std::int64_t query_columns, std::int64_t key_columns,
    std::int64_t value_columns, Device device) {
    const auto environment = tuning_environment(device);
    Bf16GroupedQkvKey key{
        .rows = rows,
        .inner = inner,
        .query_columns = query_columns,
        .key_columns = key_columns,
        .value_columns = value_columns,
        .architecture = environment.architecture,
        .hip_runtime_version = environment.runtime_version,
        .hip_driver_version = environment.driver_version,
        .hipblaslt_version = device.is_hip() ? hipblaslt_version() : 0,
    };
    if (device.is_hip()) validate_bf16_grouped_qkv_key(key);
    return key;
}

Bf16GroupedGateUpKey make_bf16_grouped_gate_up_key(
    std::int64_t rows, std::int64_t inner,
    std::int64_t columns, Device device) {
    const auto environment = tuning_environment(device);
    Bf16GroupedGateUpKey key{
        .rows = rows,
        .inner = inner,
        .columns = columns,
        .architecture = environment.architecture,
        .hip_runtime_version = environment.runtime_version,
        .hip_driver_version = environment.driver_version,
        .hipblaslt_version = device.is_hip() ? hipblaslt_version() : 0,
    };
    if (device.is_hip()) validate_bf16_grouped_gate_up_key(key);
    return key;
}

Fp32MatmulSolutionKey make_fp32_matmul_solution_key(
    const Shape& left_shape, const Shape& right_shape, Device device,
    bool transpose_left, bool transpose_right,
    const OpContext& context, float alpha) {
    if (left_shape.size() < 2 || right_shape.size() != left_shape.size() ||
        !std::isfinite(alpha)) {
        throw std::invalid_argument(
            "FP32 solution key requires equal-rank contiguous shapes and finite alpha");
    }
    std::int64_t batches = 1;
    for (std::size_t dimension = 0; dimension + 2 < left_shape.size();
         ++dimension) {
        if (left_shape[dimension] != right_shape[dimension] ||
            left_shape[dimension] <= 0) {
            throw std::invalid_argument(
                "FP32 solution key batch dimensions must match and be positive");
        }
        batches = checked_positive_product(
            batches, left_shape[dimension], "FP32 batch count");
    }
    const auto rank = left_shape.size();
    const auto left_rows = left_shape[rank - 2];
    const auto left_columns = left_shape[rank - 1];
    const auto right_rows = right_shape[rank - 2];
    const auto right_columns = right_shape[rank - 1];
    const auto rows = transpose_left ? left_columns : left_rows;
    const auto inner = transpose_left ? left_rows : left_columns;
    const auto right_inner = transpose_right ? right_columns : right_rows;
    const auto columns = transpose_right ? right_rows : right_columns;
    if (left_rows <= 0 || left_columns <= 0 || right_rows <= 0 ||
        right_columns <= 0 || inner != right_inner ||
        batches > std::numeric_limits<std::int32_t>::max()) {
        throw std::invalid_argument("FP32 solution key GEMM dimensions are invalid");
    }
    const auto environment = tuning_environment(device);
    Fp32MatmulSolutionKey key{
        .batches = batches,
        .left_rows = left_rows,
        .left_columns = left_columns,
        .right_rows = right_rows,
        .right_columns = right_columns,
        .output_rows = rows,
        .output_columns = columns,
        .left_batch_stride = checked_positive_product(
            left_rows, left_columns, "FP32 left batch stride"),
        .right_batch_stride = checked_positive_product(
            right_rows, right_columns, "FP32 right batch stride"),
        .output_batch_stride = checked_positive_product(
            rows, columns, "FP32 output batch stride"),
        .transpose_left = transpose_left,
        .transpose_right = transpose_right,
        .alpha_bits = std::bit_cast<std::uint32_t>(alpha),
        .architecture = environment.architecture,
        .hip_runtime_version = environment.runtime_version,
        .hip_driver_version = environment.driver_version,
        .hipblaslt_version = device.is_hip() ? hipblaslt_version() : 0,
        .mode = context.mode,
        .workspace_limit = context.workspace_bytes,
    };
    if (device.is_hip()) validate_fp32_solution_key(key);
    return key;
}

MatmulTuningKey make_matmul_tuning_key(
    const Tensor& left, const Tensor& right,
    bool transpose_left, bool transpose_right, const OpContext& context) {
    if (left.ndim() != 2 || right.ndim() != 2 ||
        !is_floating_point(left.dtype()) || right.dtype() != left.dtype() ||
        left.device() != right.device() || !left.is_contiguous() ||
        !right.is_contiguous()) {
        throw std::invalid_argument(
            "matmul tuning key requires matching contiguous rank-2 floating tensors");
    }
    const auto rows = transpose_left ? left.shape()[1] : left.shape()[0];
    const auto inner = transpose_left ? left.shape()[0] : left.shape()[1];
    const auto right_inner = transpose_right ? right.shape()[1] : right.shape()[0];
    const auto columns = transpose_right ? right.shape()[0] : right.shape()[1];
    if (inner != right_inner) {
        throw std::invalid_argument("matmul tuning key inner dimensions mismatch");
    }
    const auto environment = tuning_environment(left.device());
    return {.rows = rows,
            .inner = inner,
            .columns = columns,
            .dtype = left.dtype(),
            .transpose_left = transpose_left,
            .transpose_right = transpose_right,
            .left_strides = left.strides(),
            .right_strides = right.strides(),
            .architecture = environment.architecture,
            .hip_runtime_version = environment.runtime_version,
            .hip_driver_version = environment.driver_version,
            .hipblaslt_version = left.device().is_hip() ? hipblaslt_version() : 0,
            .mode = context.mode,
            .workspace_limit = context.workspace_bytes};
}

MatmulImplementation choose_matmul_implementation(const Tensor& left,
                                                  const Tensor& right) {
    return choose_matmul_implementation(left, right, false, false);
}

MatmulImplementation choose_matmul_implementation(const Tensor& left,
                                                  const Tensor& right,
                                                  bool transpose_left,
                                                  bool transpose_right) {
    return choose_matmul_implementation(
        left, right, transpose_left, transpose_right, {});
}

MatmulImplementation choose_matmul_implementation(
    const Tensor& left, const Tensor& right,
    bool transpose_left, bool transpose_right, const OpContext& context) {
    if (!hipblaslt_available() || !left.device().is_hip() || right.device() != left.device() ||
        left.ndim() != 2 || right.ndim() != 2 || !is_floating_point(left.dtype()) ||
        right.dtype() != left.dtype() || !left.is_contiguous() || !right.is_contiguous()) {
        return MatmulImplementation::Readable;
    }
    const auto rows = transpose_left ? left.shape()[1] : left.shape()[0];
    const auto inner = transpose_left ? left.shape()[0] : left.shape()[1];
    const auto right_inner = transpose_right ? right.shape()[1] : right.shape()[0];
    const auto columns = transpose_right ? right.shape()[0] : right.shape()[1];
    if (inner != right_inner) return MatmulImplementation::Readable;
    if (registry_entries.load(std::memory_order_acquire) != 0) {
        const auto key = make_matmul_tuning_key(
            left, right, transpose_left, transpose_right, context);
        const std::lock_guard<std::mutex> lock(registry_mutex);
        const auto found = registry.find(key);
        if (found != registry.end()) return found->second;
    }
    // Ordinary GEMM needs a substantial reduction before the library setup pays.
    // Weight-gradient GEMM is different: transpose(left) creates a wide output,
    // and measured Qwen K=3/32 shapes are 1.5x-22x faster in hipBLASLt.
    return columns >= 128 && (inner >= 128 || (transpose_left && rows >= 128))
               ? MatmulImplementation::HipBLASLt
               : MatmulImplementation::Readable;
}

void register_matmul_implementation(const MatmulTuningKey& key,
                                    MatmulImplementation implementation) {
    validate_matmul_tuning_key(key);
    if (implementation == MatmulImplementation::Auto) {
        throw std::invalid_argument("matmul registry choice must name a concrete implementation");
    }
    if (implementation == MatmulImplementation::HipBLASLt && !hipblaslt_available()) {
        throw std::invalid_argument("cannot register unavailable hipBLASLt implementation");
    }
    const std::lock_guard<std::mutex> lock(registry_mutex);
    const auto [unused, inserted] = registry.insert_or_assign(key, implementation);
    (void)unused;
    if (inserted) registry_entries.fetch_add(1, std::memory_order_release);
}

void clear_matmul_implementation_registry() {
    const std::lock_guard<std::mutex> lock(registry_mutex);
    registry.clear();
    registry_entries.store(0, std::memory_order_release);
}

std::size_t matmul_registered_implementation_count() noexcept {
    return registry_entries.load(std::memory_order_acquire);
}

void save_matmul_tuning_cache(const std::filesystem::path& path) {
    if (path.empty() || !path.has_filename()) {
        throw std::invalid_argument("matmul tuning cache path must name a file");
    }
    std::vector<std::pair<MatmulTuningKey, MatmulImplementation>> entries;
    {
        const std::lock_guard<std::mutex> lock(registry_mutex);
        entries.assign(registry.begin(), registry.end());
    }
    auto temporary = path;
    temporary += ".tmp";
    try {
        std::ofstream output(temporary, std::ios::trunc);
        if (!output) {
            throw std::runtime_error("cannot open temporary matmul tuning cache");
        }
        output << "{\"schema_version\":" << kMatmulTuningCacheSchema
               << ",\"kind\":\"microllm_matmul_tuning_cache\"}\n";
        for (const auto& [key, implementation] : entries) {
            output << "{\"schema_version\":" << kMatmulTuningCacheSchema
                   << ",\"kind\":\"entry\""
                   << ",\"rows\":" << key.rows
                   << ",\"inner\":" << key.inner
                   << ",\"columns\":" << key.columns
                   << ",\"dtype\":" << json_string(dtype_name(key.dtype))
                   << ",\"transpose_left\":"
                   << (key.transpose_left ? "true" : "false")
                   << ",\"transpose_right\":"
                   << (key.transpose_right ? "true" : "false")
                   << ",\"left_strides\":[" << key.left_strides[0] << ','
                   << key.left_strides[1] << ']'
                   << ",\"right_strides\":[" << key.right_strides[0] << ','
                   << key.right_strides[1] << ']'
                   << ",\"architecture\":" << json_string(key.architecture)
                   << ",\"hip_runtime_version\":" << key.hip_runtime_version
                   << ",\"hip_driver_version\":" << key.hip_driver_version
                   << ",\"hipblaslt_version\":" << key.hipblaslt_version
                   << ",\"mode\":" << json_string(mode_name(key.mode))
                   << ",\"workspace_limit\":" << key.workspace_limit
                   << ",\"implementation\":"
                   << json_string(implementation_name(implementation)) << "}\n";
        }
        output.flush();
        if (!output) throw std::runtime_error("cannot write matmul tuning cache");
        output.close();
        std::error_code error;
        std::filesystem::rename(temporary, path, error);
        if (error) {
            throw std::runtime_error(
                "cannot atomically replace matmul tuning cache: " + error.message());
        }
    } catch (...) {
        std::error_code ignored;
        std::filesystem::remove(temporary, ignored);
        throw;
    }
}

MatmulTuningCacheLoadReport load_matmul_tuning_cache(
    const std::filesystem::path& path, Device device, bool replace_existing) {
    std::ifstream input(path);
    if (!input) throw std::runtime_error("cannot open matmul tuning cache");
    std::string line;
    if (!std::getline(input, line) || line.size() > 65536 ||
        integer_field<int>(line, "schema_version") !=
            kMatmulTuningCacheSchema ||
        string_field(line, "kind") != "microllm_matmul_tuning_cache") {
        throw std::runtime_error("matmul tuning cache header is invalid");
    }
    std::vector<std::pair<MatmulTuningKey, MatmulImplementation>> accepted;
    std::set<MatmulTuningKey> keys;
    MatmulTuningCacheLoadReport report;
    const auto environment = tuning_environment(device);
    const auto expected_backend_version =
        device.is_hip() ? hipblaslt_version() : 0;
    while (std::getline(input, line)) {
        if (line.empty()) continue;
        if (line.size() > 65536 || report.parsed_entries >= 100000) {
            throw std::runtime_error("matmul tuning cache exceeds safety limits");
        }
        auto entry = parse_cache_entry(line);
        ++report.parsed_entries;
        if (!keys.insert(entry.first).second) {
            throw std::runtime_error("matmul tuning cache contains a duplicate key");
        }
        const auto& key = entry.first;
        const auto current_environment =
            key.architecture == environment.architecture &&
            key.hip_runtime_version == environment.runtime_version &&
            key.hip_driver_version == environment.driver_version &&
            key.hipblaslt_version == expected_backend_version;
        if (!current_environment) {
            ++report.stale_entries;
            continue;
        }
        if (entry.second == MatmulImplementation::HipBLASLt &&
            !hipblaslt_available()) {
            throw std::runtime_error(
                "current cache entry requires unavailable hipBLASLt");
        }
        accepted.push_back(std::move(entry));
    }
    if (!input.eof()) throw std::runtime_error("cannot read matmul tuning cache");

    {
        const std::lock_guard<std::mutex> lock(registry_mutex);
        auto updated = replace_existing
                           ? std::map<MatmulTuningKey, MatmulImplementation>{}
                           : registry;
        for (auto& [key, implementation] : accepted) {
            updated.insert_or_assign(std::move(key), implementation);
        }
        registry.swap(updated);
        registry_entries.store(registry.size(), std::memory_order_release);
        report.loaded_entries = accepted.size();
    }
    return report;
}

#if MICROLLM_HAS_HIPBLASLT
namespace {

void check_status(hipblasStatus_t status, const char* operation) {
    if (status != HIPBLAS_STATUS_SUCCESS) {
        throw std::runtime_error(std::string(operation) + " failed with hipBLASLt status " +
                                 std::to_string(static_cast<int>(status)));
    }
}

class Handle {
public:
    Handle() { check_status(hipblasLtCreate(&value_), "hipblasLtCreate"); }
    ~Handle() { (void)hipblasLtDestroy(value_); }
    Handle(const Handle&) = delete;
    Handle& operator=(const Handle&) = delete;
    hipblasLtHandle_t get() const noexcept { return value_; }

private:
    hipblasLtHandle_t value_ = nullptr;
};

Handle& handle_for_device(Device device) {
    if (!device.is_hip()) {
        throw std::invalid_argument("hipBLASLt handle requires a HIP device");
    }
    runtime::set_device(device);
    static thread_local std::map<int, std::unique_ptr<Handle>> handles;
    auto& handle = handles[device.index()];
    if (!handle) handle = std::make_unique<Handle>();
    return *handle;
}

class Layout {
public:
    Layout(hipDataType dtype, std::uint64_t rows, std::uint64_t columns,
           std::int64_t leading_dimension) {
        check_status(hipblasLtMatrixLayoutCreate(&value_, dtype, rows, columns,
                                                 leading_dimension),
                     "hipblasLtMatrixLayoutCreate");
    }
    ~Layout() { (void)hipblasLtMatrixLayoutDestroy(value_); }
    Layout(const Layout&) = delete;
    Layout& operator=(const Layout&) = delete;
    void set_batch(std::int32_t count, std::int64_t stride) {
        check_status(hipblasLtMatrixLayoutSetAttribute(
                         value_, HIPBLASLT_MATRIX_LAYOUT_BATCH_COUNT,
                         &count, sizeof(count)),
                     "hipblasLtMatrixLayoutSetAttribute(BATCH_COUNT)");
        check_status(hipblasLtMatrixLayoutSetAttribute(
                         value_, HIPBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET,
                         &stride, sizeof(stride)),
                     "hipblasLtMatrixLayoutSetAttribute(STRIDED_BATCH_OFFSET)");
    }
    hipblasLtMatrixLayout_t get() const noexcept { return value_; }

private:
    hipblasLtMatrixLayout_t value_ = nullptr;
};

hipDataType hip_dtype(DType dtype) {
    switch (dtype) {
        case DType::Float32: return HIP_R_32F;
        case DType::Float16: return HIP_R_16F;
        case DType::BFloat16: return HIP_R_16BF;
        case DType::Float8E4M3FNUZ: return HIP_R_8F_E4M3_FNUZ;
        case DType::Float8E5M2FNUZ: return HIP_R_8F_E5M2_FNUZ;
        case DType::Int32:
        case DType::Int64: break;
    }
    throw std::invalid_argument("hipBLASLt matmul requires FP32, FP16, or BF16");
}

void set_scale_pointer(hipblasLtMatmulDesc_t description,
                       hipblasLtMatmulDescAttributes_t attribute,
                       const void* pointer) {
    check_status(hipblasLtMatmulDescSetAttribute(
                     description, attribute, &pointer, sizeof(pointer)),
                 "hipblasLtMatmulDescSetAttribute(scale)");
}

void set_scale_mode(hipblasLtMatmulDesc_t description,
                    hipblasLtMatmulDescAttributes_t attribute,
                    hipblasLtMatmulMatrixScale_t mode) {
    check_status(hipblasLtMatmulDescSetAttribute(
                     description, attribute, &mode, sizeof(mode)),
                 "hipblasLtMatmulDescSetAttribute(scale mode)");
}

class MatmulDescription {
public:
    MatmulDescription(bool transpose_a = false, bool transpose_b = false) {
        check_status(hipblasLtMatmulDescCreate(&value_, HIPBLAS_COMPUTE_32F,
                                               HIP_R_32F),
                     "hipblasLtMatmulDescCreate");
        const auto operation_a = transpose_a ? HIPBLAS_OP_T : HIPBLAS_OP_N;
        const auto operation_b = transpose_b ? HIPBLAS_OP_T : HIPBLAS_OP_N;
        check_status(hipblasLtMatmulDescSetAttribute(
                         value_, HIPBLASLT_MATMUL_DESC_TRANSA,
                         &operation_a, sizeof(operation_a)),
                     "hipblasLtMatmulDescSetAttribute(TRANSA)");
        check_status(hipblasLtMatmulDescSetAttribute(
                         value_, HIPBLASLT_MATMUL_DESC_TRANSB,
                         &operation_b, sizeof(operation_b)),
                     "hipblasLtMatmulDescSetAttribute(TRANSB)");
    }
    ~MatmulDescription() { (void)hipblasLtMatmulDescDestroy(value_); }
    MatmulDescription(const MatmulDescription&) = delete;
    MatmulDescription& operator=(const MatmulDescription&) = delete;
    hipblasLtMatmulDesc_t get() const noexcept { return value_; }

private:
    hipblasLtMatmulDesc_t value_ = nullptr;
};

using Bf16PlanKey = std::tuple<std::int64_t, std::int64_t, std::int64_t, DType>;
using Bf16PlanCacheKey = std::tuple<std::int64_t, std::int64_t, std::int64_t,
                                    DType, std::int32_t>;

class Bf16Plan {
public:
    Bf16Plan(Handle& handle, std::int64_t rows, std::int64_t inner,
             std::int64_t columns, DType output_dtype, Device device,
             std::optional<int> solution_index)
        : matrix_b_(HIP_R_16BF, static_cast<std::uint64_t>(columns),
                    static_cast<std::uint64_t>(inner), columns),
          matrix_a_(HIP_R_16BF, static_cast<std::uint64_t>(inner),
                    static_cast<std::uint64_t>(rows), inner),
          matrix_c_(hip_dtype(output_dtype), static_cast<std::uint64_t>(columns),
                    static_cast<std::uint64_t>(rows), columns) {
        if (!solution_index.has_value()) return;
        std::vector<int> indices{*solution_index};
        std::vector<hipblasLtMatmulHeuristicResult_t> results;
        check_status(hipblaslt_ext::getAlgosFromIndex(
                         handle.get(), indices, results),
                     "getAlgosFromIndex");
        const float alpha = 1.0F;
        const float beta = 0.0F;
        for (auto& result : results) {
            auto candidate = result.algo;
            std::size_t workspace_bytes = 0;
            if (hipblaslt_ext::matmulIsAlgoSupported(
                    handle.get(), operation_.get(), &alpha, matrix_b_.get(),
                    matrix_a_.get(), &beta, matrix_c_.get(), matrix_c_.get(),
                    candidate, workspace_bytes) != HIPBLAS_STATUS_SUCCESS) {
                continue;
            }
            algorithm_ = candidate;
            workspace_ = Storage(workspace_bytes, device);
            return;
        }
        throw std::invalid_argument(
            "registered BF16 solution does not support the exact shape");
    }

    [[nodiscard]] hipblasLtMatmulDesc_t operation() const noexcept {
        return operation_.get();
    }
    [[nodiscard]] hipblasLtMatrixLayout_t matrix_b() const noexcept {
        return matrix_b_.get();
    }
    [[nodiscard]] hipblasLtMatrixLayout_t matrix_a() const noexcept {
        return matrix_a_.get();
    }
    [[nodiscard]] hipblasLtMatrixLayout_t matrix_c() const noexcept {
        return matrix_c_.get();
    }
    [[nodiscard]] const hipblasLtMatmulAlgo_t* algorithm() const noexcept {
        return algorithm_.has_value() ? &*algorithm_ : nullptr;
    }
    [[nodiscard]] void* workspace() noexcept { return workspace_.data(); }
    [[nodiscard]] std::size_t workspace_bytes() const noexcept {
        return workspace_.num_bytes();
    }

private:
    MatmulDescription operation_;
    Layout matrix_b_;
    Layout matrix_a_;
    Layout matrix_c_;
    std::optional<hipblasLtMatmulAlgo_t> algorithm_;
    Storage workspace_;
};

thread_local std::map<Bf16PlanCacheKey, std::unique_ptr<Bf16Plan>> bf16_plans;
thread_local std::map<Bf16PlanKey, int> bf16_algorithm_registry;
thread_local std::size_t bf16_plan_hits = 0;
thread_local std::size_t bf16_plan_misses = 0;

struct Fp32SolutionAlgorithm {
    int solution_index = -1;
    hipblasLtMatmulAlgo_t algorithm{};
    std::size_t workspace_bytes = 0;
};

using Fp32SolutionCacheKey = std::pair<Fp32MatmulSolutionKey, int>;
thread_local std::map<Fp32SolutionCacheKey, Fp32SolutionAlgorithm>
    fp32_solution_algorithms;

const hipblasLtMatmulAlgo_t* registered_fp32_solution(
    Handle& handle, const Fp32MatmulSolutionKey& key,
    int device_index,
    hipblasLtMatmulDesc_t operation, hipblasLtMatrixLayout_t matrix_b,
    hipblasLtMatrixLayout_t matrix_a, hipblasLtMatrixLayout_t matrix_c,
    const float* alpha, const float* beta, const OpContext& context) {
    if (fp32_solution_registry.empty()) return nullptr;
    const auto registered = fp32_solution_registry.find(key);
    if (registered == fp32_solution_registry.end()) {
        ++fp32_solution_registry_misses;
        return nullptr;
    }
    ++fp32_solution_registry_hits;
    const Fp32SolutionCacheKey cache_key{key, device_index};
    const auto cached = fp32_solution_algorithms.find(cache_key);
    if (cached != fp32_solution_algorithms.end()) {
        if (cached->second.workspace_bytes != 0 &&
            context.workspace == nullptr) {
            throw std::invalid_argument(
                "cached FP32 solution requires caller-owned workspace");
        }
        ++fp32_solution_cache_hits;
        ++fp32_solution_dispatches;
        return &cached->second.algorithm;
    }

    ++fp32_solution_cache_misses;
    std::vector<int> indices{registered->second};
    std::vector<hipblasLtMatmulHeuristicResult_t> results;
    check_status(hipblaslt_ext::getAlgosFromIndex(
                     handle.get(), indices, results),
                 "getAlgosFromIndex(FP32)");
    for (auto& result : results) {
        auto candidate = result.algo;
        std::size_t workspace_bytes = 0;
        if (hipblaslt_ext::matmulIsAlgoSupported(
                handle.get(), operation, alpha, matrix_b, matrix_a, beta,
                matrix_c, matrix_c, candidate,
                workspace_bytes) != HIPBLAS_STATUS_SUCCESS) {
            continue;
        }
        if (workspace_bytes > context.workspace_bytes ||
            (workspace_bytes != 0 && context.workspace == nullptr)) {
            throw std::invalid_argument(
                "registered FP32 solution exceeds the exact workspace contract");
        }
        const auto [inserted, unused] = fp32_solution_algorithms.emplace(
            cache_key, Fp32SolutionAlgorithm{registered->second, candidate,
                                             workspace_bytes});
        (void)unused;
        ++fp32_solution_dispatches;
        return &inserted->second.algorithm;
    }
    throw std::invalid_argument(
        "registered FP32 solution does not support the exact descriptor");
}

using Bf16GroupedQkvPlanKey = std::tuple<
    Bf16GroupedQkvKey, int, std::uintptr_t, std::uintptr_t, std::uintptr_t,
    std::uintptr_t, std::uintptr_t, std::uintptr_t, std::uintptr_t,
    std::uintptr_t>;

using Bf16GroupedQkvAlgorithmKey =
    std::tuple<Bf16GroupedQkvKey, int, int>;
thread_local std::map<Bf16GroupedQkvAlgorithmKey, hipblasLtMatmulAlgo_t>
    bf16_grouped_qkv_algorithms;

hipblasLtMatmulAlgo_t grouped_qkv_algorithm(
    Handle& handle, const Bf16GroupedQkvKey& key,
    int solution_index, Device device) {
    const Bf16GroupedQkvAlgorithmKey algorithm_key{
        key, solution_index, device.index()};
    const auto found = bf16_grouped_qkv_algorithms.find(algorithm_key);
    if (found != bf16_grouped_qkv_algorithms.end()) {
        ++bf16_grouped_qkv_algorithm_hits;
        return found->second;
    }
    ++bf16_grouped_qkv_algorithm_misses;
    std::vector<int> indices{solution_index};
    std::vector<hipblasLtMatmulHeuristicResult_t> algorithms;
    check_status(hipblaslt_ext::getAlgosFromIndex(
                     handle.get(), indices, algorithms),
                 "getAlgosFromIndex(BF16 grouped QKV)");
    if (algorithms.empty()) {
        throw std::invalid_argument(
            "registered BF16 grouped QKV solution index is unavailable");
    }
    const auto algorithm = algorithms.front().algo;
    bf16_grouped_qkv_algorithms.emplace(algorithm_key, algorithm);
    return algorithm;
}

using Bf16GroupedQkvKernelKey =
    std::tuple<Bf16GroupedQkvKey, int, int, std::uintptr_t>;

class Bf16GroupedQkvKernel {
public:
    Bf16GroupedQkvKernel(
        Handle& handle, const Bf16GroupedQkvKey& key, int solution_index,
        const Tensor& input_bf16, const Tensor& query_weight_bf16,
        const Tensor& key_weight_bf16, const Tensor& value_weight_bf16,
        Tensor& query_bf16, Tensor& key_bf16, Tensor& value_bf16,
        Device device, void* stream)
        : grouped_(handle.get(), HIPBLAS_OP_N, HIPBLAS_OP_N,
                   HIP_R_16BF, HIP_R_16BF, HIP_R_16BF, HIP_R_16BF,
                   HIPBLAS_COMPUTE_32F) {
        constexpr std::size_t workspace_limit = 32U * 1024U * 1024U;
        grouped_.setMaxWorkspaceBytes(workspace_limit);
        std::vector<std::int64_t> m{
            key.query_columns, key.key_columns, key.value_columns};
        std::vector<std::int64_t> n(3, key.rows);
        std::vector<std::int64_t> k(3, key.inner);
        std::vector<std::int64_t> batch(3, 1);
        std::vector<hipblaslt_ext::GemmEpilogue> epilogues(3);
        std::vector<hipblaslt_ext::GemmInputs> inputs(3);
        const std::vector<const Tensor*> weights{
            &query_weight_bf16, &key_weight_bf16, &value_weight_bf16};
        const std::vector<Tensor*> outputs{&query_bf16, &key_bf16, &value_bf16};
        for (std::size_t group = 0; group < 3; ++group) {
            inputs[group].setA(weights[group]->data());
            inputs[group].setB(input_bf16.data());
            inputs[group].setC(outputs[group]->data());
            inputs[group].setD(outputs[group]->data());
            inputs[group].setAlpha(&alpha_);
            inputs[group].setBeta(&beta_);
        }
        check_status(grouped_.setProblem(m, n, k, batch, epilogues, inputs),
                     "GroupedGemm::setProblem(BF16 QKV)");
        auto candidate = grouped_qkv_algorithm(
            handle, key, solution_index, device);
        hipblaslt_ext::GemmTuning tuning;
        std::size_t workspace_bytes = 0;
        if (grouped_.isAlgoSupported(
                candidate, tuning, workspace_bytes) !=
                HIPBLAS_STATUS_SUCCESS ||
            workspace_bytes > workspace_limit) {
            throw std::invalid_argument(
                "registered BF16 grouped QKV solution is unsupported");
        }
        workspace_ = Storage(workspace_bytes, device);
        check_status(grouped_.initialize(
                         candidate, workspace_.data(), true,
                         reinterpret_cast<hipStream_t>(stream)),
                     "GroupedGemm::initialize(BF16 QKV)");
        host_arguments_.resize(3);
        check_status(grouped_.getDefaultValueForDeviceUserArguments(
                         host_arguments_.data()),
                     "GroupedGemm::getDefaultValueForDeviceUserArguments(BF16 QKV)");
    }

    Storage make_arguments(
        const Tensor& input_bf16, const Tensor& query_weight_bf16,
        const Tensor& key_weight_bf16, const Tensor& value_weight_bf16,
        Tensor& query_bf16, Tensor& key_bf16, Tensor& value_bf16,
        Device device) const {
        auto arguments = host_arguments_;
        const std::vector<const Tensor*> weights{
            &query_weight_bf16, &key_weight_bf16, &value_weight_bf16};
        const std::vector<Tensor*> outputs{&query_bf16, &key_bf16, &value_bf16};
        for (std::size_t group = 0; group < arguments.size(); ++group) {
            arguments[group].a = const_cast<void*>(weights[group]->data());
            arguments[group].b = const_cast<void*>(input_bf16.data());
            arguments[group].c = outputs[group]->data();
            arguments[group].d = outputs[group]->data();
        }
        Storage device_arguments(
            arguments.size() * sizeof(hipblaslt_ext::UserArguments), device);
        runtime::copy_bytes(
            device_arguments.data(), device, arguments.data(), Device::cpu(),
            device_arguments.num_bytes());
        return device_arguments;
    }

    void run(Storage& arguments, void* stream) {
        check_status(grouped_.run(
                         arguments.data(), reinterpret_cast<hipStream_t>(stream)),
                     "GroupedGemm::run(BF16 QKV)");
    }

private:
    float alpha_ = 1.0F;
    float beta_ = 0.0F;
    hipblaslt_ext::GroupedGemm grouped_;
    Storage workspace_;
    std::vector<hipblaslt_ext::UserArguments> host_arguments_;
};

thread_local std::map<Bf16GroupedQkvKernelKey,
                      std::shared_ptr<Bf16GroupedQkvKernel>>
    bf16_grouped_qkv_kernels;

class Bf16GroupedQkvPlan {
public:
    Bf16GroupedQkvPlan(
        std::shared_ptr<Bf16GroupedQkvKernel> kernel,
        const Tensor& input_bf16, const Tensor& query_weight_bf16,
        const Tensor& key_weight_bf16, const Tensor& value_weight_bf16,
        Tensor& query_bf16, Tensor& key_bf16, Tensor& value_bf16,
        Device device)
        : kernel_(std::move(kernel)),
          arguments_(kernel_->make_arguments(
              input_bf16, query_weight_bf16, key_weight_bf16,
              value_weight_bf16, query_bf16, key_bf16, value_bf16,
              device)) {}

    void run(void* stream) { kernel_->run(arguments_, stream); }

private:
    std::shared_ptr<Bf16GroupedQkvKernel> kernel_;
    Storage arguments_;
};

thread_local std::map<Bf16GroupedQkvPlanKey,
                      std::unique_ptr<Bf16GroupedQkvPlan>>
    bf16_grouped_qkv_plans;

Bf16GroupedQkvPlanKey grouped_qkv_plan_key(
    const Bf16GroupedQkvKey& key, Device device,
    const Tensor& input_bf16, const Tensor& query_weight_bf16,
    const Tensor& key_weight_bf16, const Tensor& value_weight_bf16,
    const Bf16QkvWorkspace& workspace,
    void* stream) {
    const auto address = [](const void* pointer) {
        return reinterpret_cast<std::uintptr_t>(pointer);
    };
    return {
        key, device.index(), address(input_bf16.data()),
        address(query_weight_bf16.data()), address(key_weight_bf16.data()),
        address(value_weight_bf16.data()),
        address(workspace.query_fallback_bf16.data()),
        address(workspace.key_fallback_bf16.data()),
        address(workspace.value_fallback_bf16.data()), address(stream)};
}

bool try_bf16_grouped_qkv(
    Bf16QkvWorkspace& workspace,
    const Tensor& query_weight_bf16, const Tensor& key_weight_bf16,
    const Tensor& value_weight_bf16, const OpContext& context) {
    if (bf16_grouped_qkv_registry.empty() ||
        !workspace.input_bf16.device().is_hip()) {
        return false;
    }
    const auto key = make_bf16_grouped_qkv_key(
        workspace.input_bf16.shape()[0], workspace.input_bf16.shape()[1],
        query_weight_bf16.shape()[1], key_weight_bf16.shape()[1],
        value_weight_bf16.shape()[1], workspace.input_bf16.device());
    const auto registered = bf16_grouped_qkv_registry.find(key);
    if (registered == bf16_grouped_qkv_registry.end()) return false;
    auto* stream = context.native_stream(workspace.input_bf16.device());
    const auto plan_key = grouped_qkv_plan_key(
        key, workspace.input_bf16.device(), workspace.input_bf16,
        query_weight_bf16, key_weight_bf16, value_weight_bf16,
        workspace, stream);
    auto found = bf16_grouped_qkv_plans.find(plan_key);
    if (found == bf16_grouped_qkv_plans.end()) {
        ++bf16_grouped_qkv_plan_misses;
        auto& handle = handle_for_device(workspace.input_bf16.device());
        const Bf16GroupedQkvKernelKey kernel_key{
            key, registered->second, workspace.input_bf16.device().index(),
            reinterpret_cast<std::uintptr_t>(stream)};
        auto kernel = bf16_grouped_qkv_kernels.find(kernel_key);
        if (kernel == bf16_grouped_qkv_kernels.end()) {
            ++bf16_grouped_qkv_kernel_misses;
            const auto setup_start = std::chrono::steady_clock::now();
            auto grouped_kernel = std::make_shared<Bf16GroupedQkvKernel>(
                handle, key, registered->second, workspace.input_bf16,
                query_weight_bf16, key_weight_bf16, value_weight_bf16,
                workspace.query_fallback_bf16,
                workspace.key_fallback_bf16,
                workspace.value_fallback_bf16,
                workspace.input_bf16.device(), stream);
            const auto setup_finish = std::chrono::steady_clock::now();
            bf16_grouped_qkv_kernel_setup_ms +=
                std::chrono::duration<double, std::milli>(
                    setup_finish - setup_start).count();
            kernel = bf16_grouped_qkv_kernels.emplace(
                kernel_key, std::move(grouped_kernel)).first;
        } else {
            ++bf16_grouped_qkv_kernel_hits;
        }
        const auto argument_start = std::chrono::steady_clock::now();
        auto plan = std::make_unique<Bf16GroupedQkvPlan>(
            kernel->second, workspace.input_bf16,
            query_weight_bf16, key_weight_bf16, value_weight_bf16,
            workspace.query_fallback_bf16, workspace.key_fallback_bf16,
            workspace.value_fallback_bf16, workspace.input_bf16.device());
        const auto argument_finish = std::chrono::steady_clock::now();
        bf16_grouped_qkv_argument_setup_ms +=
            std::chrono::duration<double, std::milli>(
                argument_finish - argument_start).count();
        found = bf16_grouped_qkv_plans.emplace(
            plan_key, std::move(plan)).first;
    } else {
        ++bf16_grouped_qkv_plan_hits;
    }
    found->second->run(stream);
    ++bf16_grouped_qkv_dispatches;
    return true;
}

using Bf16GroupedGateUpAlgorithmKey =
    std::tuple<Bf16GroupedGateUpKey, int, int>;
thread_local std::map<Bf16GroupedGateUpAlgorithmKey, hipblasLtMatmulAlgo_t>
    bf16_grouped_gate_up_algorithms;

hipblasLtMatmulAlgo_t grouped_gate_up_algorithm(
    Handle& handle, const Bf16GroupedGateUpKey& key,
    int solution_index, Device device) {
    const Bf16GroupedGateUpAlgorithmKey algorithm_key{
        key, solution_index, device.index()};
    const auto found = bf16_grouped_gate_up_algorithms.find(algorithm_key);
    if (found != bf16_grouped_gate_up_algorithms.end()) {
        ++bf16_grouped_gate_up_algorithm_hits;
        return found->second;
    }
    ++bf16_grouped_gate_up_algorithm_misses;
    std::vector<int> indices{solution_index};
    std::vector<hipblasLtMatmulHeuristicResult_t> algorithms;
    check_status(hipblaslt_ext::getAlgosFromIndex(
                     handle.get(), indices, algorithms),
                 "getAlgosFromIndex(BF16 grouped gate/up)");
    if (algorithms.empty()) {
        throw std::invalid_argument(
            "registered BF16 grouped gate/up solution index is unavailable");
    }
    const auto algorithm = algorithms.front().algo;
    bf16_grouped_gate_up_algorithms.emplace(algorithm_key, algorithm);
    return algorithm;
}

using Bf16GroupedGateUpKernelKey =
    std::tuple<Bf16GroupedGateUpKey, int, int, std::uintptr_t, bool>;

class Bf16GroupedGateUpKernel {
public:
    Bf16GroupedGateUpKernel(
        Handle& handle, const Bf16GroupedGateUpKey& key,
        int solution_index, const Tensor& input_bf16,
        const Tensor& gate_weight_bf16,
        const Tensor& up_weight_bf16, Tensor& gate_bf16,
        Tensor& up_bf16, Device device, void* stream, bool gate_swish)
        : grouped_(handle.get(), HIPBLAS_OP_N, HIPBLAS_OP_N,
                   HIP_R_16BF, HIP_R_16BF, HIP_R_16BF, HIP_R_16BF,
                   HIPBLAS_COMPUTE_32F) {
        constexpr std::size_t workspace_limit = 32U * 1024U * 1024U;
        grouped_.setMaxWorkspaceBytes(workspace_limit);
        std::vector<std::int64_t> m(2, key.columns);
        std::vector<std::int64_t> n(2, key.rows);
        std::vector<std::int64_t> k(2, key.inner);
        std::vector<std::int64_t> batch(2, 1);
        std::vector<hipblaslt_ext::GemmEpilogue> epilogues(2);
        if (gate_swish) {
            epilogues[0].setMode(HIPBLASLT_EPILOGUE_SWISH_EXT);
        }
        std::vector<hipblaslt_ext::GemmInputs> inputs(2);
        const std::vector<const Tensor*> weights{
            &gate_weight_bf16, &up_weight_bf16};
        const std::vector<Tensor*> outputs{&gate_bf16, &up_bf16};
        for (std::size_t group = 0; group < 2; ++group) {
            inputs[group].setA(weights[group]->data());
            inputs[group].setB(input_bf16.data());
            inputs[group].setC(outputs[group]->data());
            inputs[group].setD(outputs[group]->data());
            inputs[group].setAlpha(&alpha_);
            inputs[group].setBeta(&beta_);
        }
        check_status(grouped_.setProblem(
                         m, n, k, batch, epilogues, inputs),
                     "GroupedGemm::setProblem(BF16 gate/up)");
        auto candidate = grouped_gate_up_algorithm(
            handle, key, solution_index, device);
        hipblaslt_ext::GemmTuning tuning;
        std::size_t workspace_bytes = 0;
        if (grouped_.isAlgoSupported(
                candidate, tuning, workspace_bytes) !=
                HIPBLAS_STATUS_SUCCESS ||
            workspace_bytes > workspace_limit) {
            throw std::invalid_argument(
                "registered BF16 grouped gate/up solution is unsupported");
        }
        workspace_ = Storage(workspace_bytes, device);
        check_status(grouped_.initialize(
                         candidate, workspace_.data(), true,
                         reinterpret_cast<hipStream_t>(stream)),
                     "GroupedGemm::initialize(BF16 gate/up)");
        host_arguments_.resize(2);
        check_status(grouped_.getDefaultValueForDeviceUserArguments(
                         host_arguments_.data()),
                     "GroupedGemm::getDefaultValueForDeviceUserArguments(BF16 gate/up)");
    }

    Storage make_arguments(
        const Tensor& input_bf16,
        const Tensor& gate_weight_bf16,
        const Tensor& up_weight_bf16,
        Tensor& gate_bf16, Tensor& up_bf16,
        Device device) const {
        auto arguments = host_arguments_;
        const std::vector<const Tensor*> weights{
            &gate_weight_bf16, &up_weight_bf16};
        const std::vector<Tensor*> outputs{&gate_bf16, &up_bf16};
        for (std::size_t group = 0; group < arguments.size(); ++group) {
            arguments[group].a =
                const_cast<void*>(weights[group]->data());
            arguments[group].b =
                const_cast<void*>(input_bf16.data());
            arguments[group].c = outputs[group]->data();
            arguments[group].d = outputs[group]->data();
        }
        Storage device_arguments(
            arguments.size() * sizeof(hipblaslt_ext::UserArguments),
            device);
        runtime::copy_bytes(
            device_arguments.data(), device, arguments.data(),
            Device::cpu(), device_arguments.num_bytes());
        return device_arguments;
    }

    void run(Storage& arguments, void* stream) {
        check_status(grouped_.run(
                         arguments.data(),
                         reinterpret_cast<hipStream_t>(stream)),
                     "GroupedGemm::run(BF16 gate/up)");
    }

private:
    float alpha_ = 1.0F;
    float beta_ = 0.0F;
    hipblaslt_ext::GroupedGemm grouped_;
    Storage workspace_;
    std::vector<hipblaslt_ext::UserArguments> host_arguments_;
};

thread_local std::map<Bf16GroupedGateUpKernelKey,
                      std::shared_ptr<Bf16GroupedGateUpKernel>>
    bf16_grouped_gate_up_kernels;

class Bf16GroupedGateUpPlan {
public:
    Bf16GroupedGateUpPlan(
        std::shared_ptr<Bf16GroupedGateUpKernel> kernel,
        const Tensor& input_bf16,
        const Tensor& gate_weight_bf16,
        const Tensor& up_weight_bf16,
        Tensor& gate_bf16, Tensor& up_bf16,
        Device device)
        : kernel_(std::move(kernel)),
          arguments_(kernel_->make_arguments(
              input_bf16, gate_weight_bf16, up_weight_bf16,
              gate_bf16, up_bf16, device)) {}

    void run(void* stream) { kernel_->run(arguments_, stream); }

private:
    std::shared_ptr<Bf16GroupedGateUpKernel> kernel_;
    Storage arguments_;
};

using Bf16GroupedGateUpPlanKey = std::tuple<
    Bf16GroupedGateUpKey, int, std::uintptr_t, std::uintptr_t,
    std::uintptr_t, std::uintptr_t, std::uintptr_t, std::uintptr_t, bool>;
thread_local std::map<Bf16GroupedGateUpPlanKey,
                      std::unique_ptr<Bf16GroupedGateUpPlan>>
    bf16_grouped_gate_up_plans;

Bf16GroupedGateUpPlanKey grouped_gate_up_plan_key(
    const Bf16GroupedGateUpKey& key, Device device,
    const Tensor& input_bf16,
    const Tensor& gate_weight_bf16,
    const Tensor& up_weight_bf16,
    const Tensor& gate_bf16, const Tensor& up_bf16,
    void* stream, bool gate_swish) {
    const auto address = [](const void* pointer) {
        return reinterpret_cast<std::uintptr_t>(pointer);
    };
    return {
        key, device.index(), address(input_bf16.data()),
        address(gate_weight_bf16.data()),
        address(up_weight_bf16.data()), address(gate_bf16.data()),
        address(up_bf16.data()), address(stream), gate_swish};
}

bool try_bf16_grouped_gate_up(
    Bf16FfnWorkspace& workspace,
    const Tensor& gate_weight_bf16,
    const Tensor& up_weight_bf16,
    const OpContext& context) {
    if (bf16_grouped_gate_up_registry.empty() ||
        !workspace.input_bf16.device().is_hip()) {
        return false;
    }
    const auto key = make_bf16_grouped_gate_up_key(
        workspace.input_bf16.shape()[0],
        workspace.input_bf16.shape()[1],
        gate_weight_bf16.shape()[1],
        workspace.input_bf16.device());
    const auto registered =
        bf16_grouped_gate_up_registry.find(key);
    if (registered == bf16_grouped_gate_up_registry.end()) return false;
    auto* stream =
        context.native_stream(workspace.input_bf16.device());
    const auto gate_swish = bf16_grouped_gate_up_swish;
    const auto plan_key = grouped_gate_up_plan_key(
        key, workspace.input_bf16.device(), workspace.input_bf16,
        gate_weight_bf16, up_weight_bf16,
        workspace.gate, workspace.up, stream, gate_swish);
    auto found = bf16_grouped_gate_up_plans.find(plan_key);
    if (found == bf16_grouped_gate_up_plans.end()) {
        ++bf16_grouped_gate_up_plan_misses;
        auto& handle =
            handle_for_device(workspace.input_bf16.device());
        const Bf16GroupedGateUpKernelKey kernel_key{
            key, registered->second,
            workspace.input_bf16.device().index(),
            reinterpret_cast<std::uintptr_t>(stream), gate_swish};
        auto kernel = bf16_grouped_gate_up_kernels.find(kernel_key);
        if (kernel == bf16_grouped_gate_up_kernels.end()) {
            ++bf16_grouped_gate_up_kernel_misses;
            const auto setup_start =
                std::chrono::steady_clock::now();
            auto grouped_kernel =
                std::make_shared<Bf16GroupedGateUpKernel>(
                    handle, key, registered->second,
                    workspace.input_bf16,
                    gate_weight_bf16, up_weight_bf16,
                    workspace.gate, workspace.up,
                    workspace.input_bf16.device(), stream, gate_swish);
            const auto setup_finish =
                std::chrono::steady_clock::now();
            bf16_grouped_gate_up_kernel_setup_ms +=
                std::chrono::duration<double, std::milli>(
                    setup_finish - setup_start).count();
            kernel = bf16_grouped_gate_up_kernels.emplace(
                kernel_key, std::move(grouped_kernel)).first;
        } else {
            ++bf16_grouped_gate_up_kernel_hits;
        }
        const auto argument_start =
            std::chrono::steady_clock::now();
        auto plan = std::make_unique<Bf16GroupedGateUpPlan>(
            kernel->second, workspace.input_bf16,
            gate_weight_bf16, up_weight_bf16,
            workspace.gate, workspace.up,
            workspace.input_bf16.device());
        const auto argument_finish =
            std::chrono::steady_clock::now();
        bf16_grouped_gate_up_argument_setup_ms +=
            std::chrono::duration<double, std::milli>(
                argument_finish - argument_start).count();
        found = bf16_grouped_gate_up_plans.emplace(
            plan_key, std::move(plan)).first;
    } else {
        ++bf16_grouped_gate_up_plan_hits;
    }
    found->second->run(stream);
    ++bf16_grouped_gate_up_dispatches;
    return true;
}

Bf16Plan& bf16_plan(Handle& handle, std::int64_t rows, std::int64_t inner,
                    std::int64_t columns, DType output_dtype, Device device) {
    const Bf16PlanKey registry_key{rows, inner, columns, output_dtype};
    const Bf16PlanCacheKey key{rows, inner, columns, output_dtype,
                               device.index()};
    const auto found = bf16_plans.find(key);
    if (found != bf16_plans.end()) {
        ++bf16_plan_hits;
        return *found->second;
    }
    ++bf16_plan_misses;
    const auto registered = bf16_algorithm_registry.find(registry_key);
    const auto solution = registered == bf16_algorithm_registry.end()
                              ? std::optional<int>{}
                              : std::optional<int>{registered->second};
    auto plan = std::make_unique<Bf16Plan>(
        handle, rows, inner, columns, output_dtype, device, solution);
    auto* result = plan.get();
    bf16_plans.emplace(key, std::move(plan));
    return *result;
}

void hipblaslt_matmul_out(Tensor& output, const Tensor& left,
                          const Tensor& right, bool transpose_left,
                          bool transpose_right, const OpContext& context,
                          float alpha = 1.0F) {
    const auto output_shape = matmul_output_shape(
        left, right, transpose_left, transpose_right);
    if (!left.device().is_hip() || output.device() != left.device() ||
        output.dtype() != left.dtype() || output.shape() != output_shape ||
        !output.is_contiguous()) {
        throw std::invalid_argument(
            "hipBLASLt output must be a matching contiguous Tensor on the input HIP device");
    }
    const auto output_storage = output.storage();
    const auto left_storage = left.storage();
    const auto right_storage = right.storage();
    if (output_storage.data() != nullptr &&
        (output_storage.data() == left_storage.data() ||
         output_storage.data() == right_storage.data())) {
        throw std::invalid_argument("hipBLASLt output must not alias an input Storage");
    }
    const auto rank = static_cast<std::size_t>(left.ndim());
    std::int64_t batches = 1;
    for (std::size_t dimension = 0; dimension + 2 < rank; ++dimension) {
        batches *= left.shape()[dimension];
    }
    if (batches > std::numeric_limits<std::int32_t>::max()) {
        throw std::overflow_error("hipBLASLt matmul batch count exceeds int32");
    }
    const auto left_rows = left.shape()[rank - 2];
    const auto left_columns = left.shape()[rank - 1];
    const auto right_rows = right.shape()[rank - 2];
    const auto right_columns = right.shape()[rank - 1];
    const auto rows = transpose_left ? left_columns : left_rows;
    const auto columns = transpose_right ? right_rows : right_columns;
    auto& handle = handle_for_device(left.device());
    // The row-major expression is submitted as C^T=op(right)^T*op(left)^T.
    // Physical row-major memory is a column-major view of its transpose, so
    // the user's transpose flags map directly to hipBLASLt A(right) and B(left).
    MatmulDescription operation(transpose_right, transpose_left);
    const auto data_type = hip_dtype(left.dtype());
    Layout matrix_b(data_type, static_cast<std::uint64_t>(right_columns),
                    static_cast<std::uint64_t>(right_rows), right_columns);
    Layout matrix_a(data_type, static_cast<std::uint64_t>(left_columns),
                    static_cast<std::uint64_t>(left_rows), left_columns);
    Layout matrix_c(data_type, static_cast<std::uint64_t>(columns), static_cast<std::uint64_t>(rows),
                    columns);
    if (batches > 1) {
        const auto batch_count = static_cast<std::int32_t>(batches);
        matrix_b.set_batch(batch_count, right_rows * right_columns);
        matrix_a.set_batch(batch_count, left_rows * left_columns);
        matrix_c.set_batch(batch_count, rows * columns);
    }
    const float beta = 0.0F;
    const hipblasLtMatmulAlgo_t* algorithm = nullptr;
    if (left.dtype() == DType::Float32 &&
        !fp32_solution_registry.empty()) {
        const auto key = make_fp32_matmul_solution_key(
            left.shape(), right.shape(), left.device(), transpose_left,
            transpose_right, context, alpha);
        algorithm = registered_fp32_solution(
            handle, key, left.device().index(), operation.get(),
            matrix_b.get(), matrix_a.get(), matrix_c.get(), &alpha, &beta,
            context);
    }
    check_status(hipblasLtMatmul(
                     handle.get(), operation.get(), &alpha, right.data(), matrix_b.get(),
                     left.data(), matrix_a.get(), &beta, output.data(), matrix_c.get(),
                     output.data(), matrix_c.get(), algorithm, context.workspace,
                     context.workspace_bytes,
                     reinterpret_cast<hipStream_t>(context.native_stream(left.device()))),
                 "hipblasLtMatmul");
}

Tensor hipblaslt_matmul(const Tensor& left, const Tensor& right,
                        bool transpose_left, bool transpose_right,
                        const OpContext& context, float alpha = 1.0F) {
    Tensor output(matmul_output_shape(left, right, transpose_left, transpose_right),
                  left.dtype(), left.device());
    hipblaslt_matmul_out(output, left, right, transpose_left, transpose_right,
                         context, alpha);
    return output;
}

void validate_bf16_matmul_output(
    const Tensor& output, const Tensor& left, const Tensor& right) {
    if (!left.device().is_hip() || right.device() != left.device() ||
        left.dtype() != DType::BFloat16 || right.dtype() != DType::BFloat16 ||
        left.ndim() != 2 || right.ndim() != 2 || !left.is_contiguous() ||
        !right.is_contiguous() || left.shape()[1] != right.shape()[0]) {
        throw std::invalid_argument(
            "BF16 mixed matmul requires matching contiguous 2D HIP tensors");
    }
    const auto rows = left.shape()[0];
    const auto columns = right.shape()[1];
    if (output.device() != left.device() ||
        (output.dtype() != DType::Float32 &&
         output.dtype() != DType::BFloat16) ||
        output.shape() != Shape({rows, columns}) || !output.is_contiguous()) {
        throw std::invalid_argument("BF16 matmul output must be FP32 or BF16");
    }
    const auto output_storage = output.storage();
    if (output_storage.data() == left.storage().data() ||
        output_storage.data() == right.storage().data()) {
        throw std::invalid_argument("BF16 matmul output must not alias an input Storage");
    }
}

void hipblaslt_bf16_matmul_out(
    Tensor& output, const Tensor& left, const Tensor& right,
    Tensor* output_fallback_bf16, const OpContext& context) {
    validate_bf16_matmul_output(output, left, right);
    const auto rows = left.shape()[0];
    const auto inner = left.shape()[1];
    const auto columns = right.shape()[1];
    const auto output_dtype = output.dtype();
    const MatmulShapeKey shape{rows, inner, columns};
    const auto run = [&](Tensor& destination) {
        auto& handle = handle_for_device(left.device());
        auto& plan = bf16_plan(handle, rows, inner, columns,
                               destination.dtype(), left.device());
        const float alpha = 1.0F;
        const float beta = 0.0F;
        return hipblasLtMatmul(
            handle.get(), plan.operation(), &alpha,
            right.data(), plan.matrix_b(), left.data(), plan.matrix_a(),
            &beta, destination.data(), plan.matrix_c(), destination.data(),
            plan.matrix_c(), plan.algorithm(),
            plan.algorithm() == nullptr ? context.workspace : plan.workspace(),
            plan.algorithm() == nullptr ? context.workspace_bytes
                                        : plan.workspace_bytes(),
            reinterpret_cast<hipStream_t>(
                context.native_stream(left.device())));
    };
    Tensor local_fallback;
    const auto run_fallback = [&] {
        auto* fallback = output_fallback_bf16;
        if (fallback == nullptr) {
            local_fallback = Tensor(
                {rows, columns}, DType::BFloat16, left.device());
            fallback = &local_fallback;
        }
        validate_bf16_matmul_output(*fallback, left, right);
        if (fallback->dtype() != DType::BFloat16 ||
            fallback->storage().data() == output.storage().data()) {
            throw std::invalid_argument(
                "BF16 matmul fallback must be distinct caller-owned BF16 Storage");
        }
        check_status(run(*fallback),
                     "hipblasLtMatmul(BF16 fallback)");
        cast_out_(*fallback, output, context);
    };
    if (output_dtype == DType::Float32) {
        const auto found = bf16_fp32_direct_registry.find(shape);
        if (found != bf16_fp32_direct_registry.end() && !found->second) {
            run_fallback();
            return;
        }
    }
    const auto status = run(output);
    if (status == HIPBLAS_STATUS_SUCCESS) {
        if (output_dtype == DType::Float32) {
            bf16_fp32_direct_registry[shape] = true;
        }
        return;
    }
    if (output_dtype == DType::Float32 &&
        (status == HIPBLAS_STATUS_INTERNAL_ERROR ||
         status == HIPBLAS_STATUS_NOT_SUPPORTED)) {
        bf16_fp32_direct_registry[shape] = false;
        run_fallback();
        return;
    }
    check_status(status, "hipblasLtMatmul(BF16)");
}

Tensor hipblaslt_bf16_matmul(const Tensor& left, const Tensor& right,
                             DType output_dtype, const OpContext& context) {
    if (left.ndim() != 2 || right.ndim() != 2) {
        throw std::invalid_argument(
            "BF16 mixed matmul requires matching contiguous 2D HIP tensors");
    }
    Tensor output({left.shape()[0], right.shape()[1]}, output_dtype,
                  left.device());
    hipblaslt_bf16_matmul_out(output, left, right, nullptr, context);
    return output;
}

Tensor hipblaslt_fp8_matmul(const ScaledTensor& left, const ScaledTensor& right,
                            DType output_dtype, const OpContext& context) {
    if (!left.values.device().is_hip() || right.values.device() != left.values.device() ||
        !is_fp8_fnuz(left.values.dtype()) || !is_fp8_fnuz(right.values.dtype()) ||
        left.values.ndim() != 2 || right.values.ndim() != 2 ||
        !left.values.is_contiguous() || !right.values.is_contiguous()) {
        throw std::invalid_argument("FP8 hipBLASLt matmul requires contiguous 2D FNUZ tensors");
    }
    if (output_dtype != DType::Float32 && output_dtype != DType::Float16 &&
        output_dtype != DType::BFloat16) {
        throw std::invalid_argument("FP8 matmul output must be FP32, FP16, or BF16");
    }
    const auto rows = left.values.shape()[0];
    const auto inner = left.values.shape()[1];
    const auto columns = right.values.shape()[1];
    if (right.values.shape()[0] != inner) throw std::invalid_argument("FP8 matmul inner mismatch");
    const auto valid_left_scale =
        (left.scale_mode == Fp8ScaleMode::Scalar && left.scale.numel() == 1) ||
        (left.scale_mode == Fp8ScaleMode::OuterRow &&
         left.scale.numel() == rows);
    const auto valid_right_scale =
        (right.scale_mode == Fp8ScaleMode::Scalar && right.scale.numel() == 1) ||
        (right.scale_mode == Fp8ScaleMode::OuterColumn &&
         right.scale.numel() == columns);
    if (!valid_left_scale || !valid_right_scale) {
        throw std::invalid_argument(
            "FP8 matmul supports scalar/outer-row left and scalar/outer-column right scales");
    }
    const MatmulShapeKey shape{rows, inner, columns};
    const auto bf16_software_fallback = [&] {
        ++fp8_software_fallback_calls;
        const auto left_bf16 = dequantize_fp8(
            left, DType::BFloat16, context);
        const auto right_bf16 = dequantize_fp8(
            right, DType::BFloat16, context);
        return bf16_matmul_output(
            left_bf16, right_bf16, output_dtype, context);
    };
    const auto outer_row_software_fallback = [&] {
        ++fp8_outer_row_fallback_calls;
        return bf16_software_fallback();
    };
    if (left.scale_mode == Fp8ScaleMode::OuterRow &&
        fp8_outer_row_native.has_value() && !*fp8_outer_row_native) {
        return outer_row_software_fallback();
    }
    const auto native = fp8_native_matrix_registry.find(shape);
    if (native != fp8_native_matrix_registry.end() && !native->second) {
        return left.scale_mode == Fp8ScaleMode::OuterRow
                   ? outer_row_software_fallback()
                   : bf16_software_fallback();
    }
    if (output_dtype == DType::Float32) {
        const auto found = fp8_fp32_direct_registry.find(shape);
        if (found != fp8_fp32_direct_registry.end() && !found->second) {
            return cast(hipblaslt_fp8_matmul(
                            left, right, DType::BFloat16, context),
                        DType::Float32, context);
        }
    }
    Tensor output({rows, columns}, output_dtype, left.values.device());
    auto& handle = handle_for_device(left.values.device());
    MatmulDescription operation;
    // Row-major C=A*B is submitted as column-major C^T=B^T*A^T, so scale A belongs
    // to the user-visible right operand and scale B to the left operand.
    set_scale_pointer(operation.get(), HIPBLASLT_MATMUL_DESC_A_SCALE_POINTER,
                      right.scale.data());
    set_scale_pointer(operation.get(), HIPBLASLT_MATMUL_DESC_B_SCALE_POINTER,
                      left.scale.data());
    if (left.scale_mode == Fp8ScaleMode::OuterRow) {
        set_scale_mode(operation.get(), HIPBLASLT_MATMUL_DESC_B_SCALE_MODE,
                       HIPBLASLT_MATMUL_MATRIX_SCALE_OUTER_VEC_32F);
    }
    const auto try_native_output_column =
        right.scale_mode == Fp8ScaleMode::OuterColumn &&
        (!fp8_output_column_native.has_value() ||
         *fp8_output_column_native);
    if (try_native_output_column) {
        set_scale_mode(operation.get(), HIPBLASLT_MATMUL_DESC_A_SCALE_MODE,
                       HIPBLASLT_MATMUL_MATRIX_SCALE_OUTER_VEC_32F);
    }
    Layout matrix_b(hip_dtype(right.values.dtype()),
                    static_cast<std::uint64_t>(columns),
                    static_cast<std::uint64_t>(inner), columns);
    Layout matrix_a(hip_dtype(left.values.dtype()),
                    static_cast<std::uint64_t>(inner),
                    static_cast<std::uint64_t>(rows), inner);
    Layout matrix_c(hip_dtype(output_dtype), static_cast<std::uint64_t>(columns),
                    static_cast<std::uint64_t>(rows), columns);
    const float alpha = 1.0F;
    const float beta = 0.0F;
    const auto status = hipblasLtMatmul(
        handle.get(), operation.get(), &alpha, right.values.data(), matrix_b.get(),
        left.values.data(), matrix_a.get(), &beta, output.data(), matrix_c.get(),
        output.data(), matrix_c.get(), nullptr, context.workspace,
        context.workspace_bytes,
        reinterpret_cast<hipStream_t>(context.native_stream(left.values.device())));
    if (status == HIPBLAS_STATUS_SUCCESS) {
        if (left.scale_mode == Fp8ScaleMode::OuterRow) {
            fp8_outer_row_native = true;
        }
        fp8_native_matrix_registry[shape] = true;
        if (output_dtype == DType::Float32) {
            fp8_fp32_direct_registry[shape] = true;
        }
        if (try_native_output_column) {
            fp8_output_column_native = true;
        } else if (right.scale_mode == Fp8ScaleMode::OuterColumn) {
            hip::launch_scale_columns_by_first(
                output.data(), output.dtype(), rows, columns,
                static_cast<const float*>(right.scale.data()),
                context.native_stream(output.device()));
            ++fp8_output_column_scale_calls;
        }
        return output;
    }
    if (try_native_output_column &&
        (status == HIPBLAS_STATUS_INVALID_VALUE ||
         status == HIPBLAS_STATUS_INTERNAL_ERROR ||
         status == HIPBLAS_STATUS_NOT_SUPPORTED)) {
        fp8_output_column_native = false;
        return hipblaslt_fp8_matmul(left, right, output_dtype, context);
    }
    if (left.scale_mode == Fp8ScaleMode::OuterRow &&
        (status == HIPBLAS_STATUS_INVALID_VALUE ||
         status == HIPBLAS_STATUS_INTERNAL_ERROR ||
         status == HIPBLAS_STATUS_NOT_SUPPORTED)) {
        fp8_outer_row_native = false;
        return outer_row_software_fallback();
    }
    if (output_dtype == DType::Float32 &&
        (status == HIPBLAS_STATUS_INTERNAL_ERROR ||
         status == HIPBLAS_STATUS_NOT_SUPPORTED)) {
        fp8_fp32_direct_registry[shape] = false;
        return cast(hipblaslt_fp8_matmul(
                        left, right, DType::BFloat16, context),
                    DType::Float32, context);
    }
    if (output_dtype == DType::BFloat16 &&
        (status == HIPBLAS_STATUS_INTERNAL_ERROR ||
         status == HIPBLAS_STATUS_NOT_SUPPORTED)) {
        fp8_native_matrix_registry[shape] = false;
        return bf16_software_fallback();
    }
    if (status != HIPBLAS_STATUS_SUCCESS) {
        throw std::runtime_error(
            "hipblasLtMatmul(FP8) failed for shape " +
            std::to_string(rows) + "x" + std::to_string(inner) + "x" +
            std::to_string(columns) + " output=" +
            std::string(dtype_name(output_dtype)) +
            " status=" + std::to_string(static_cast<int>(status)));
    }
    return output;
}

enum class AttentionLayoutMode { ProbabilityValue, ProbabilityGradient, ValueGradient };

using AttentionLayoutPlanKey =
    std::tuple<AttentionLayoutMode, std::int64_t, std::int64_t,
               std::int64_t, int>;

class AttentionLayoutPlan {
public:
    AttentionLayoutPlan(AttentionLayoutMode mode, std::int64_t heads,
                        std::int64_t sequence, std::int64_t width)
        : operation_(mode == AttentionLayoutMode::ProbabilityGradient,
                     mode == AttentionLayoutMode::ValueGradient),
          matrix_right_(HIP_R_32F, static_cast<std::uint64_t>(width),
                        static_cast<std::uint64_t>(sequence), heads * width),
          matrix_left_(
              HIP_R_32F,
              static_cast<std::uint64_t>(
                  mode == AttentionLayoutMode::ProbabilityGradient
                      ? width : sequence),
              static_cast<std::uint64_t>(sequence),
              mode == AttentionLayoutMode::ProbabilityGradient
                  ? heads * width : sequence),
          matrix_output_(
              HIP_R_32F,
              static_cast<std::uint64_t>(
                  mode == AttentionLayoutMode::ProbabilityGradient
                      ? sequence : width),
              static_cast<std::uint64_t>(sequence),
              mode == AttentionLayoutMode::ProbabilityGradient
                  ? sequence : heads * width) {
        const auto batch_count = static_cast<std::int32_t>(heads);
        matrix_right_.set_batch(batch_count, width);
        matrix_left_.set_batch(
            batch_count,
            mode == AttentionLayoutMode::ProbabilityGradient
                ? width : sequence * sequence);
        matrix_output_.set_batch(
            batch_count,
            mode == AttentionLayoutMode::ProbabilityGradient
                ? sequence * sequence : width);
    }

    [[nodiscard]] hipblasLtMatmulDesc_t operation() const noexcept {
        return operation_.get();
    }
    [[nodiscard]] hipblasLtMatrixLayout_t matrix_right() const noexcept {
        return matrix_right_.get();
    }
    [[nodiscard]] hipblasLtMatrixLayout_t matrix_left() const noexcept {
        return matrix_left_.get();
    }
    [[nodiscard]] hipblasLtMatrixLayout_t matrix_output() const noexcept {
        return matrix_output_.get();
    }

private:
    MatmulDescription operation_;
    Layout matrix_right_;
    Layout matrix_left_;
    Layout matrix_output_;
};

thread_local bool attention_layout_cache_enabled = false;
thread_local std::map<AttentionLayoutPlanKey,
                      std::unique_ptr<AttentionLayoutPlan>> attention_layout_plans;
thread_local std::size_t attention_layout_plan_hits = 0;
thread_local std::size_t attention_layout_plan_misses = 0;

Handle& attention_layout_handle(Device device) {
    return handle_for_device(device);
}

AttentionLayoutPlan& attention_layout_plan(
    AttentionLayoutMode mode, std::int64_t heads, std::int64_t sequence,
    std::int64_t width, Device device) {
    const AttentionLayoutPlanKey key{
        mode, heads, sequence, width, device.index()};
    const auto found = attention_layout_plans.find(key);
    if (found != attention_layout_plans.end()) {
        ++attention_layout_plan_hits;
        return *found->second;
    }
    ++attention_layout_plan_misses;
    auto plan = std::make_unique<AttentionLayoutPlan>(
        mode, heads, sequence, width);
    auto* result = plan.get();
    attention_layout_plans.emplace(key, std::move(plan));
    return *result;
}

Tensor hipblaslt_attention_probability_value_bthd(
    const Tensor& probabilities, const Tensor& value,
    const OpContext& context) {
    const auto batches = probabilities.shape()[0];
    const auto heads = probabilities.shape()[1];
    const auto sequence = probabilities.shape()[2];
    const auto width = value.shape()[3];
    if (heads > std::numeric_limits<std::int32_t>::max()) {
        throw std::overflow_error("Attention head count exceeds hipBLASLt batch range");
    }
    Tensor output({batches, sequence, heads, width}, DType::Float32,
                  probabilities.device());
    std::unique_ptr<AttentionLayoutPlan> ephemeral;
    AttentionLayoutPlan* plan = nullptr;
    if (attention_layout_cache_enabled) {
        plan = &attention_layout_plan(
            AttentionLayoutMode::ProbabilityValue, heads, sequence, width,
            probabilities.device());
    } else {
        ephemeral = std::make_unique<AttentionLayoutPlan>(
            AttentionLayoutMode::ProbabilityValue, heads, sequence, width);
        plan = ephemeral.get();
    }
    const auto probability_batch_elements = heads * sequence * sequence;
    const auto value_batch_elements = sequence * heads * width;
    const auto* probability_data =
        static_cast<const float*>(probabilities.data());
    const auto* value_data = static_cast<const float*>(value.data());
    auto* output_data = static_cast<float*>(output.data());
    const float alpha = 1.0F;
    const float beta = 0.0F;
    for (std::int64_t batch = 0; batch < batches; ++batch) {
        check_status(
            hipblasLtMatmul(
                attention_layout_handle(probabilities.device()).get(),
                plan->operation(), &alpha,
                value_data + batch * value_batch_elements, plan->matrix_right(),
                probability_data + batch * probability_batch_elements,
                plan->matrix_left(), &beta,
                output_data + batch * value_batch_elements,
                plan->matrix_output(),
                output_data + batch * value_batch_elements,
                plan->matrix_output(),
                nullptr, context.workspace, context.workspace_bytes,
                reinterpret_cast<hipStream_t>(
                    context.native_stream(probabilities.device()))),
            "hipblasLtMatmul(Attention BTHD P*V)");
    }
    return output;
}

Tensor hipblaslt_attention_probability_value_gqa_bthd(
    const Tensor& probabilities, const Tensor& value, std::int64_t repeats,
    const OpContext& context) {
    const auto batches = probabilities.shape()[0];
    const auto heads = probabilities.shape()[1];
    const auto sequence = probabilities.shape()[2];
    const auto kv_heads = value.shape()[2];
    const auto width = value.shape()[3];
    Tensor output({batches, sequence, heads, width}, DType::Float32,
                  probabilities.device());
    MatmulDescription operation;
    Layout matrix_value(HIP_R_32F, static_cast<std::uint64_t>(width),
                        static_cast<std::uint64_t>(sequence), kv_heads * width);
    Layout matrix_probability(HIP_R_32F,
                              static_cast<std::uint64_t>(sequence),
                              static_cast<std::uint64_t>(sequence), sequence);
    Layout matrix_context(HIP_R_32F, static_cast<std::uint64_t>(width),
                          static_cast<std::uint64_t>(sequence), heads * width);
    const auto batch_count = static_cast<std::int32_t>(repeats);
    matrix_value.set_batch(batch_count, 0);
    matrix_probability.set_batch(batch_count, sequence * sequence);
    matrix_context.set_batch(batch_count, width);
    const auto probability_head_elements = sequence * sequence;
    const auto probability_batch_elements = heads * probability_head_elements;
    const auto value_batch_elements = sequence * kv_heads * width;
    const auto context_batch_elements = sequence * heads * width;
    const auto* probability_data =
        static_cast<const float*>(probabilities.data());
    const auto* value_data = static_cast<const float*>(value.data());
    auto* output_data = static_cast<float*>(output.data());
    const float alpha = 1.0F;
    const float beta = 0.0F;
    for (std::int64_t batch = 0; batch < batches; ++batch) {
        for (std::int64_t kv_head = 0; kv_head < kv_heads; ++kv_head) {
            const auto head = kv_head * repeats;
            check_status(
                hipblasLtMatmul(
                    attention_layout_handle(probabilities.device()).get(),
                    operation.get(), &alpha,
                    value_data + batch * value_batch_elements + kv_head * width,
                    matrix_value.get(),
                    probability_data + batch * probability_batch_elements +
                        head * probability_head_elements,
                    matrix_probability.get(), &beta,
                    output_data + batch * context_batch_elements + head * width,
                    matrix_context.get(),
                    output_data + batch * context_batch_elements + head * width,
                    matrix_context.get(), nullptr, context.workspace,
                    context.workspace_bytes,
                    reinterpret_cast<hipStream_t>(
                        context.native_stream(probabilities.device()))),
                "hipblasLtMatmul(Attention GQA BTHD P*V)");
        }
    }
    return output;
}

Tensor hipblaslt_attention_probability_gradient_bthd(
    const Tensor& output_gradient, const Tensor& value,
    const OpContext& context) {
    const auto batches = output_gradient.shape()[0];
    const auto sequence = output_gradient.shape()[1];
    const auto heads = output_gradient.shape()[2];
    const auto width = output_gradient.shape()[3];
    Tensor output({batches, heads, sequence, sequence}, DType::Float32,
                  output_gradient.device());
    std::unique_ptr<AttentionLayoutPlan> ephemeral;
    AttentionLayoutPlan* plan = nullptr;
    if (attention_layout_cache_enabled) {
        plan = &attention_layout_plan(
            AttentionLayoutMode::ProbabilityGradient, heads, sequence, width,
            output_gradient.device());
    } else {
        ephemeral = std::make_unique<AttentionLayoutPlan>(
            AttentionLayoutMode::ProbabilityGradient, heads, sequence, width);
        plan = ephemeral.get();
    }
    const auto value_batch_elements = sequence * heads * width;
    const auto probability_batch_elements = heads * sequence * sequence;
    const auto* gradient_data =
        static_cast<const float*>(output_gradient.data());
    const auto* value_data = static_cast<const float*>(value.data());
    auto* output_data = static_cast<float*>(output.data());
    const float alpha = 1.0F;
    const float beta = 0.0F;
    for (std::int64_t batch = 0; batch < batches; ++batch) {
        check_status(
            hipblasLtMatmul(
                attention_layout_handle(output_gradient.device()).get(),
                plan->operation(), &alpha,
                value_data + batch * value_batch_elements, plan->matrix_right(),
                gradient_data + batch * value_batch_elements,
                plan->matrix_left(), &beta,
                output_data + batch * probability_batch_elements,
                plan->matrix_output(),
                output_data + batch * probability_batch_elements,
                plan->matrix_output(), nullptr, context.workspace,
                context.workspace_bytes,
                reinterpret_cast<hipStream_t>(
                    context.native_stream(output_gradient.device()))),
            "hipblasLtMatmul(Attention BTHD dP)");
    }
    return output;
}

Tensor hipblaslt_attention_probability_gradient_gqa_bthd(
    const Tensor& output_gradient, const Tensor& value,
    std::int64_t repeats, const OpContext& context) {
    const auto batches = output_gradient.shape()[0];
    const auto sequence = output_gradient.shape()[1];
    const auto heads = output_gradient.shape()[2];
    const auto kv_heads = value.shape()[2];
    const auto width = output_gradient.shape()[3];
    Tensor output({batches, heads, sequence, sequence}, DType::Float32,
                  output_gradient.device());
    MatmulDescription operation(true, false);
    Layout matrix_value(HIP_R_32F, static_cast<std::uint64_t>(width),
                        static_cast<std::uint64_t>(sequence), kv_heads * width);
    Layout matrix_gradient(HIP_R_32F, static_cast<std::uint64_t>(width),
                           static_cast<std::uint64_t>(sequence), heads * width);
    Layout matrix_probability(HIP_R_32F,
                              static_cast<std::uint64_t>(sequence),
                              static_cast<std::uint64_t>(sequence), sequence);
    const auto batch_count = static_cast<std::int32_t>(repeats);
    matrix_value.set_batch(batch_count, 0);
    matrix_gradient.set_batch(batch_count, width);
    matrix_probability.set_batch(batch_count, sequence * sequence);
    const auto value_batch_elements = sequence * kv_heads * width;
    const auto gradient_batch_elements = sequence * heads * width;
    const auto probability_head_elements = sequence * sequence;
    const auto probability_batch_elements = heads * probability_head_elements;
    const auto* gradient_data =
        static_cast<const float*>(output_gradient.data());
    const auto* value_data = static_cast<const float*>(value.data());
    auto* output_data = static_cast<float*>(output.data());
    const float alpha = 1.0F;
    const float beta = 0.0F;
    for (std::int64_t batch = 0; batch < batches; ++batch) {
        for (std::int64_t kv_head = 0; kv_head < kv_heads; ++kv_head) {
            const auto head = kv_head * repeats;
            check_status(
                hipblasLtMatmul(
                    attention_layout_handle(output_gradient.device()).get(),
                    operation.get(), &alpha,
                    value_data + batch * value_batch_elements + kv_head * width,
                    matrix_value.get(),
                    gradient_data + batch * gradient_batch_elements + head * width,
                    matrix_gradient.get(), &beta,
                    output_data + batch * probability_batch_elements +
                        head * probability_head_elements,
                    matrix_probability.get(),
                    output_data + batch * probability_batch_elements +
                        head * probability_head_elements,
                    matrix_probability.get(), nullptr, context.workspace,
                    context.workspace_bytes,
                    reinterpret_cast<hipStream_t>(
                        context.native_stream(output_gradient.device()))),
                "hipblasLtMatmul(Attention GQA BTHD dP)");
        }
    }
    return output;
}

Tensor hipblaslt_attention_value_gradient_bthd(
    const Tensor& probabilities, const Tensor& output_gradient,
    const OpContext& context) {
    const auto batches = output_gradient.shape()[0];
    const auto sequence = output_gradient.shape()[1];
    const auto heads = output_gradient.shape()[2];
    const auto width = output_gradient.shape()[3];
    Tensor output(output_gradient.shape(), DType::Float32,
                  output_gradient.device());
    std::unique_ptr<AttentionLayoutPlan> ephemeral;
    AttentionLayoutPlan* plan = nullptr;
    if (attention_layout_cache_enabled) {
        plan = &attention_layout_plan(
            AttentionLayoutMode::ValueGradient, heads, sequence, width,
            output_gradient.device());
    } else {
        ephemeral = std::make_unique<AttentionLayoutPlan>(
            AttentionLayoutMode::ValueGradient, heads, sequence, width);
        plan = ephemeral.get();
    }
    const auto value_batch_elements = sequence * heads * width;
    const auto probability_batch_elements = heads * sequence * sequence;
    const auto* gradient_data =
        static_cast<const float*>(output_gradient.data());
    const auto* probability_data =
        static_cast<const float*>(probabilities.data());
    auto* output_data = static_cast<float*>(output.data());
    const float alpha = 1.0F;
    const float beta = 0.0F;
    for (std::int64_t batch = 0; batch < batches; ++batch) {
        check_status(
            hipblasLtMatmul(
                attention_layout_handle(probabilities.device()).get(),
                plan->operation(), &alpha,
                gradient_data + batch * value_batch_elements,
                plan->matrix_right(),
                probability_data + batch * probability_batch_elements,
                plan->matrix_left(), &beta,
                output_data + batch * value_batch_elements,
                plan->matrix_output(),
                output_data + batch * value_batch_elements,
                plan->matrix_output(),
                nullptr, context.workspace, context.workspace_bytes,
                reinterpret_cast<hipStream_t>(
                    context.native_stream(output_gradient.device()))),
            "hipblasLtMatmul(Attention BTHD dV)");
    }
    return output;
}

}  // namespace
#endif

Bf16PlanCacheStats bf16_plan_cache_stats() noexcept {
#if MICROLLM_HAS_HIPBLASLT
    return {bf16_plans.size(), bf16_plan_hits, bf16_plan_misses};
#else
    return {};
#endif
}

void clear_bf16_plan_cache() noexcept {
#if MICROLLM_HAS_HIPBLASLT
    bf16_plans.clear();
    bf16_plan_hits = 0;
    bf16_plan_misses = 0;
#endif
}

void enable_attention_layout_plan_cache(bool enabled) noexcept {
#if MICROLLM_HAS_HIPBLASLT
    attention_layout_cache_enabled = enabled;
    if (!enabled) {
        attention_layout_plans.clear();
        attention_layout_plan_hits = 0;
        attention_layout_plan_misses = 0;
    }
#else
    (void)enabled;
#endif
}

bool attention_layout_plan_cache_enabled() noexcept {
#if MICROLLM_HAS_HIPBLASLT
    return attention_layout_cache_enabled;
#else
    return false;
#endif
}

AttentionLayoutPlanCacheStats attention_layout_plan_cache_stats() noexcept {
#if MICROLLM_HAS_HIPBLASLT
    return {attention_layout_plans.size(), attention_layout_plan_hits,
            attention_layout_plan_misses};
#else
    return {};
#endif
}

void clear_attention_layout_plan_cache() noexcept {
#if MICROLLM_HAS_HIPBLASLT
    attention_layout_plans.clear();
    attention_layout_plan_hits = 0;
    attention_layout_plan_misses = 0;
#endif
}

void register_bf16_algorithm(std::int64_t rows, std::int64_t inner,
                             std::int64_t columns, DType output_dtype,
                             int solution_index) {
#if MICROLLM_HAS_HIPBLASLT
    if (rows <= 0 || inner <= 0 || columns <= 0 || solution_index < 0 ||
        (output_dtype != DType::Float32 && output_dtype != DType::BFloat16)) {
        throw std::invalid_argument("invalid BF16 algorithm registration");
    }
    bf16_algorithm_registry[{rows, inner, columns, output_dtype}] =
        solution_index;
    bf16_plans.clear();
#else
    (void)rows;
    (void)inner;
    (void)columns;
    (void)output_dtype;
    (void)solution_index;
    throw std::runtime_error("BF16 algorithm registry requires hipBLASLt");
#endif
}

void clear_bf16_algorithm_registry() noexcept {
#if MICROLLM_HAS_HIPBLASLT
    bf16_algorithm_registry.clear();
    bf16_plans.clear();
#endif
}

std::size_t bf16_registered_algorithm_count() noexcept {
#if MICROLLM_HAS_HIPBLASLT
    return bf16_algorithm_registry.size();
#else
    return 0;
#endif
}

void register_bf16_grouped_qkv_algorithm(
    const Bf16GroupedQkvKey& key, int solution_index) {
#if MICROLLM_HAS_HIPBLASLT
    validate_bf16_grouped_qkv_key(key);
    if (solution_index < 0) {
        throw std::invalid_argument(
            "BF16 grouped QKV solution index must be nonnegative");
    }
    bool current_environment = false;
    for (int index = 0; index < runtime::hip_device_count(); ++index) {
        const auto environment = tuning_environment(Device::hip(index));
        if (key.architecture == environment.architecture &&
            key.hip_runtime_version == environment.runtime_version &&
            key.hip_driver_version == environment.driver_version &&
            key.hipblaslt_version == hipblaslt_version()) {
            current_environment = true;
            break;
        }
    }
    if (!current_environment) {
        throw std::invalid_argument(
            "BF16 grouped QKV key does not match a visible environment");
    }
    bf16_grouped_qkv_registry.insert_or_assign(key, solution_index);
    bf16_grouped_qkv_algorithms.clear();
    bf16_grouped_qkv_kernels.clear();
    bf16_grouped_qkv_plans.clear();
#else
    (void)key;
    (void)solution_index;
    throw std::runtime_error("BF16 grouped QKV requires hipBLASLt");
#endif
}

void clear_bf16_grouped_qkv_registry() noexcept {
    bf16_grouped_qkv_registry.clear();
    bf16_grouped_qkv_plan_hits = 0;
    bf16_grouped_qkv_plan_misses = 0;
    bf16_grouped_qkv_dispatches = 0;
    bf16_grouped_qkv_retained_query_key_dispatches = 0;
    bf16_grouped_qkv_algorithm_hits = 0;
    bf16_grouped_qkv_algorithm_misses = 0;
    bf16_grouped_qkv_kernel_hits = 0;
    bf16_grouped_qkv_kernel_misses = 0;
    bf16_grouped_qkv_kernel_setup_ms = 0.0;
    bf16_grouped_qkv_argument_setup_ms = 0.0;
#if MICROLLM_HAS_HIPBLASLT
    bf16_grouped_qkv_algorithms.clear();
    bf16_grouped_qkv_kernels.clear();
    bf16_grouped_qkv_plans.clear();
#endif
}

Bf16GroupedQkvStats bf16_grouped_qkv_stats() noexcept {
    return {
        .registered_entries = bf16_grouped_qkv_registry.size(),
#if MICROLLM_HAS_HIPBLASLT
        .algorithm_entries = bf16_grouped_qkv_algorithms.size(),
#else
        .algorithm_entries = 0,
#endif
        .algorithm_hits = bf16_grouped_qkv_algorithm_hits,
        .algorithm_misses = bf16_grouped_qkv_algorithm_misses,
#if MICROLLM_HAS_HIPBLASLT
        .kernel_entries = bf16_grouped_qkv_kernels.size(),
#else
        .kernel_entries = 0,
#endif
        .kernel_hits = bf16_grouped_qkv_kernel_hits,
        .kernel_misses = bf16_grouped_qkv_kernel_misses,
#if MICROLLM_HAS_HIPBLASLT
        .plan_entries = bf16_grouped_qkv_plans.size(),
#else
        .plan_entries = 0,
#endif
        .plan_hits = bf16_grouped_qkv_plan_hits,
        .plan_misses = bf16_grouped_qkv_plan_misses,
        .dispatches = bf16_grouped_qkv_dispatches,
        .retained_query_key_dispatches =
            bf16_grouped_qkv_retained_query_key_dispatches,
        .kernel_setup_ms = bf16_grouped_qkv_kernel_setup_ms,
        .argument_setup_ms = bf16_grouped_qkv_argument_setup_ms,
    };
}

void register_bf16_grouped_gate_up_algorithm(
    const Bf16GroupedGateUpKey& key, int solution_index) {
#if MICROLLM_HAS_HIPBLASLT
    validate_bf16_grouped_gate_up_key(key);
    if (solution_index < 0) {
        throw std::invalid_argument(
            "BF16 grouped gate/up solution index must be nonnegative");
    }
    bool current_environment = false;
    for (int index = 0; index < runtime::hip_device_count(); ++index) {
        const auto environment = tuning_environment(Device::hip(index));
        if (key.architecture == environment.architecture &&
            key.hip_runtime_version == environment.runtime_version &&
            key.hip_driver_version == environment.driver_version &&
            key.hipblaslt_version == hipblaslt_version()) {
            current_environment = true;
            break;
        }
    }
    if (!current_environment) {
        throw std::invalid_argument(
            "BF16 grouped gate/up key does not match a visible environment");
    }
    bf16_grouped_gate_up_registry.insert_or_assign(
        key, solution_index);
    bf16_grouped_gate_up_algorithms.clear();
    bf16_grouped_gate_up_kernels.clear();
    bf16_grouped_gate_up_plans.clear();
#else
    (void)key;
    (void)solution_index;
    throw std::runtime_error(
        "BF16 grouped gate/up requires hipBLASLt");
#endif
}

void clear_bf16_grouped_gate_up_registry() noexcept {
    bf16_grouped_gate_up_registry.clear();
    bf16_grouped_gate_up_plan_hits = 0;
    bf16_grouped_gate_up_plan_misses = 0;
    bf16_grouped_gate_up_dispatches = 0;
    bf16_grouped_gate_up_algorithm_hits = 0;
    bf16_grouped_gate_up_algorithm_misses = 0;
    bf16_grouped_gate_up_kernel_hits = 0;
    bf16_grouped_gate_up_kernel_misses = 0;
    bf16_grouped_gate_up_kernel_setup_ms = 0.0;
    bf16_grouped_gate_up_argument_setup_ms = 0.0;
#if MICROLLM_HAS_HIPBLASLT
    bf16_grouped_gate_up_algorithms.clear();
    bf16_grouped_gate_up_kernels.clear();
    bf16_grouped_gate_up_plans.clear();
#endif
}

Bf16GroupedGateUpStats bf16_grouped_gate_up_stats() noexcept {
    return {
        .registered_entries = bf16_grouped_gate_up_registry.size(),
#if MICROLLM_HAS_HIPBLASLT
        .algorithm_entries = bf16_grouped_gate_up_algorithms.size(),
#else
        .algorithm_entries = 0,
#endif
        .algorithm_hits = bf16_grouped_gate_up_algorithm_hits,
        .algorithm_misses = bf16_grouped_gate_up_algorithm_misses,
#if MICROLLM_HAS_HIPBLASLT
        .kernel_entries = bf16_grouped_gate_up_kernels.size(),
#else
        .kernel_entries = 0,
#endif
        .kernel_hits = bf16_grouped_gate_up_kernel_hits,
        .kernel_misses = bf16_grouped_gate_up_kernel_misses,
#if MICROLLM_HAS_HIPBLASLT
        .plan_entries = bf16_grouped_gate_up_plans.size(),
#else
        .plan_entries = 0,
#endif
        .plan_hits = bf16_grouped_gate_up_plan_hits,
        .plan_misses = bf16_grouped_gate_up_plan_misses,
        .dispatches = bf16_grouped_gate_up_dispatches,
        .kernel_setup_ms = bf16_grouped_gate_up_kernel_setup_ms,
        .argument_setup_ms =
            bf16_grouped_gate_up_argument_setup_ms,
    };
}

void enable_bf16_grouped_gate_up_swish(bool enabled) noexcept {
    if (bf16_grouped_gate_up_swish == enabled) return;
    bf16_grouped_gate_up_swish = enabled;
#if MICROLLM_HAS_HIPBLASLT
    bf16_grouped_gate_up_kernels.clear();
    bf16_grouped_gate_up_plans.clear();
#endif
}

bool bf16_grouped_gate_up_swish_enabled() noexcept {
    return bf16_grouped_gate_up_swish;
}

void register_fp32_matmul_solution(
    const Fp32MatmulSolutionKey& key, int solution_index) {
#if MICROLLM_HAS_HIPBLASLT
    validate_fp32_solution_key(key);
    if (solution_index < 0) {
        throw std::invalid_argument(
            "FP32 solution index must be nonnegative");
    }
    bool current_environment = false;
    const auto devices = runtime::hip_device_count();
    for (int index = 0; index < devices; ++index) {
        const auto environment = tuning_environment(Device::hip(index));
        if (key.architecture == environment.architecture &&
            key.hip_runtime_version == environment.runtime_version &&
            key.hip_driver_version == environment.driver_version &&
            key.hipblaslt_version == hipblaslt_version()) {
            current_environment = true;
            break;
        }
    }
    if (!current_environment) {
        throw std::invalid_argument(
            "FP32 solution key does not match a visible current environment");
    }
    fp32_solution_registry.insert_or_assign(key, solution_index);
    fp32_solution_algorithms.clear();
#else
    (void)key;
    (void)solution_index;
    throw std::runtime_error("FP32 solution registry requires hipBLASLt");
#endif
}

void clear_fp32_matmul_solution_registry() noexcept {
    fp32_solution_registry.clear();
    fp32_solution_registry_hits = 0;
    fp32_solution_registry_misses = 0;
    fp32_solution_cache_hits = 0;
    fp32_solution_cache_misses = 0;
    fp32_solution_dispatches = 0;
#if MICROLLM_HAS_HIPBLASLT
    fp32_solution_algorithms.clear();
#endif
}

Fp32MatmulSolutionStats fp32_matmul_solution_stats() noexcept {
    return {
        .registered_entries = fp32_solution_registry.size(),
#if MICROLLM_HAS_HIPBLASLT
        .cached_algorithms = fp32_solution_algorithms.size(),
#else
        .cached_algorithms = 0,
#endif
        .registry_hits = fp32_solution_registry_hits,
        .registry_misses = fp32_solution_registry_misses,
        .cache_hits = fp32_solution_cache_hits,
        .cache_misses = fp32_solution_cache_misses,
        .dispatches = fp32_solution_dispatches,
    };
}

Fp8DispatchStats fp8_dispatch_stats() noexcept {
#if MICROLLM_HAS_HIPBLASLT
    std::size_t native = 0;
    std::size_t fallback = 0;
    for (const auto& [shape, supported] : fp8_native_matrix_registry) {
        (void)shape;
        supported ? ++native : ++fallback;
    }
    return {native, fallback, fp8_software_fallback_calls,
            fp8_outer_row_fallback_calls,
            fp8_outer_row_native.has_value()
                ? *fp8_outer_row_native ? 1 : 0 : -1,
            fp8_output_column_scale_calls,
            fp8_output_column_native.has_value()
                ? *fp8_output_column_native ? 1 : 0 : -1};
#else
    return {};
#endif
}

void clear_fp8_dispatch_registry() noexcept {
#if MICROLLM_HAS_HIPBLASLT
    fp8_fp32_direct_registry.clear();
    fp8_native_matrix_registry.clear();
    fp8_software_fallback_calls = 0;
    fp8_outer_row_native.reset();
    fp8_outer_row_fallback_calls = 0;
    fp8_output_column_scale_calls = 0;
    fp8_output_column_native.reset();
#endif
}

Tensor matmul_with_implementation(const Tensor& left, const Tensor& right,
                                  MatmulImplementation implementation,
                                  const OpContext& context) {
    if (implementation == MatmulImplementation::Auto) {
        implementation = choose_matmul_implementation(
            left, right, false, false, context);
    }
    if (implementation == MatmulImplementation::Readable) return matmul(left, right, context);
#if MICROLLM_HAS_HIPBLASLT
    return hipblaslt_matmul(left, right, false, false, context);
#else
    (void)left;
    (void)right;
    (void)context;
    throw std::runtime_error("microLLM was built without hipBLASLt");
#endif
}

Tensor matmul_with_implementation(const Tensor& left, const Tensor& right,
                                  MatmulImplementation implementation,
                                  bool transpose_left, bool transpose_right,
                                  [[maybe_unused]] const OpContext& context) {
    if (left.ndim() < 2 || right.ndim() != left.ndim() || !is_floating_point(left.dtype()) ||
        right.dtype() != left.dtype() || left.device() != right.device() ||
        !left.is_contiguous() || !right.is_contiguous()) {
        throw std::invalid_argument(
            "transpose-aware matmul requires matching-rank contiguous floating tensors");
    }
    const auto rank = static_cast<std::size_t>(left.ndim());
    for (std::size_t dimension = 0; dimension + 2 < rank; ++dimension) {
        if (left.shape()[dimension] != right.shape()[dimension]) {
            throw std::invalid_argument("transpose-aware matmul batch dimensions must match");
        }
    }
    const auto left_rows = left.shape()[rank - 2];
    const auto left_columns = left.shape()[rank - 1];
    const auto right_rows = right.shape()[rank - 2];
    const auto right_columns = right.shape()[rank - 1];
    const auto rows = transpose_left ? left_columns : left_rows;
    const auto inner = transpose_left ? left_rows : left_columns;
    const auto right_inner = transpose_right ? right_columns : right_rows;
    const auto columns = transpose_right ? right_rows : right_columns;
    if (inner != right_inner) throw std::invalid_argument("matmul inner dimensions mismatch");

    if (implementation == MatmulImplementation::Auto) {
        implementation = choose_matmul_implementation(
            left, right, transpose_left, transpose_right, context);
    }
    if (implementation == MatmulImplementation::HipBLASLt) {
#if MICROLLM_HAS_HIPBLASLT
        return hipblaslt_matmul(left, right, transpose_left, transpose_right, context);
#else
        throw std::runtime_error("microLLM was built without hipBLASLt");
#endif
    }

    if (rank > 2) {
        const auto left_operand = transpose_left
                                      ? left.transpose(-2, -1).contiguous() : left;
        const auto right_operand = transpose_right
                                       ? right.transpose(-2, -1).contiguous() : right;
        return matmul(left_operand, right_operand, context);
    }

    Tensor output({rows, columns}, left.dtype(), left.device());
    if (left.device().is_hip()) {
#if MICROLLM_HAS_HIP
        hip::launch_matmul_transposed_typed(
            left.data(), right.data(), output.data(), left.dtype(),
            left_rows, left_columns, right_rows, right_columns,
            transpose_left, transpose_right, context.native_stream(left.device()));
        return output;
#else
        throw std::runtime_error("microLLM was built without HIP operator support");
#endif
    }

    const auto left_values = left.to_vector();
    const auto right_values = right.to_vector();
    std::vector<float> values(static_cast<std::size_t>(rows * columns), 0.0F);
    for (std::int64_t row = 0; row < rows; ++row) {
        for (std::int64_t column = 0; column < columns; ++column) {
            float sum = 0.0F;
            for (std::int64_t reduction = 0; reduction < inner; ++reduction) {
                const auto left_index = transpose_left
                                            ? reduction * left_columns + row
                                            : row * left_columns + reduction;
                const auto right_index = transpose_right
                                             ? column * right_columns + reduction
                                             : reduction * right_columns + column;
                sum += left_values[static_cast<std::size_t>(left_index)] *
                       right_values[static_cast<std::size_t>(right_index)];
            }
            values[static_cast<std::size_t>(row * columns + column)] = sum;
        }
    }
    return Tensor::from_vector(values, {rows, columns}, left.dtype());
}

void matmul_out_(Tensor& output, const Tensor& left, const Tensor& right,
                 MatmulImplementation implementation, bool transpose_left,
                 bool transpose_right, const OpContext& context) {
    const auto expected_shape = matmul_output_shape(
        left, right, transpose_left, transpose_right);
    if (!output.defined() || output.shape() != expected_shape ||
        output.dtype() != left.dtype() || output.device() != left.device() ||
        !output.is_contiguous()) {
        throw std::invalid_argument(
            "matmul output must match result shape, dtype, device, and contiguity");
    }
    const auto output_storage = output.storage();
    if (output_storage.data() != nullptr &&
        (output_storage.data() == left.storage().data() ||
         output_storage.data() == right.storage().data())) {
        throw std::invalid_argument("matmul output must not alias an input Storage");
    }
    if (implementation == MatmulImplementation::Auto) {
        implementation = choose_matmul_implementation(
            left, right, transpose_left, transpose_right, context);
    }
    if (implementation == MatmulImplementation::HipBLASLt) {
#if MICROLLM_HAS_HIPBLASLT
        hipblaslt_matmul_out(output, left, right, transpose_left,
                             transpose_right, context);
        return;
#else
        throw std::runtime_error("microLLM was built without hipBLASLt");
#endif
    }
    if (left.device().is_hip()) {
        if (left.ndim() != 2) {
            throw std::invalid_argument(
                "readable caller-owned HIP matmul currently requires rank two");
        }
#if MICROLLM_HAS_HIP
        hip::launch_matmul_transposed_typed(
            left.data(), right.data(), output.data(), left.dtype(),
            left.shape()[0], left.shape()[1], right.shape()[0], right.shape()[1],
            transpose_left, transpose_right,
            context.native_stream(left.device()));
        return;
#else
        throw std::runtime_error("microLLM was built without HIP operator support");
#endif
    }
    const auto reference = matmul_with_implementation(
        left, right, MatmulImplementation::Readable,
        transpose_left, transpose_right, context);
    runtime::copy_bytes(
        output.data(), output.device(), reference.data(), reference.device(),
        static_cast<std::size_t>(output.numel()) * dtype_size(output.dtype()));
}

Tensor matmul_scaled_with_implementation(
    const Tensor& left, const Tensor& right, float factor,
    MatmulImplementation implementation, bool transpose_left,
    bool transpose_right, const OpContext& context) {
    if (!std::isfinite(factor)) {
        throw std::invalid_argument("scaled matmul factor must be finite");
    }
    if (implementation == MatmulImplementation::Auto) {
        implementation = choose_matmul_implementation(
            left, right, transpose_left, transpose_right, context);
    }
    if (implementation == MatmulImplementation::HipBLASLt) {
#if MICROLLM_HAS_HIPBLASLT
        return hipblaslt_matmul(
            left, right, transpose_left, transpose_right, context, factor);
#else
        throw std::runtime_error("microLLM was built without hipBLASLt");
#endif
    }
    return scale(
        matmul_with_implementation(
            left, right, MatmulImplementation::Readable,
            transpose_left, transpose_right, context),
        factor, context);
}

Tensor attention_probability_value_bthd(
    const Tensor& probabilities, const Tensor& value,
    const OpContext& context) {
    if (probabilities.dtype() != DType::Float32 ||
        value.dtype() != DType::Float32 || probabilities.device() != value.device() ||
        probabilities.ndim() != 4 || value.ndim() != 4 ||
        probabilities.shape()[0] != value.shape()[0] ||
        probabilities.shape()[1] != value.shape()[2] ||
        probabilities.shape()[2] != probabilities.shape()[3] ||
        probabilities.shape()[2] != value.shape()[1] ||
        probabilities.shape()[2] <= 0 || value.shape()[3] <= 0 ||
        !probabilities.is_contiguous() || !value.is_contiguous()) {
        throw std::invalid_argument(
            "Attention P*V BTHD requires contiguous FP32 probabilities [B,H,T,T] "
            "and value [B,T,H,D]");
    }
    if (probabilities.device().is_hip()) {
#if MICROLLM_HAS_HIPBLASLT
        return hipblaslt_attention_probability_value_bthd(
            probabilities, value, context);
#else
        throw std::runtime_error(
            "Attention P*V BTHD on HIP requires hipBLASLt");
#endif
    }
    const auto value_bhtd = value.transpose(1, 2).contiguous();
    return matmul(probabilities, value_bhtd, context)
        .transpose(1, 2).contiguous();
}

Tensor attention_probability_value_gqa_bthd(
    const Tensor& probabilities, const Tensor& value, std::int64_t repeats,
    const OpContext& context) {
    if (probabilities.dtype() != DType::Float32 ||
        value.dtype() != DType::Float32 || probabilities.device() != value.device() ||
        probabilities.ndim() != 4 || value.ndim() != 4 || repeats <= 0 ||
        probabilities.shape()[0] != value.shape()[0] ||
        probabilities.shape()[1] != value.shape()[2] * repeats ||
        probabilities.shape()[2] != probabilities.shape()[3] ||
        probabilities.shape()[2] != value.shape()[1] ||
        probabilities.shape()[2] <= 0 || value.shape()[3] <= 0 ||
        repeats > std::numeric_limits<std::int32_t>::max() ||
        !probabilities.is_contiguous() || !value.is_contiguous()) {
        throw std::invalid_argument(
            "Attention GQA P*V requires P[B,H,T,T], V[B,T,KV,D], H=KV*repeats");
    }
    if (probabilities.device().is_hip()) {
#if MICROLLM_HAS_HIPBLASLT
        return hipblaslt_attention_probability_value_gqa_bthd(
            probabilities, value, repeats, context);
#else
        throw std::runtime_error("Attention GQA P*V BTHD on HIP requires hipBLASLt");
#endif
    }
    return attention_probability_value_bthd(
        probabilities, repeat_interleave(value, 2, repeats, context), context);
}

Tensor attention_probability_gradient_bthd(
    const Tensor& output_gradient, const Tensor& value,
    const OpContext& context) {
    if (output_gradient.dtype() != DType::Float32 ||
        value.dtype() != DType::Float32 ||
        output_gradient.device() != value.device() ||
        output_gradient.ndim() != 4 || output_gradient.shape() != value.shape() ||
        output_gradient.shape()[1] <= 0 || output_gradient.shape()[2] <= 0 ||
        output_gradient.shape()[3] <= 0 || !output_gradient.is_contiguous() ||
        !value.is_contiguous()) {
        throw std::invalid_argument(
            "Attention dP BTHD requires matching contiguous FP32 dO/value [B,T,H,D]");
    }
    if (output_gradient.shape()[2] > std::numeric_limits<std::int32_t>::max()) {
        throw std::overflow_error("Attention head count exceeds hipBLASLt batch range");
    }
    if (output_gradient.device().is_hip()) {
#if MICROLLM_HAS_HIPBLASLT
        return hipblaslt_attention_probability_gradient_bthd(
            output_gradient, value, context);
#else
        throw std::runtime_error("Attention dP BTHD on HIP requires hipBLASLt");
#endif
    }
    const auto gradient_bhtd = output_gradient.transpose(1, 2).contiguous();
    const auto value_bhtd = value.transpose(1, 2).contiguous();
    return matmul_with_implementation(
        gradient_bhtd, value_bhtd, MatmulImplementation::Readable,
        false, true, context);
}

Tensor attention_probability_gradient_gqa_bthd(
    const Tensor& output_gradient, const Tensor& value,
    std::int64_t repeats, const OpContext& context) {
    if (output_gradient.dtype() != DType::Float32 ||
        value.dtype() != DType::Float32 ||
        output_gradient.device() != value.device() ||
        output_gradient.ndim() != 4 || value.ndim() != 4 || repeats <= 0 ||
        output_gradient.shape()[0] != value.shape()[0] ||
        output_gradient.shape()[1] != value.shape()[1] ||
        output_gradient.shape()[2] != value.shape()[2] * repeats ||
        output_gradient.shape()[3] != value.shape()[3] ||
        repeats > std::numeric_limits<std::int32_t>::max() ||
        !output_gradient.is_contiguous() || !value.is_contiguous()) {
        throw std::invalid_argument(
            "Attention GQA dP requires dO[B,T,H,D], V[B,T,KV,D], H=KV*repeats");
    }
    if (output_gradient.device().is_hip()) {
#if MICROLLM_HAS_HIPBLASLT
        return hipblaslt_attention_probability_gradient_gqa_bthd(
            output_gradient, value, repeats, context);
#else
        throw std::runtime_error("Attention GQA dP BTHD on HIP requires hipBLASLt");
#endif
    }
    return attention_probability_gradient_bthd(
        output_gradient, repeat_interleave(value, 2, repeats, context), context);
}

Tensor attention_value_gradient_bthd(
    const Tensor& probabilities, const Tensor& output_gradient,
    const OpContext& context) {
    if (probabilities.dtype() != DType::Float32 ||
        output_gradient.dtype() != DType::Float32 ||
        probabilities.device() != output_gradient.device() ||
        probabilities.ndim() != 4 || output_gradient.ndim() != 4 ||
        probabilities.shape()[0] != output_gradient.shape()[0] ||
        probabilities.shape()[1] != output_gradient.shape()[2] ||
        probabilities.shape()[2] != probabilities.shape()[3] ||
        probabilities.shape()[2] != output_gradient.shape()[1] ||
        !probabilities.is_contiguous() || !output_gradient.is_contiguous()) {
        throw std::invalid_argument(
            "Attention dV BTHD requires contiguous FP32 P[B,H,T,T] and dO[B,T,H,D]");
    }
    if (output_gradient.shape()[2] > std::numeric_limits<std::int32_t>::max()) {
        throw std::overflow_error("Attention head count exceeds hipBLASLt batch range");
    }
    if (probabilities.device().is_hip()) {
#if MICROLLM_HAS_HIPBLASLT
        return hipblaslt_attention_value_gradient_bthd(
            probabilities, output_gradient, context);
#else
        throw std::runtime_error("Attention dV BTHD on HIP requires hipBLASLt");
#endif
    }
    const auto gradient_bhtd = output_gradient.transpose(1, 2).contiguous();
    return matmul_with_implementation(
               probabilities, gradient_bhtd, MatmulImplementation::Readable,
               true, false, context)
        .transpose(1, 2).contiguous();
}

Tensor bf16_matmul(const Tensor& left_fp32, const Tensor& right_bf16,
                   const OpContext& context) {
    if (left_fp32.dtype() != DType::Float32 ||
        right_bf16.dtype() != DType::BFloat16 ||
        left_fp32.device() != right_bf16.device()) {
        throw std::invalid_argument(
            "bf16_matmul requires FP32 left, BF16 right, and matching devices");
    }
    if (left_fp32.device().is_cpu()) {
        return matmul(left_fp32.cast(DType::BFloat16).cast(DType::Float32),
                      right_bf16.cast(DType::Float32), context);
    }
    const auto left_bf16 = cast(left_fp32, DType::BFloat16, context);
#if MICROLLM_HAS_HIPBLASLT
    return hipblaslt_bf16_matmul(left_bf16, right_bf16, DType::Float32, context);
#else
    return matmul(cast(left_bf16, DType::Float32, context),
                  cast(right_bf16, DType::Float32, context), context);
#endif
}

Tensor bf16_matmul_output(const Tensor& left_bf16, const Tensor& right_bf16,
                          DType output_dtype, const OpContext& context) {
    if (left_bf16.dtype() != DType::BFloat16 ||
        right_bf16.dtype() != DType::BFloat16 ||
        left_bf16.device() != right_bf16.device() ||
        (output_dtype != DType::Float32 && output_dtype != DType::BFloat16)) {
        throw std::invalid_argument("bf16_matmul_output requires BF16 inputs and FP32/BF16 output");
    }
    if (left_bf16.device().is_cpu()) {
        auto output = matmul(left_bf16.cast(DType::Float32),
                             right_bf16.cast(DType::Float32), context);
        return output_dtype == DType::Float32 ? output : output.cast(DType::BFloat16);
    }
#if MICROLLM_HAS_HIPBLASLT
    return hipblaslt_bf16_matmul(left_bf16, right_bf16, output_dtype, context);
#else
    throw std::runtime_error("BF16 output matmul requires hipBLASLt");
#endif
}

void bf16_matmul_output_out_(
    Tensor& output, const Tensor& left_bf16, const Tensor& right_bf16,
    Tensor& output_fallback_bf16, const OpContext& context) {
    if (left_bf16.dtype() != DType::BFloat16 ||
        right_bf16.dtype() != DType::BFloat16 ||
        left_bf16.device() != right_bf16.device() ||
        output.device() != left_bf16.device() ||
        (output.dtype() != DType::Float32 &&
         output.dtype() != DType::BFloat16)) {
        throw std::invalid_argument(
            "bf16_matmul_output_out requires BF16 inputs and FP32/BF16 output");
    }
    if (left_bf16.ndim() != 2 || right_bf16.ndim() != 2 ||
        !left_bf16.is_contiguous() || !right_bf16.is_contiguous() ||
        left_bf16.shape()[1] != right_bf16.shape()[0] ||
        output.shape() != Shape({left_bf16.shape()[0],
                                 right_bf16.shape()[1]}) ||
        !output.is_contiguous()) {
        throw std::invalid_argument(
            "bf16_matmul_output_out shape/layout mismatch");
    }
    if (output.dtype() == DType::Float32 &&
        (output_fallback_bf16.dtype() != DType::BFloat16 ||
         output_fallback_bf16.device() != output.device() ||
         output_fallback_bf16.shape() != output.shape() ||
         !output_fallback_bf16.is_contiguous() ||
         output_fallback_bf16.storage().data() == output.storage().data())) {
        throw std::invalid_argument(
            "FP32 BF16 matmul out requires distinct shape-compatible BF16 fallback");
    }
    if (output.storage().data() == left_bf16.storage().data() ||
        output.storage().data() == right_bf16.storage().data() ||
        (output.dtype() == DType::Float32 &&
         (output_fallback_bf16.storage().data() ==
              left_bf16.storage().data() ||
          output_fallback_bf16.storage().data() ==
              right_bf16.storage().data()))) {
        throw std::invalid_argument(
            "BF16 matmul output/fallback must not alias input Storage");
    }
    if (left_bf16.device().is_cpu()) {
        const auto reference = bf16_matmul_output(
            left_bf16, right_bf16, output.dtype(), context);
        if (reference.shape() != output.shape() || !output.is_contiguous()) {
            throw std::invalid_argument(
                "bf16_matmul_output_out output shape/layout mismatch");
        }
        runtime::copy_bytes(
            output.data(), output.device(), reference.data(), reference.device(),
            static_cast<std::size_t>(output.numel()) * dtype_size(output.dtype()));
        return;
    }
#if MICROLLM_HAS_HIPBLASLT
    hipblaslt_bf16_matmul_out(
        output, left_bf16, right_bf16,
        output.dtype() == DType::Float32 ? &output_fallback_bf16 : nullptr,
        context);
#else
    throw std::runtime_error("BF16 output matmul requires hipBLASLt");
#endif
}

void bf16_ffn_out_(Tensor& output_fp32, Bf16FfnWorkspace& workspace,
                   const Tensor& input_fp32,
                   const Tensor& gate_weight_bf16,
                   const Tensor& up_weight_bf16,
                   const Tensor& down_weight_bf16,
                   const OpContext& context) {
    if (input_fp32.dtype() != DType::Float32 || input_fp32.ndim() != 2 ||
        !input_fp32.is_contiguous() || output_fp32.dtype() != DType::Float32 ||
        !output_fp32.is_contiguous()) {
        throw std::invalid_argument(
            "bf16_ffn_out requires contiguous 2D FP32 input/output");
    }
    const auto rows = input_fp32.shape()[0];
    const auto hidden = input_fp32.shape()[1];
    if (gate_weight_bf16.dtype() != DType::BFloat16 ||
        up_weight_bf16.dtype() != DType::BFloat16 ||
        down_weight_bf16.dtype() != DType::BFloat16 ||
        gate_weight_bf16.ndim() != 2 || up_weight_bf16.ndim() != 2 ||
        down_weight_bf16.ndim() != 2 ||
        !gate_weight_bf16.is_contiguous() ||
        !up_weight_bf16.is_contiguous() ||
        !down_weight_bf16.is_contiguous() ||
        gate_weight_bf16.shape()[0] != hidden ||
        up_weight_bf16.shape() != gate_weight_bf16.shape() ||
        down_weight_bf16.shape()[0] != gate_weight_bf16.shape()[1] ||
        output_fp32.shape() != Shape({rows, down_weight_bf16.shape()[1]})) {
        throw std::invalid_argument("bf16_ffn_out weight/output shapes are incompatible");
    }
    const auto intermediate = gate_weight_bf16.shape()[1];
    const auto device = input_fp32.device();
    const Shape input_shape{rows, hidden};
    const Shape intermediate_shape{rows, intermediate};
    const Shape output_shape{rows, down_weight_bf16.shape()[1]};
    const auto valid_workspace = [&](const Tensor& tensor,
                                     const Shape& shape) {
        return tensor.dtype() == DType::BFloat16 &&
               tensor.device() == device && tensor.shape() == shape &&
               tensor.is_contiguous();
    };
    if (gate_weight_bf16.device() != device ||
        up_weight_bf16.device() != device ||
        down_weight_bf16.device() != device || output_fp32.device() != device ||
        !valid_workspace(workspace.input_bf16, input_shape) ||
        !valid_workspace(workspace.gate, intermediate_shape) ||
        !valid_workspace(workspace.up, intermediate_shape) ||
        !valid_workspace(workspace.activated, intermediate_shape) ||
        !valid_workspace(workspace.output_fallback_bf16, output_shape)) {
        throw std::invalid_argument(
            "bf16_ffn_out workspace must contain matching contiguous BF16 tensors");
    }
    const std::vector<const void*> writable{
        output_fp32.data(), workspace.input_bf16.data(), workspace.gate.data(),
        workspace.up.data(), workspace.activated.data(),
        workspace.output_fallback_bf16.data()};
    if (std::set<const void*>(writable.begin(), writable.end()).size() !=
        writable.size()) {
        throw std::invalid_argument("bf16_ffn_out workspace tensors must not alias");
    }
    const std::set<const void*> writable_set(writable.begin(), writable.end());
    for (const auto* readable : {input_fp32.data(), gate_weight_bf16.data(),
                                 up_weight_bf16.data(),
                                 down_weight_bf16.data()}) {
        if (writable_set.contains(readable)) {
            throw std::invalid_argument(
                "bf16_ffn_out workspace/output must not alias an input");
        }
    }
    cast_out_(input_fp32, workspace.input_bf16, context);
    bf16_ffn_precast_out_(
        output_fp32, workspace, gate_weight_bf16, up_weight_bf16,
        down_weight_bf16, context);
}

void bf16_ffn_precast_out_(
    Tensor& output_fp32, Bf16FfnWorkspace& workspace,
    const Tensor& gate_weight_bf16, const Tensor& up_weight_bf16,
    const Tensor& down_weight_bf16, const OpContext& context) {
    const auto& input_bf16 = workspace.input_bf16;
    if (input_bf16.dtype() != DType::BFloat16 || input_bf16.ndim() != 2 ||
        !input_bf16.is_contiguous() || output_fp32.dtype() != DType::Float32 ||
        !output_fp32.is_contiguous()) {
        throw std::invalid_argument(
            "bf16_ffn_precast_out requires contiguous BF16 input and FP32 output");
    }
    const auto rows = input_bf16.shape()[0];
    const auto hidden = input_bf16.shape()[1];
    const auto device = input_bf16.device();
    if (gate_weight_bf16.dtype() != DType::BFloat16 ||
        up_weight_bf16.dtype() != DType::BFloat16 ||
        down_weight_bf16.dtype() != DType::BFloat16 ||
        gate_weight_bf16.ndim() != 2 || up_weight_bf16.ndim() != 2 ||
        down_weight_bf16.ndim() != 2 || !gate_weight_bf16.is_contiguous() ||
        !up_weight_bf16.is_contiguous() || !down_weight_bf16.is_contiguous() ||
        gate_weight_bf16.shape()[0] != hidden ||
        up_weight_bf16.shape() != gate_weight_bf16.shape() ||
        down_weight_bf16.shape()[0] != gate_weight_bf16.shape()[1] ||
        output_fp32.shape() != Shape({rows, down_weight_bf16.shape()[1]})) {
        throw std::invalid_argument(
            "bf16_ffn_precast_out weight/output shapes are incompatible");
    }
    const Shape intermediate_shape{rows, gate_weight_bf16.shape()[1]};
    const Shape output_shape{rows, down_weight_bf16.shape()[1]};
    const auto valid_workspace = [&](const Tensor& tensor, const Shape& shape) {
        return tensor.dtype() == DType::BFloat16 && tensor.device() == device &&
               tensor.shape() == shape && tensor.is_contiguous();
    };
    if (gate_weight_bf16.device() != device || up_weight_bf16.device() != device ||
        down_weight_bf16.device() != device || output_fp32.device() != device ||
        !valid_workspace(workspace.gate, intermediate_shape) ||
        !valid_workspace(workspace.up, intermediate_shape) ||
        !valid_workspace(workspace.activated, intermediate_shape) ||
        !valid_workspace(workspace.output_fallback_bf16, output_shape)) {
        throw std::invalid_argument(
            "bf16_ffn_precast_out workspace mismatch");
    }
    const std::vector<const void*> buffers{
        output_fp32.data(), input_bf16.data(), workspace.gate.data(),
        workspace.up.data(), workspace.activated.data(),
        workspace.output_fallback_bf16.data()};
    if (std::set<const void*>(buffers.begin(), buffers.end()).size() !=
        buffers.size()) {
        throw std::invalid_argument(
            "bf16_ffn_precast_out buffers must not alias");
    }
    const std::set<const void*> buffer_set(buffers.begin(), buffers.end());
    for (const auto* weight : {gate_weight_bf16.data(), up_weight_bf16.data(),
                               down_weight_bf16.data()}) {
        if (buffer_set.contains(weight)) {
            throw std::invalid_argument(
                "bf16_ffn_precast_out buffers must not alias weights");
        }
    }
    bool grouped_gate_up = false;
#if MICROLLM_HAS_HIPBLASLT
    grouped_gate_up = try_bf16_grouped_gate_up(
        workspace, gate_weight_bf16, up_weight_bf16, context);
#endif
    if (!grouped_gate_up) {
        bf16_matmul_output_out_(
            workspace.gate, workspace.input_bf16,
            gate_weight_bf16,
            workspace.output_fallback_bf16, context);
        bf16_matmul_output_out_(
            workspace.up, workspace.input_bf16,
            up_weight_bf16,
            workspace.output_fallback_bf16, context);
    }
    if (grouped_gate_up && bf16_grouped_gate_up_swish) {
        multiply_out_(workspace.activated, workspace.gate, workspace.up, context);
    } else {
        swiglu_out_(workspace.activated, workspace.gate, workspace.up, context);
    }
    bf16_matmul_output_out_(output_fp32, workspace.activated,
                            down_weight_bf16,
                            workspace.output_fallback_bf16, context);
}

static Tensor bf16_ffn_impl(const Tensor& input_fp32,
                            const Tensor& gate_weight_bf16,
                            const Tensor& up_weight_bf16,
                            const Tensor& down_weight_bf16,
                            Bf16FfnDiagnostics* diagnostics,
                            const OpContext& context) {
    if (input_fp32.dtype() != DType::Float32 ||
        gate_weight_bf16.dtype() != DType::BFloat16 ||
        up_weight_bf16.dtype() != DType::BFloat16 ||
        down_weight_bf16.dtype() != DType::BFloat16) {
        throw std::invalid_argument(
            "bf16_ffn requires FP32 input and BF16 gate/up/down weights");
    }
    if (input_fp32.device() != gate_weight_bf16.device() ||
        input_fp32.device() != up_weight_bf16.device() ||
        input_fp32.device() != down_weight_bf16.device()) {
        throw std::invalid_argument("bf16_ffn tensors must use one device");
    }
    if (input_fp32.ndim() != 2 || gate_weight_bf16.ndim() != 2 ||
        up_weight_bf16.ndim() != 2 || down_weight_bf16.ndim() != 2 ||
        !input_fp32.is_contiguous() || !gate_weight_bf16.is_contiguous() ||
        !up_weight_bf16.is_contiguous() || !down_weight_bf16.is_contiguous()) {
        throw std::invalid_argument("bf16_ffn requires contiguous 2D tensors");
    }
    const auto hidden = input_fp32.shape()[1];
    const auto intermediate = gate_weight_bf16.shape()[1];
    if (gate_weight_bf16.shape()[0] != hidden ||
        up_weight_bf16.shape() != gate_weight_bf16.shape() ||
        down_weight_bf16.shape()[0] != intermediate) {
        throw std::invalid_argument("bf16_ffn weight shapes are incompatible");
    }

    const auto input_bf16 = cast(input_fp32, DType::BFloat16, context);
    const auto gate = bf16_matmul_output(input_bf16, gate_weight_bf16,
                                         DType::BFloat16, context);
    const auto up = bf16_matmul_output(input_bf16, up_weight_bf16,
                                       DType::BFloat16, context);
    const auto activated = swiglu(gate, up, context);
    auto output = bf16_matmul_output(activated, down_weight_bf16,
                                     DType::Float32, context);
    if (diagnostics != nullptr) {
        *diagnostics = {.input_bf16 = input_bf16,
                        .gate = gate,
                        .up = up,
                        .activated = activated,
                        .output = output};
    }
    return output;
}

Tensor bf16_ffn(const Tensor& input_fp32,
                const Tensor& gate_weight_bf16,
                const Tensor& up_weight_bf16,
                const Tensor& down_weight_bf16,
                const OpContext& context) {
    return bf16_ffn_impl(input_fp32, gate_weight_bf16, up_weight_bf16,
                         down_weight_bf16, nullptr, context);
}

Bf16FfnDiagnostics bf16_ffn_diagnostics(
    const Tensor& input_fp32, const Tensor& gate_weight_bf16,
    const Tensor& up_weight_bf16, const Tensor& down_weight_bf16,
    const OpContext& context) {
    Bf16FfnDiagnostics diagnostics;
    (void)bf16_ffn_impl(input_fp32, gate_weight_bf16, up_weight_bf16,
                        down_weight_bf16, &diagnostics, context);
    return diagnostics;
}

TensorTriple bf16_qkv_projection(const Tensor& input_fp32,
                                 const Tensor& query_weight_bf16,
                                 const Tensor& key_weight_bf16,
                                 const Tensor& value_weight_bf16,
                                 const OpContext& context) {
    if (input_fp32.dtype() != DType::Float32 || input_fp32.ndim() != 2 ||
        !input_fp32.is_contiguous()) {
        throw std::invalid_argument("bf16_qkv_projection requires contiguous 2D FP32 input");
    }
    for (const auto* weight : {&query_weight_bf16, &key_weight_bf16,
                               &value_weight_bf16}) {
        if (weight->dtype() != DType::BFloat16 || weight->ndim() != 2 ||
            !weight->is_contiguous() || weight->device() != input_fp32.device() ||
            weight->shape()[0] != input_fp32.shape()[1]) {
            throw std::invalid_argument(
                "bf16_qkv_projection weights must be compatible contiguous BF16 matrices");
        }
    }
    const auto input_bf16 = cast(input_fp32, DType::BFloat16, context);
    return {bf16_matmul_output(input_bf16, query_weight_bf16, DType::Float32, context),
            bf16_matmul_output(input_bf16, key_weight_bf16, DType::Float32, context),
            bf16_matmul_output(input_bf16, value_weight_bf16, DType::Float32, context)};
}

TensorPair bf16_gate_up_projection(
    const Tensor& input_fp32, const Tensor& gate_weight_bf16,
    const Tensor& up_weight_bf16, const OpContext& context) {
    if (input_fp32.dtype() != DType::Float32 || input_fp32.ndim() != 2 ||
        !input_fp32.is_contiguous()) {
        throw std::invalid_argument(
            "bf16_gate_up_projection requires contiguous 2D FP32 input");
    }
    if (gate_weight_bf16.dtype() != DType::BFloat16 ||
        up_weight_bf16.dtype() != DType::BFloat16 ||
        gate_weight_bf16.ndim() != 2 || up_weight_bf16.ndim() != 2 ||
        !gate_weight_bf16.is_contiguous() ||
        !up_weight_bf16.is_contiguous() ||
        gate_weight_bf16.device() != input_fp32.device() ||
        up_weight_bf16.device() != input_fp32.device() ||
        gate_weight_bf16.shape() != up_weight_bf16.shape() ||
        gate_weight_bf16.shape()[0] != input_fp32.shape()[1]) {
        throw std::invalid_argument(
            "bf16_gate_up_projection weights must be equal compatible BF16 matrices");
    }
    const auto input_bf16 = cast(input_fp32, DType::BFloat16, context);
    return {
        bf16_matmul_output(input_bf16, gate_weight_bf16,
                           DType::Float32, context),
        bf16_matmul_output(input_bf16, up_weight_bf16,
                           DType::Float32, context),
    };
}

bool bf16_qkv_projection_out_(
    Tensor& query_output_fp32, Tensor& key_output_fp32,
    Tensor& value_output_fp32, Bf16QkvWorkspace& workspace,
    const Tensor& input_fp32, const Tensor& query_weight_bf16,
    const Tensor& key_weight_bf16, const Tensor& value_weight_bf16,
    const OpContext& context, [[maybe_unused]] bool retain_query_key_bf16,
    [[maybe_unused]] bool retain_value_bf16) {
    if (retain_value_bf16 && !retain_query_key_bf16) {
        throw std::invalid_argument(
            "retaining BF16 value requires retaining BF16 query/key");
    }
    if (input_fp32.dtype() != DType::Float32 || input_fp32.ndim() != 2 ||
        !input_fp32.is_contiguous()) {
        throw std::invalid_argument(
            "bf16_qkv_projection_out requires contiguous 2D FP32 input");
    }
    const auto rows = input_fp32.shape()[0];
    const auto hidden = input_fp32.shape()[1];
    const auto device = input_fp32.device();
    const auto valid_weight = [&](const Tensor& weight) {
        return weight.dtype() == DType::BFloat16 && weight.ndim() == 2 &&
               weight.is_contiguous() && weight.device() == device &&
               weight.shape()[0] == hidden;
    };
    if (!valid_weight(query_weight_bf16) ||
        !valid_weight(key_weight_bf16) ||
        !valid_weight(value_weight_bf16)) {
        throw std::invalid_argument(
            "bf16_qkv_projection_out weights must be compatible BF16 matrices");
    }
    const auto valid_output = [&](const Tensor& output,
                                  std::int64_t columns) {
        return output.dtype() == DType::Float32 &&
               output.device() == device &&
               output.shape() == Shape({rows, columns}) &&
               output.is_contiguous();
    };
    const auto valid_fallback = [&](const Tensor& fallback,
                                    const Tensor& output) {
        return fallback.dtype() == DType::BFloat16 &&
               fallback.device() == device &&
               fallback.shape() == output.shape() &&
               fallback.is_contiguous();
    };
    if (!valid_output(query_output_fp32, query_weight_bf16.shape()[1]) ||
        !valid_output(key_output_fp32, key_weight_bf16.shape()[1]) ||
        !valid_output(value_output_fp32, value_weight_bf16.shape()[1]) ||
        workspace.input_bf16.dtype() != DType::BFloat16 ||
        workspace.input_bf16.device() != device ||
        workspace.input_bf16.shape() != input_fp32.shape() ||
        !workspace.input_bf16.is_contiguous() ||
        !valid_fallback(workspace.query_fallback_bf16,
                        query_output_fp32) ||
        !valid_fallback(workspace.key_fallback_bf16, key_output_fp32) ||
        !valid_fallback(workspace.value_fallback_bf16,
                        value_output_fp32)) {
        throw std::invalid_argument(
            "bf16_qkv_projection_out workspace/output mismatch");
    }
    const std::vector<const void*> writable{
        query_output_fp32.data(), key_output_fp32.data(),
        value_output_fp32.data(), workspace.input_bf16.data(),
        workspace.query_fallback_bf16.data(),
        workspace.key_fallback_bf16.data(),
        workspace.value_fallback_bf16.data()};
    const std::set<const void*> writable_set(writable.begin(), writable.end());
    if (writable_set.size() != writable.size()) {
        throw std::invalid_argument(
            "bf16_qkv_projection_out writable tensors must not alias");
    }
    for (const auto* readable : {input_fp32.data(), query_weight_bf16.data(),
                                 key_weight_bf16.data(),
                                 value_weight_bf16.data()}) {
        if (writable_set.contains(readable)) {
            throw std::invalid_argument(
                "bf16_qkv_projection_out workspace/output must not alias input");
        }
    }
    cast_out_(input_fp32, workspace.input_bf16, context);
    return bf16_qkv_projection_precast_out_(
        query_output_fp32, key_output_fp32, value_output_fp32, workspace,
        query_weight_bf16, key_weight_bf16, value_weight_bf16, context,
        retain_query_key_bf16, retain_value_bf16);
}

bool bf16_qkv_projection_precast_out_(
    Tensor& query_output_fp32, Tensor& key_output_fp32,
    Tensor& value_output_fp32, Bf16QkvWorkspace& workspace,
    const Tensor& query_weight_bf16, const Tensor& key_weight_bf16,
    const Tensor& value_weight_bf16, const OpContext& context,
    bool retain_query_key_bf16, bool retain_value_bf16) {
    if (retain_value_bf16 && !retain_query_key_bf16) {
        throw std::invalid_argument(
            "retaining BF16 value requires retaining BF16 query/key");
    }
    const auto& input_bf16 = workspace.input_bf16;
    if (input_bf16.dtype() != DType::BFloat16 || input_bf16.ndim() != 2 ||
        !input_bf16.is_contiguous()) {
        throw std::invalid_argument(
            "bf16_qkv_projection_precast_out requires contiguous BF16 input");
    }
    const auto rows = input_bf16.shape()[0];
    const auto hidden = input_bf16.shape()[1];
    const auto device = input_bf16.device();
    const auto valid_weight = [&](const Tensor& weight) {
        return weight.dtype() == DType::BFloat16 && weight.ndim() == 2 &&
               weight.is_contiguous() && weight.device() == device &&
               weight.shape()[0] == hidden;
    };
    const auto valid_output = [&](const Tensor& output, std::int64_t columns) {
        return output.dtype() == DType::Float32 && output.device() == device &&
               output.shape() == Shape({rows, columns}) && output.is_contiguous();
    };
    const auto valid_fallback = [&](const Tensor& fallback, const Tensor& output) {
        return fallback.dtype() == DType::BFloat16 &&
               fallback.device() == device && fallback.shape() == output.shape() &&
               fallback.is_contiguous();
    };
    if (!valid_weight(query_weight_bf16) || !valid_weight(key_weight_bf16) ||
        !valid_weight(value_weight_bf16) ||
        !valid_output(query_output_fp32, query_weight_bf16.shape()[1]) ||
        !valid_output(key_output_fp32, key_weight_bf16.shape()[1]) ||
        !valid_output(value_output_fp32, value_weight_bf16.shape()[1]) ||
        !valid_fallback(workspace.query_fallback_bf16, query_output_fp32) ||
        !valid_fallback(workspace.key_fallback_bf16, key_output_fp32) ||
        !valid_fallback(workspace.value_fallback_bf16, value_output_fp32)) {
        throw std::invalid_argument(
            "bf16_qkv_projection_precast_out workspace/output mismatch");
    }
    const std::vector<const void*> buffers{
        query_output_fp32.data(), key_output_fp32.data(), value_output_fp32.data(),
        input_bf16.data(), workspace.query_fallback_bf16.data(),
        workspace.key_fallback_bf16.data(), workspace.value_fallback_bf16.data()};
    const std::set<const void*> buffer_set(buffers.begin(), buffers.end());
    if (buffer_set.size() != buffers.size()) {
        throw std::invalid_argument(
            "bf16_qkv_projection_precast_out buffers must not alias");
    }
    for (const auto* weight : {query_weight_bf16.data(), key_weight_bf16.data(),
                               value_weight_bf16.data()}) {
        if (buffer_set.contains(weight)) {
            throw std::invalid_argument(
                "bf16_qkv_projection_precast_out buffers must not alias weights");
        }
    }
#if MICROLLM_HAS_HIPBLASLT
    if (try_bf16_grouped_qkv(
            workspace, query_weight_bf16, key_weight_bf16,
            value_weight_bf16, context)) {
        if (!retain_query_key_bf16) {
            cast_out_(workspace.query_fallback_bf16,
                      query_output_fp32, context);
            cast_out_(workspace.key_fallback_bf16,
                      key_output_fp32, context);
        }
        if (!retain_value_bf16) {
            cast_out_(workspace.value_fallback_bf16,
                      value_output_fp32, context);
        }
        if (retain_query_key_bf16) {
            ++bf16_grouped_qkv_retained_query_key_dispatches;
        }
        return retain_query_key_bf16;
    }
#endif
    bf16_matmul_output_out_(
        query_output_fp32, workspace.input_bf16, query_weight_bf16,
        workspace.query_fallback_bf16, context);
    bf16_matmul_output_out_(
        key_output_fp32, workspace.input_bf16, key_weight_bf16,
        workspace.key_fallback_bf16, context);
    bf16_matmul_output_out_(
        value_output_fp32, workspace.input_bf16, value_weight_bf16,
        workspace.value_fallback_bf16, context);
    return false;
}

Tensor fp8_matmul(const ScaledTensor& left, const ScaledTensor& right,
                  DType output_dtype, const OpContext& context) {
    if (left.values.device() != right.values.device()) {
        throw std::invalid_argument("FP8 matmul devices must match");
    }
    if (left.values.device().is_cpu()) {
        return matmul(dequantize_fp8(left, DType::Float32),
                      dequantize_fp8(right, DType::Float32)).cast(output_dtype);
    }
#if MICROLLM_HAS_HIPBLASLT
    return hipblaslt_fp8_matmul(left, right, output_dtype, context);
#else
    (void)output_dtype;
    (void)context;
    throw std::runtime_error("FP8 matmul requires hipBLASLt");
#endif
}

}  // namespace microllm::ops
