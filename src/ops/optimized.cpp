#include <microllm/ops/ops.h>

#include <stdexcept>
#include <string>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <tuple>

#if MICROLLM_HAS_HIP
#include "hip/kernels.h"
#endif

#if MICROLLM_HAS_HIPBLASLT
#include <hipblaslt/hipblaslt.h>
#endif

namespace microllm::ops {

namespace {
using MatmulShapeKey = std::tuple<std::int64_t, std::int64_t, std::int64_t>;
std::mutex registry_mutex;
std::map<MatmulShapeKey, MatmulImplementation> registry;
#if MICROLLM_HAS_HIPBLASLT
// A few gfx942 decode shapes cannot write BF16-input GEMM results directly to
// FP32 even though the same problem can write BF16. Remember the observed
// capability per shape so repeated transformer layers do not retry a rejected
// library path on every token.
thread_local std::map<MatmulShapeKey, bool> bf16_fp32_direct_registry;
#endif
}  // namespace

bool hipblaslt_available() noexcept { return MICROLLM_HAS_HIPBLASLT != 0; }

MatmulImplementation choose_matmul_implementation(const Tensor& left,
                                                  const Tensor& right) {
    return choose_matmul_implementation(left, right, false, false);
}

MatmulImplementation choose_matmul_implementation(const Tensor& left,
                                                  const Tensor& right,
                                                  bool transpose_left,
                                                  bool transpose_right) {
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
    const MatmulShapeKey key{rows, inner, columns};
    {
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

void register_matmul_implementation(std::int64_t rows, std::int64_t inner,
                                    std::int64_t columns,
                                    MatmulImplementation implementation) {
    if (rows <= 0 || inner <= 0 || columns <= 0) {
        throw std::invalid_argument("registered matmul dimensions must be positive");
    }
    if (implementation == MatmulImplementation::Auto) {
        throw std::invalid_argument("matmul registry choice must name a concrete implementation");
    }
    if (implementation == MatmulImplementation::HipBLASLt && !hipblaslt_available()) {
        throw std::invalid_argument("cannot register unavailable hipBLASLt implementation");
    }
    const std::lock_guard<std::mutex> lock(registry_mutex);
    registry[{rows, inner, columns}] = implementation;
}

void clear_matmul_implementation_registry() {
    const std::lock_guard<std::mutex> lock(registry_mutex);
    registry.clear();
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

class Bf16Plan {
public:
    Bf16Plan(std::int64_t rows, std::int64_t inner, std::int64_t columns,
             DType output_dtype)
        : matrix_b_(HIP_R_16BF, static_cast<std::uint64_t>(columns),
                    static_cast<std::uint64_t>(inner), columns),
          matrix_a_(HIP_R_16BF, static_cast<std::uint64_t>(inner),
                    static_cast<std::uint64_t>(rows), inner),
          matrix_c_(hip_dtype(output_dtype), static_cast<std::uint64_t>(columns),
                    static_cast<std::uint64_t>(rows), columns) {}

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

private:
    MatmulDescription operation_;
    Layout matrix_b_;
    Layout matrix_a_;
    Layout matrix_c_;
};

thread_local std::map<Bf16PlanKey, std::unique_ptr<Bf16Plan>> bf16_plans;
thread_local std::size_t bf16_plan_hits = 0;
thread_local std::size_t bf16_plan_misses = 0;

Bf16Plan& bf16_plan(std::int64_t rows, std::int64_t inner,
                    std::int64_t columns, DType output_dtype) {
    const Bf16PlanKey key{rows, inner, columns, output_dtype};
    const auto found = bf16_plans.find(key);
    if (found != bf16_plans.end()) {
        ++bf16_plan_hits;
        return *found->second;
    }
    ++bf16_plan_misses;
    auto plan = std::make_unique<Bf16Plan>(rows, inner, columns, output_dtype);
    auto* result = plan.get();
    bf16_plans.emplace(key, std::move(plan));
    return *result;
}

Tensor hipblaslt_matmul(const Tensor& left, const Tensor& right,
                        bool transpose_left, bool transpose_right,
                        const OpContext& context) {
    if (!left.device().is_hip() || right.device() != left.device() ||
        !is_floating_point(left.dtype()) || right.dtype() != left.dtype() ||
        left.ndim() < 2 || right.ndim() != left.ndim() || !left.is_contiguous() ||
        !right.is_contiguous()) {
        throw std::invalid_argument(
            "hipBLASLt matmul requires matching contiguous floating tensors on one HIP device");
    }
    const auto rank = static_cast<std::size_t>(left.ndim());
    std::int64_t batches = 1;
    Shape output_shape(left.shape().begin(), left.shape().end() - 2);
    for (std::size_t dimension = 0; dimension + 2 < rank; ++dimension) {
        if (left.shape()[dimension] != right.shape()[dimension]) {
            throw std::invalid_argument("hipBLASLt matmul batch dimensions must match");
        }
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
    const auto inner = transpose_left ? left_rows : left_columns;
    const auto right_inner = transpose_right ? right_columns : right_rows;
    const auto columns = transpose_right ? right_rows : right_columns;
    if (right_inner != inner) throw std::invalid_argument("matmul inner dimensions mismatch");
    output_shape.push_back(rows);
    output_shape.push_back(columns);
    Tensor output(output_shape, left.dtype(), left.device());
    static Handle handle;
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
    const float alpha = 1.0F;
    const float beta = 0.0F;
    check_status(hipblasLtMatmul(
                     handle.get(), operation.get(), &alpha, right.data(), matrix_b.get(),
                     left.data(), matrix_a.get(), &beta, output.data(), matrix_c.get(),
                     output.data(), matrix_c.get(), nullptr, context.workspace,
                     context.workspace_bytes,
                     reinterpret_cast<hipStream_t>(context.native_stream(left.device()))),
                 "hipblasLtMatmul");
    return output;
}

Tensor hipblaslt_bf16_matmul(const Tensor& left, const Tensor& right,
                             DType output_dtype, const OpContext& context) {
    if (!left.device().is_hip() || right.device() != left.device() ||
        left.dtype() != DType::BFloat16 || right.dtype() != DType::BFloat16 ||
        left.ndim() != 2 || right.ndim() != 2 || !left.is_contiguous() ||
        !right.is_contiguous() || left.shape()[1] != right.shape()[0]) {
        throw std::invalid_argument(
            "BF16 mixed matmul requires matching contiguous 2D HIP tensors");
    }
    const auto rows = left.shape()[0];
    const auto inner = left.shape()[1];
    const auto columns = right.shape()[1];
    if (output_dtype != DType::Float32 && output_dtype != DType::BFloat16) {
        throw std::invalid_argument("BF16 matmul output must be FP32 or BF16");
    }
    const MatmulShapeKey shape{rows, inner, columns};
    if (output_dtype == DType::Float32) {
        const auto found = bf16_fp32_direct_registry.find(shape);
        if (found != bf16_fp32_direct_registry.end() && !found->second) {
            return cast(hipblaslt_bf16_matmul(
                            left, right, DType::BFloat16, context),
                        DType::Float32, context);
        }
    }
    Tensor output({rows, columns}, output_dtype, left.device());
    static Handle handle;
    auto& plan = bf16_plan(rows, inner, columns, output_dtype);
    const float alpha = 1.0F;
    const float beta = 0.0F;
    const auto status = hipblasLtMatmul(
        handle.get(), plan.operation(), &alpha,
        right.data(), plan.matrix_b(), left.data(), plan.matrix_a(),
        &beta, output.data(), plan.matrix_c(), output.data(), plan.matrix_c(),
        nullptr, context.workspace, context.workspace_bytes,
        reinterpret_cast<hipStream_t>(context.native_stream(left.device())));
    if (status == HIPBLAS_STATUS_SUCCESS) {
        if (output_dtype == DType::Float32) {
            bf16_fp32_direct_registry[shape] = true;
        }
        return output;
    }
    if (output_dtype == DType::Float32 &&
        (status == HIPBLAS_STATUS_INTERNAL_ERROR ||
         status == HIPBLAS_STATUS_NOT_SUPPORTED)) {
        bf16_fp32_direct_registry[shape] = false;
        return cast(hipblaslt_bf16_matmul(left, right, DType::BFloat16, context),
                    DType::Float32, context);
    }
    check_status(status, "hipblasLtMatmul(BF16)");
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
    Tensor output({rows, columns}, output_dtype, left.values.device());
    static Handle handle;
    MatmulDescription operation;
    // Row-major C=A*B is submitted as column-major C^T=B^T*A^T, so scale A belongs
    // to the user-visible right operand and scale B to the left operand.
    set_scale_pointer(operation.get(), HIPBLASLT_MATMUL_DESC_A_SCALE_POINTER,
                      right.scale.data());
    set_scale_pointer(operation.get(), HIPBLASLT_MATMUL_DESC_B_SCALE_POINTER,
                      left.scale.data());
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
    check_status(hipblasLtMatmul(
                     handle.get(), operation.get(), &alpha, right.values.data(), matrix_b.get(),
                     left.values.data(), matrix_a.get(), &beta, output.data(), matrix_c.get(),
                     output.data(), matrix_c.get(), nullptr, context.workspace,
                     context.workspace_bytes,
                     reinterpret_cast<hipStream_t>(context.native_stream(left.values.device()))),
                 "hipblasLtMatmul(FP8)");
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

Tensor matmul_with_implementation(const Tensor& left, const Tensor& right,
                                  MatmulImplementation implementation,
                                  const OpContext& context) {
    if (implementation == MatmulImplementation::Auto) {
        implementation = choose_matmul_implementation(left, right);
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
            left, right, transpose_left, transpose_right);
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

Tensor bf16_ffn(const Tensor& input_fp32,
                const Tensor& gate_weight_bf16,
                const Tensor& up_weight_bf16,
                const Tensor& down_weight_bf16,
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
    return bf16_matmul_output(activated, down_weight_bf16,
                              DType::Float32, context);
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
