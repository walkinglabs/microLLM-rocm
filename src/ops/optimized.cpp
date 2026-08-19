#include <microllm/ops/ops.h>

#include <stdexcept>
#include <string>
#include <map>
#include <mutex>
#include <tuple>

#if MICROLLM_HAS_HIPBLASLT
#include <hipblaslt/hipblaslt.h>
#endif

namespace microllm::ops {

namespace {
using MatmulShapeKey = std::tuple<std::int64_t, std::int64_t, std::int64_t>;
std::mutex registry_mutex;
std::map<MatmulShapeKey, MatmulImplementation> registry;
}  // namespace

bool hipblaslt_available() noexcept { return MICROLLM_HAS_HIPBLASLT != 0; }

MatmulImplementation choose_matmul_implementation(const Tensor& left,
                                                  const Tensor& right) {
    if (!hipblaslt_available() || !left.device().is_hip() || right.device() != left.device() ||
        left.ndim() != 2 || right.ndim() != 2 || !is_floating_point(left.dtype()) ||
        right.dtype() != left.dtype() || !left.is_contiguous() || !right.is_contiguous() ||
        left.shape()[1] != right.shape()[0]) {
        return MatmulImplementation::Readable;
    }
    const MatmulShapeKey key{left.shape()[0], left.shape()[1], right.shape()[1]};
    {
        const std::lock_guard<std::mutex> lock(registry_mutex);
        const auto found = registry.find(key);
        if (found != registry.end()) return found->second;
    }
    return left.shape()[1] >= 128 && right.shape()[1] >= 128
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
    hipblasLtMatrixLayout_t get() const noexcept { return value_; }

private:
    hipblasLtMatrixLayout_t value_ = nullptr;
};

hipDataType hip_dtype(DType dtype) {
    switch (dtype) {
        case DType::Float32: return HIP_R_32F;
        case DType::Float16: return HIP_R_16F;
        case DType::BFloat16: return HIP_R_16BF;
        case DType::Int32:
        case DType::Int64: break;
    }
    throw std::invalid_argument("hipBLASLt matmul requires FP32, FP16, or BF16");
}

class MatmulDescription {
public:
    MatmulDescription() {
        check_status(hipblasLtMatmulDescCreate(&value_, HIPBLAS_COMPUTE_32F,
                                               HIP_R_32F),
                     "hipblasLtMatmulDescCreate");
    }
    ~MatmulDescription() { (void)hipblasLtMatmulDescDestroy(value_); }
    MatmulDescription(const MatmulDescription&) = delete;
    MatmulDescription& operator=(const MatmulDescription&) = delete;
    hipblasLtMatmulDesc_t get() const noexcept { return value_; }

private:
    hipblasLtMatmulDesc_t value_ = nullptr;
};

Tensor hipblaslt_matmul(const Tensor& left, const Tensor& right,
                        const OpContext& context) {
    if (!left.device().is_hip() || right.device() != left.device() ||
        !is_floating_point(left.dtype()) || right.dtype() != left.dtype() ||
        left.ndim() != 2 || right.ndim() != 2 || !left.is_contiguous() ||
        !right.is_contiguous()) {
        throw std::invalid_argument(
            "hipBLASLt matmul requires matching contiguous 2D floating tensors on one HIP device");
    }
    const auto rows = left.shape()[0];
    const auto inner = left.shape()[1];
    const auto columns = right.shape()[1];
    if (right.shape()[0] != inner) throw std::invalid_argument("matmul inner dimensions mismatch");
    Tensor output({rows, columns}, left.dtype(), left.device());
    static Handle handle;
    MatmulDescription operation;
    const auto data_type = hip_dtype(left.dtype());
    // Row-major C=A*B is column-major C^T=B^T*A^T without moving data.
    Layout matrix_b(data_type, static_cast<std::uint64_t>(columns), static_cast<std::uint64_t>(inner),
                    columns);
    Layout matrix_a(data_type, static_cast<std::uint64_t>(inner), static_cast<std::uint64_t>(rows), inner);
    Layout matrix_c(data_type, static_cast<std::uint64_t>(columns), static_cast<std::uint64_t>(rows),
                    columns);
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

}  // namespace
#endif

Tensor matmul_with_implementation(const Tensor& left, const Tensor& right,
                                  MatmulImplementation implementation,
                                  const OpContext& context) {
    if (implementation == MatmulImplementation::Auto) {
        implementation = choose_matmul_implementation(left, right);
    }
    if (implementation == MatmulImplementation::Readable) return matmul(left, right, context);
#if MICROLLM_HAS_HIPBLASLT
    return hipblaslt_matmul(left, right, context);
#else
    (void)left;
    (void)right;
    (void)context;
    throw std::runtime_error("microLLM was built without hipBLASLt");
#endif
}

}  // namespace microllm::ops
