#include <microllm/core/tensor.h>
#include <microllm/runtime/memory.h>

#include <algorithm>
#include <cstring>
#include <limits>
#include <ostream>
#include <sstream>
#include <stdexcept>
#include <utility>

namespace microllm {
namespace {

std::int64_t checked_multiply(std::int64_t left, std::int64_t right,
                              const char* message) {
    if (left < 0 || right < 0) throw std::invalid_argument(message);
    if (left != 0 && right > std::numeric_limits<std::int64_t>::max() / left) {
        throw std::overflow_error(message);
    }
    return left * right;
}

std::size_t byte_count(std::int64_t elements, DType dtype) {
    const auto element_bytes = dtype_size(dtype);
    const auto unsigned_elements = static_cast<std::uint64_t>(elements);
    if (unsigned_elements > std::numeric_limits<std::size_t>::max() / element_bytes) {
        throw std::overflow_error("tensor byte size overflows size_t");
    }
    return static_cast<std::size_t>(unsigned_elements) * element_bytes;
}

std::int64_t logical_to_storage_index(std::int64_t logical, const Shape& shape,
                                      const Strides& strides, std::int64_t offset) {
    auto index = offset;
    for (std::size_t reversed = shape.size(); reversed > 0; --reversed) {
        const auto dim = reversed - 1;
        const auto coordinate = logical % shape[dim];
        logical /= shape[dim];
        index += coordinate * strides[dim];
    }
    return index;
}

void require_cpu_float32(const Tensor& tensor, const char* operation) {
    if (!tensor.device().is_cpu()) {
        throw std::runtime_error(std::string(operation) + " currently requires a CPU tensor");
    }
    if (tensor.dtype() != DType::Float32) {
        throw std::runtime_error(std::string(operation) + " currently requires float32");
    }
}

float read_float_value(const void* data, DType dtype, std::int64_t index) {
    switch (dtype) {
        case DType::Float32:
            return static_cast<const float*>(data)[index];
        case DType::Float16:
            return static_cast<float>(static_cast<const Float16*>(data)[index]);
        case DType::BFloat16:
            return static_cast<float>(static_cast<const BFloat16*>(data)[index]);
        case DType::Float8E4M3FNUZ:
            return static_cast<float>(static_cast<const Float8E4M3FNUZ*>(data)[index]);
        case DType::Float8E5M2FNUZ:
            return static_cast<float>(static_cast<const Float8E5M2FNUZ*>(data)[index]);
        case DType::Int32:
        case DType::Int64:
            throw std::invalid_argument("floating-point access requires a floating dtype");
    }
    throw std::invalid_argument("unknown dtype");
}

void write_float_value(void* data, DType dtype, std::int64_t index, float value) {
    switch (dtype) {
        case DType::Float32:
            static_cast<float*>(data)[index] = value;
            return;
        case DType::Float16:
            static_cast<Float16*>(data)[index] = Float16(value);
            return;
        case DType::BFloat16:
            static_cast<BFloat16*>(data)[index] = BFloat16(value);
            return;
        case DType::Float8E4M3FNUZ:
            static_cast<Float8E4M3FNUZ*>(data)[index] = Float8E4M3FNUZ(value);
            return;
        case DType::Float8E5M2FNUZ:
            static_cast<Float8E5M2FNUZ*>(data)[index] = Float8E5M2FNUZ(value);
            return;
        case DType::Int32:
        case DType::Int64:
            throw std::invalid_argument("floating-point access requires a floating dtype");
    }
    throw std::invalid_argument("unknown dtype");
}

}  // namespace

Strides contiguous_strides(const Shape& shape) {
    (void)checked_numel(shape);
    Strides strides(shape.size(), 1);
    std::int64_t running = 1;
    for (std::size_t reversed = shape.size(); reversed > 0; --reversed) {
        const auto dim = reversed - 1;
        strides[dim] = running;
        running = checked_multiply(running, shape[dim], "shape stride overflows int64");
    }
    return strides;
}

std::int64_t checked_numel(const Shape& shape) {
    std::int64_t elements = 1;
    for (const auto dimension : shape) {
        if (dimension < 0) throw std::invalid_argument("shape dimensions must be non-negative");
        elements = checked_multiply(elements, dimension, "shape element count overflows int64");
    }
    return elements;
}

Tensor::Tensor(Shape shape, DType dtype, Device device)
    : shape_(std::move(shape)), dtype_(dtype), defined_(true) {
    numel_ = checked_numel(shape_);
    strides_ = contiguous_strides(shape_);
    storage_ = Storage(byte_count(numel_, dtype_), device);
    validate_layout();
}

Tensor::Tensor(std::initializer_list<std::int64_t> shape, DType dtype, Device device)
    : Tensor(Shape(shape), dtype, device) {}

Tensor::Tensor(Storage storage, Shape shape, Strides strides, std::int64_t storage_offset,
               DType dtype, bool defined)
    : storage_(std::move(storage)),
      shape_(std::move(shape)),
      strides_(std::move(strides)),
      storage_offset_(storage_offset),
      dtype_(dtype),
      defined_(defined) {
    numel_ = checked_numel(shape_);
    validate_layout();
}

Tensor Tensor::from_vector(const std::vector<float>& values, Shape shape, DType dtype) {
    if (!is_floating_point(dtype)) {
        throw std::invalid_argument("float vector requires a floating tensor dtype");
    }
    Tensor result(std::move(shape), dtype);
    if (static_cast<std::uint64_t>(result.numel()) != values.size()) {
        throw std::invalid_argument("vector size does not match tensor shape");
    }
    for (std::size_t index = 0; index < values.size(); ++index) {
        write_float_value(result.data(), dtype, static_cast<std::int64_t>(index), values[index]);
    }
    return result;
}

Tensor Tensor::from_int32_vector(const std::vector<std::int32_t>& values, Shape shape) {
    Tensor result(std::move(shape), DType::Int32);
    if (static_cast<std::uint64_t>(result.numel()) != values.size()) {
        throw std::invalid_argument("vector size does not match tensor shape");
    }
    if (!values.empty()) {
        std::memcpy(result.data(), values.data(), values.size() * sizeof(std::int32_t));
    }
    return result;
}

Tensor Tensor::from_storage(Storage storage, Shape shape, Strides strides,
                            std::int64_t storage_offset, DType dtype) {
    return Tensor(std::move(storage), std::move(shape), std::move(strides), storage_offset,
                  dtype, true);
}

std::int64_t Tensor::ndim() const noexcept {
    return defined_ ? static_cast<std::int64_t>(shape_.size()) : 0;
}

std::int64_t Tensor::numel() const noexcept { return defined_ ? numel_ : 0; }

std::int64_t Tensor::normalize_dim(std::int64_t dim, bool allow_end) const {
    const auto dimensions = ndim();
    const auto upper = dimensions + (allow_end ? 1 : 0);
    if (dim < 0) dim += upper;
    if (dim < 0 || dim >= upper) throw std::out_of_range("tensor dimension is out of range");
    return dim;
}

std::int64_t Tensor::size(std::int64_t dim) const {
    if (!defined_) throw std::logic_error("undefined tensor has no dimensions");
    return shape_[static_cast<std::size_t>(normalize_dim(dim))];
}

std::int64_t Tensor::stride(std::int64_t dim) const {
    if (!defined_) throw std::logic_error("undefined tensor has no strides");
    return strides_[static_cast<std::size_t>(normalize_dim(dim))];
}

bool Tensor::is_contiguous() const noexcept {
    if (!defined_) return false;
    if (numel_ == 0) return true;
    std::int64_t expected = 1;
    for (std::size_t reversed = shape_.size(); reversed > 0; --reversed) {
        const auto dim = reversed - 1;
        if (shape_[dim] != 1 && strides_[dim] != expected) return false;
        expected *= shape_[dim];
    }
    return true;
}

void Tensor::validate_layout() const {
    if (!defined_) return;
    if (shape_.size() != strides_.size()) {
        throw std::invalid_argument("shape and strides must have the same rank");
    }
    if (storage_offset_ < 0) throw std::invalid_argument("storage offset must be non-negative");
    for (const auto stride_value : strides_) {
        if (stride_value < 0) {
            throw std::invalid_argument("negative strides are not supported in the first release");
        }
    }

    std::int64_t greatest_index = storage_offset_;
    if (numel_ != 0) {
        for (std::size_t dim = 0; dim < shape_.size(); ++dim) {
            if (shape_[dim] > 0) {
                const auto reach = checked_multiply(shape_[dim] - 1, strides_[dim],
                                                    "tensor layout overflows int64");
                if (reach > std::numeric_limits<std::int64_t>::max() - greatest_index) {
                    throw std::overflow_error("tensor layout overflows int64");
                }
                greatest_index += reach;
            }
        }
        if (byte_count(greatest_index + 1, dtype_) > storage_.num_bytes()) {
            throw std::out_of_range("tensor view exceeds its Storage");
        }
    } else if (byte_count(storage_offset_, dtype_) > storage_.num_bytes()) {
        throw std::out_of_range("empty tensor offset exceeds its Storage");
    }
}

void* Tensor::data() {
    if (!defined_) throw std::logic_error("undefined tensor has no data");
    auto* base = static_cast<std::byte*>(storage_.data());
    if (base == nullptr) return nullptr;
    return base + byte_count(storage_offset_, dtype_);
}

const void* Tensor::data() const {
    if (!defined_) throw std::logic_error("undefined tensor has no data");
    const auto* base = static_cast<const std::byte*>(storage_.data());
    if (base == nullptr) return nullptr;
    return base + byte_count(storage_offset_, dtype_);
}

float* Tensor::data_float() {
    require_cpu_float32(*this, "data_float");
    return static_cast<float*>(data());
}

const float* Tensor::data_float() const {
    require_cpu_float32(*this, "data_float");
    return static_cast<const float*>(data());
}

TensorView Tensor::view() { return {data(), dtype_, device(), shape_, strides_}; }
ConstTensorView Tensor::view() const { return {data(), dtype_, device(), shape_, strides_}; }

void Tensor::fill(float value) {
    if (!device().is_cpu()) throw std::runtime_error("fill currently requires a CPU tensor");
    if (!is_floating_point(dtype_)) throw std::runtime_error("fill requires a floating dtype");
    auto* base = storage_.data();
    for (std::int64_t logical = 0; logical < numel_; ++logical) {
        write_float_value(base, dtype_,
                          logical_to_storage_index(logical, shape_, strides_, storage_offset_),
                          value);
    }
}

std::vector<float> Tensor::to_vector() const {
    if (!is_floating_point(dtype_)) throw std::runtime_error("to_vector requires a floating dtype");
    if (device().is_hip() && !is_contiguous()) return contiguous().to_vector();
    std::vector<float> values(static_cast<std::size_t>(numel_));
    if (device().is_hip()) {
        Tensor host(shape_, dtype_, Device::cpu());
        runtime::copy_bytes(host.data(), Device::cpu(), data(), device(),
                            byte_count(numel_, dtype_));
        return host.to_vector();
    }
    const auto* base = storage_.data();
    for (std::int64_t logical = 0; logical < numel_; ++logical) {
        values[static_cast<std::size_t>(logical)] = read_float_value(
            base, dtype_, logical_to_storage_index(logical, shape_, strides_, storage_offset_));
    }
    return values;
}

Tensor Tensor::cast(DType target) const {
    if (!defined_) throw std::logic_error("cannot cast an undefined tensor");
    if (!is_floating_point(dtype_) || !is_floating_point(target)) {
        throw std::invalid_argument("cast currently supports floating dtypes only");
    }
    if (target == dtype_) return *this;
    if (device().is_cpu()) {
        return Tensor::from_vector(to_vector(), shape_, target);
    }
    const auto host = to(Device::cpu()).cast(target);
    return host.to(device());
}

std::vector<std::int32_t> Tensor::to_int32_vector() const {
    if (dtype_ != DType::Int32) {
        throw std::runtime_error("to_int32_vector requires int32");
    }
    if (!is_contiguous()) {
        throw std::runtime_error("to_int32_vector currently requires a contiguous tensor");
    }
    std::vector<std::int32_t> values(static_cast<std::size_t>(numel_));
    runtime::copy_bytes(values.data(), Device::cpu(), data(), device(),
                        byte_count(numel_, dtype_));
    return values;
}

Tensor Tensor::reshape(Shape new_shape) const {
    if (!defined_) throw std::logic_error("cannot reshape an undefined tensor");
    if (!is_contiguous()) throw std::invalid_argument("reshape requires a contiguous tensor");
    if (checked_numel(new_shape) != numel_) {
        throw std::invalid_argument("reshape must preserve the number of elements");
    }
    auto new_strides = contiguous_strides(new_shape);
    return Tensor(storage_, std::move(new_shape), std::move(new_strides), storage_offset_, dtype_,
                  true);
}

Tensor Tensor::transpose(std::int64_t dim0, std::int64_t dim1) const {
    if (!defined_) throw std::logic_error("cannot transpose an undefined tensor");
    const auto first = static_cast<std::size_t>(normalize_dim(dim0));
    const auto second = static_cast<std::size_t>(normalize_dim(dim1));
    auto new_shape = shape_;
    auto new_strides = strides_;
    std::swap(new_shape[first], new_shape[second]);
    std::swap(new_strides[first], new_strides[second]);
    return Tensor(storage_, std::move(new_shape), std::move(new_strides), storage_offset_, dtype_,
                  true);
}

Tensor Tensor::slice(std::int64_t dim, std::int64_t start, std::int64_t end,
                     std::int64_t step) const {
    if (!defined_) throw std::logic_error("cannot slice an undefined tensor");
    const auto normalized = static_cast<std::size_t>(normalize_dim(dim));
    const auto length = shape_[normalized];
    if (step <= 0) throw std::invalid_argument("slice step must be positive");
    if (start < 0 || end < start || end > length) {
        throw std::out_of_range("slice bounds are invalid");
    }
    auto new_shape = shape_;
    auto new_strides = strides_;
    new_shape[normalized] = (end - start + step - 1) / step;
    const auto new_offset = storage_offset_ + start * strides_[normalized];
    new_strides[normalized] *= step;
    return Tensor(storage_, std::move(new_shape), std::move(new_strides), new_offset, dtype_, true);
}

Tensor Tensor::unsqueeze(std::int64_t dim) const {
    if (!defined_) throw std::logic_error("cannot unsqueeze an undefined tensor");
    const auto normalized = static_cast<std::size_t>(normalize_dim(dim, true));
    auto new_shape = shape_;
    auto new_strides = strides_;
    const auto inserted_stride = normalized == shape_.size()
                                     ? 1
                                     : strides_[normalized] * std::max<std::int64_t>(shape_[normalized], 1);
    new_shape.insert(new_shape.begin() + static_cast<std::ptrdiff_t>(normalized), 1);
    new_strides.insert(new_strides.begin() + static_cast<std::ptrdiff_t>(normalized),
                       inserted_stride);
    return Tensor(storage_, std::move(new_shape), std::move(new_strides), storage_offset_, dtype_,
                  true);
}

Tensor Tensor::squeeze(std::int64_t dim) const {
    if (!defined_) throw std::logic_error("cannot squeeze an undefined tensor");
    auto new_shape = shape_;
    auto new_strides = strides_;
    if (dim == -1) {
        for (std::size_t reversed = new_shape.size(); reversed > 0; --reversed) {
            const auto index = reversed - 1;
            if (new_shape[index] == 1) {
                new_shape.erase(new_shape.begin() + static_cast<std::ptrdiff_t>(index));
                new_strides.erase(new_strides.begin() + static_cast<std::ptrdiff_t>(index));
            }
        }
    } else {
        const auto normalized = static_cast<std::size_t>(normalize_dim(dim));
        if (new_shape[normalized] != 1) {
            throw std::invalid_argument("squeeze dimension must have size one");
        }
        new_shape.erase(new_shape.begin() + static_cast<std::ptrdiff_t>(normalized));
        new_strides.erase(new_strides.begin() + static_cast<std::ptrdiff_t>(normalized));
    }
    return Tensor(storage_, std::move(new_shape), std::move(new_strides), storage_offset_, dtype_,
                  true);
}

Tensor Tensor::contiguous() const {
    if (!defined_) throw std::logic_error("cannot copy an undefined tensor");
    if (is_contiguous()) return *this;
    Tensor output(shape_, dtype_, device());
    runtime::copy_strided(output.data(), data(), dtype_size(dtype_), device(), shape_, strides_);
    return output;
}

Tensor Tensor::to(Device target) const {
    if (!defined_) throw std::logic_error("cannot transfer an undefined tensor");
    if (target == device()) return *this;
    if (!is_contiguous()) {
        if (device().is_hip()) {
            throw std::runtime_error("non-contiguous HIP transfer is not implemented yet");
        }
        return contiguous().to(target);
    }
    Tensor result(shape_, dtype_, target);
    runtime::copy_bytes(result.data(), target, data(), device(), byte_count(numel_, dtype_));
    return result;
}

std::string Tensor::str() const {
    if (!defined_) return "Tensor(undefined)";
    std::ostringstream output;
    output << "Tensor(shape=[";
    for (std::size_t index = 0; index < shape_.size(); ++index) {
        if (index != 0) output << ", ";
        output << shape_[index];
    }
    output << "], dtype=" << dtype_name(dtype_) << ", device=" << device().str()
           << ", contiguous=" << (is_contiguous() ? "true" : "false") << ')';
    return output.str();
}

std::ostream& operator<<(std::ostream& stream, const Tensor& tensor) {
    return stream << tensor.str();
}

}  // namespace microllm
