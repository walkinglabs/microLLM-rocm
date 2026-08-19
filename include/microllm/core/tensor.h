#pragma once

#include <cstdint>
#include <initializer_list>
#include <iosfwd>
#include <string>
#include <vector>

#include <microllm/base/dtype.h>
#include <microllm/core/storage.h>
#include <microllm/core/tensor_view.h>

namespace microllm {

using Shape = std::vector<std::int64_t>;
using Strides = std::vector<std::int64_t>;

[[nodiscard]] Strides contiguous_strides(const Shape& shape);
[[nodiscard]] std::int64_t checked_numel(const Shape& shape);

class Tensor {
public:
    Tensor() = default;
    explicit Tensor(Shape shape, DType dtype = DType::Float32,
                    Device device = Device::cpu());
    Tensor(std::initializer_list<std::int64_t> shape,
           DType dtype = DType::Float32, Device device = Device::cpu());

    [[nodiscard]] static Tensor from_vector(const std::vector<float>& values, Shape shape);
    [[nodiscard]] static Tensor from_int32_vector(const std::vector<std::int32_t>& values,
                                                  Shape shape);
    [[nodiscard]] static Tensor from_storage(Storage storage, Shape shape, Strides strides,
                                             std::int64_t storage_offset = 0,
                                             DType dtype = DType::Float32);

    [[nodiscard]] bool defined() const noexcept { return defined_; }
    [[nodiscard]] bool empty() const noexcept { return !defined_ || numel() == 0; }
    [[nodiscard]] const Shape& shape() const noexcept { return shape_; }
    [[nodiscard]] const Strides& strides() const noexcept { return strides_; }
    [[nodiscard]] std::int64_t ndim() const noexcept;
    [[nodiscard]] std::int64_t numel() const noexcept;
    [[nodiscard]] std::int64_t size(std::int64_t dim) const;
    [[nodiscard]] std::int64_t stride(std::int64_t dim) const;
    [[nodiscard]] DType dtype() const noexcept { return dtype_; }
    [[nodiscard]] Device device() const noexcept { return storage_.device(); }
    [[nodiscard]] std::int64_t storage_offset() const noexcept { return storage_offset_; }
    [[nodiscard]] Storage storage() const noexcept { return storage_; }
    [[nodiscard]] bool is_contiguous() const noexcept;

    [[nodiscard]] void* data();
    [[nodiscard]] const void* data() const;
    [[nodiscard]] float* data_float();
    [[nodiscard]] const float* data_float() const;
    [[nodiscard]] TensorView view();
    [[nodiscard]] ConstTensorView view() const;

    void fill(float value);
    [[nodiscard]] std::vector<float> to_vector() const;
    [[nodiscard]] std::vector<std::int32_t> to_int32_vector() const;

    [[nodiscard]] Tensor reshape(Shape new_shape) const;
    [[nodiscard]] Tensor transpose(std::int64_t dim0, std::int64_t dim1) const;
    [[nodiscard]] Tensor slice(std::int64_t dim, std::int64_t start,
                               std::int64_t end, std::int64_t step = 1) const;
    [[nodiscard]] Tensor unsqueeze(std::int64_t dim) const;
    [[nodiscard]] Tensor squeeze(std::int64_t dim = -1) const;
    [[nodiscard]] Tensor contiguous() const;
    [[nodiscard]] Tensor to(Device target) const;

    [[nodiscard]] std::string str() const;
    friend std::ostream& operator<<(std::ostream& stream, const Tensor& tensor);

private:
    Tensor(Storage storage, Shape shape, Strides strides, std::int64_t storage_offset,
           DType dtype, bool defined);
    void validate_layout() const;
    [[nodiscard]] std::int64_t normalize_dim(std::int64_t dim, bool allow_end = false) const;

    Storage storage_;
    Shape shape_;
    Strides strides_;
    std::int64_t storage_offset_ = 0;
    std::int64_t numel_ = 0;
    DType dtype_ = DType::Float32;
    bool defined_ = false;
};

}  // namespace microllm
