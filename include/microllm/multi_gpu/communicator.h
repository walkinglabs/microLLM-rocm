#pragma once

#include <cstddef>
#include <memory>
#include <vector>

#include <microllm/core/tensor.h>
#include <microllm/runtime/runtime.h>

namespace microllm::multi_gpu {

class Communicator {
public:
    explicit Communicator(std::vector<int> device_indices);
    ~Communicator();
    Communicator(Communicator&&) noexcept;
    Communicator& operator=(Communicator&&) noexcept;
    Communicator(const Communicator&) = delete;
    Communicator& operator=(const Communicator&) = delete;

    [[nodiscard]] std::size_t size() const noexcept;
    [[nodiscard]] const std::vector<int>& devices() const noexcept;
    [[nodiscard]] bool aborted() const noexcept;
    [[nodiscard]] runtime::Stream& stream(std::size_t rank);

    void all_reduce(std::vector<Tensor>& tensors, bool average = true);
    void synchronize();
    void abort() noexcept;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace microllm::multi_gpu
