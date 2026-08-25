#pragma once

#include <cstddef>
#include <cstdint>
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

    void all_reduce(std::vector<Tensor>& tensors, bool average = true,
                    bool in_place_average = true);
    void enqueue_all_reduce_sum(std::vector<Tensor>& tensors);
    void enqueue_all_reduce_average_in_place(std::vector<Tensor>& tensors);
    void synchronize();
    void abort() noexcept;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

using CommunicatorId = std::vector<std::uint8_t>;

[[nodiscard]] CommunicatorId create_communicator_id();
[[nodiscard]] std::size_t communicator_id_bytes() noexcept;

// One process owns one rank and one local GPU. The opaque ID must be generated
// once and delivered byte-for-byte to every rank before construction.
class RankCommunicator {
public:
    RankCommunicator(int rank, int world_size, int local_device,
                     const CommunicatorId& id);
    ~RankCommunicator();
    RankCommunicator(RankCommunicator&&) noexcept;
    RankCommunicator& operator=(RankCommunicator&&) noexcept;
    RankCommunicator(const RankCommunicator&) = delete;
    RankCommunicator& operator=(const RankCommunicator&) = delete;

    [[nodiscard]] int rank() const noexcept;
    [[nodiscard]] int world_size() const noexcept;
    [[nodiscard]] Device device() const noexcept;
    [[nodiscard]] bool aborted() const noexcept;
    [[nodiscard]] runtime::Stream& stream();
    // Optionally weights this rank's local mean before the collective, then
    // divides the global sum by world size on the same communication Stream.
    void enqueue_all_reduce_average_in_place(
        Tensor& tensor, float local_scale = 1.0F);
    void synchronize();
    void abort() noexcept;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace microllm::multi_gpu
