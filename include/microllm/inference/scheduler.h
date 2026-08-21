#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

#include <microllm/inference/generator.h>

namespace microllm::inference {

using RequestId = std::uint64_t;

enum class RequestState : std::uint8_t {
    PendingPrefill,
    Decoding,
    Completed,
};

struct RequestSnapshot {
    RequestId id = 0;
    RequestState state = RequestState::PendingPrefill;
    std::vector<std::int32_t> prompt;
    std::vector<std::int32_t> generated;
    std::int64_t max_new_tokens = 0;
    std::int64_t arrival_step = 0;
    std::int64_t completion_step = -1;
    std::size_t cache_bytes = 0;
};

struct SchedulerMetrics {
    std::int64_t scheduler_steps = 0;
    std::int64_t submitted_requests = 0;
    std::int64_t completed_requests = 0;
    std::int64_t prefill_calls = 0;
    std::int64_t decode_calls = 0;
    std::int64_t peak_active_requests = 0;
    std::size_t active_cache_bytes = 0;
    std::size_t peak_cache_bytes = 0;
};

// Correctness-first serving reference. Requests own independent B=1 caches and
// advance one generated token per scheduler step. It intentionally performs no
// cross-request batching; optimized schedulers must match this state machine.
class ReferenceScheduler {
public:
    explicit ReferenceScheduler(model::TransformerModel& model);
    ~ReferenceScheduler();
    ReferenceScheduler(ReferenceScheduler&&) noexcept;
    ReferenceScheduler& operator=(ReferenceScheduler&&) noexcept;
    ReferenceScheduler(const ReferenceScheduler&) = delete;
    ReferenceScheduler& operator=(const ReferenceScheduler&) = delete;

    [[nodiscard]] RequestId submit(
        std::vector<std::int32_t> prompt,
        GenerationConfig config = {});
    void step();
    void run_until_idle(std::int64_t maximum_steps = -1);

    [[nodiscard]] bool has_active_requests() const noexcept;
    [[nodiscard]] std::size_t active_request_count() const noexcept;
    [[nodiscard]] RequestSnapshot request(RequestId id) const;
    [[nodiscard]] std::vector<RequestSnapshot> requests() const;
    [[nodiscard]] SchedulerMetrics metrics() const noexcept;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace microllm::inference
