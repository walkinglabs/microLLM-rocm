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
    Cancelled,
};

enum class CompletionReason : std::uint8_t {
    None,
    Length,
    StopToken,
    Cancelled,
};

struct RequestSnapshot {
    RequestId id = 0;
    RequestState state = RequestState::PendingPrefill;
    CompletionReason completion_reason = CompletionReason::None;
    std::vector<std::int32_t> prompt;
    std::vector<std::int32_t> generated;
    std::int64_t max_new_tokens = 0;
    std::int64_t arrival_step = 0;
    std::int64_t completion_step = -1;
    std::size_t cache_bytes = 0;
    // Active slot in a shared continuous batch. -1 means pending or terminal.
    std::int64_t slot = -1;
};

struct SchedulerMetrics {
    std::int64_t scheduler_steps = 0;
    std::int64_t submitted_requests = 0;
    std::int64_t completed_requests = 0;
    std::int64_t cancelled_requests = 0;
    std::int64_t stop_completed_requests = 0;
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
    [[nodiscard]] bool cancel(RequestId id);
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

struct AdmissionBatchMetrics {
    std::int64_t drain_calls = 0;
    std::int64_t submitted_requests = 0;
    std::int64_t completed_requests = 0;
    std::int64_t cancelled_requests = 0;
    std::int64_t stop_completed_requests = 0;
    std::int64_t batch_groups = 0;
    std::int64_t singleton_groups = 0;
    std::int64_t batched_requests = 0;
    std::int64_t maximum_batch_size = 0;
};

// Groups currently pending requests by an exact compatibility key and executes
// each group through generate_batch(). Requests submitted after drain() wait for
// the next admission window. This is admission batching, not token-level refill.
class AdmissionBatchScheduler {
public:
    explicit AdmissionBatchScheduler(model::TransformerModel& model);
    ~AdmissionBatchScheduler();
    AdmissionBatchScheduler(AdmissionBatchScheduler&&) noexcept;
    AdmissionBatchScheduler& operator=(AdmissionBatchScheduler&&) noexcept;
    AdmissionBatchScheduler(const AdmissionBatchScheduler&) = delete;
    AdmissionBatchScheduler& operator=(const AdmissionBatchScheduler&) = delete;

    [[nodiscard]] RequestId submit(
        std::vector<std::int32_t> prompt,
        GenerationConfig config = {});
    [[nodiscard]] bool cancel(RequestId id);
    void drain();
    [[nodiscard]] std::size_t pending_request_count() const noexcept;
    [[nodiscard]] RequestSnapshot request(RequestId id) const;
    [[nodiscard]] std::vector<RequestSnapshot> requests() const;
    [[nodiscard]] AdmissionBatchMetrics metrics() const noexcept;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

struct ContinuousBatchConfig {
    std::int64_t max_slots = 1;
    // Zero uses the model maximum. A positive value bounds shared KV capacity
    // for a known serving workload and must not exceed the model maximum.
    std::int64_t max_sequence_length = 0;
    DType kv_cache_dtype = DType::Float32;
    // Empty applies kv_cache_dtype to every layer.
    std::vector<DType> kv_cache_layer_dtypes;
};

struct ContinuousBatchMetrics {
    std::int64_t scheduler_steps = 0;
    std::int64_t submitted_requests = 0;
    std::int64_t completed_requests = 0;
    std::int64_t cancelled_requests = 0;
    std::int64_t stop_completed_requests = 0;
    std::int64_t slot_admissions = 0;
    std::int64_t slot_refills = 0;
    std::int64_t row_prefill_calls = 0;
    std::int64_t prefill_batch_calls = 0;
    std::int64_t batched_prefill_calls = 0;
    std::int64_t batched_prefill_rows = 0;
    std::int64_t batch_decode_calls = 0;
    std::int64_t uniform_batch_decode_calls = 0;
    std::int64_t divergent_batch_decode_calls = 0;
    std::int64_t compacted_batch_decode_calls = 0;
    std::int64_t positions_aware_batch_decode_calls = 0;
    std::int64_t logical_decode_rows = 0;
    std::int64_t dummy_decode_rows = 0;
    std::int64_t inactive_rows_skipped = 0;
    std::int64_t selection_calls = 0;
    std::int64_t occupied_slots = 0;
    std::int64_t peak_occupied_slots = 0;
    std::int64_t occupied_slot_steps = 0;
    double slot_utilization = 0.0;
    std::size_t allocated_cache_bytes = 0;
    std::size_t active_cache_bytes = 0;
    std::size_t peak_active_cache_bytes = 0;
};

// Correctness-first continuous batching. A fixed shared KV cache owns
// max_slots rows. Pending requests enter free rows, completed/cancelled rows
// are reset, and later requests may reuse them. Divergent positions currently
// execute through TransformerModel::forward_cached_rows(), so this class proves
// request/slot semantics before positions-aware parallel kernels are added.
class ContinuousBatchScheduler {
public:
    explicit ContinuousBatchScheduler(
        model::TransformerModel& model,
        ContinuousBatchConfig config = {});
    ~ContinuousBatchScheduler();
    ContinuousBatchScheduler(ContinuousBatchScheduler&&) noexcept;
    ContinuousBatchScheduler& operator=(ContinuousBatchScheduler&&) noexcept;
    ContinuousBatchScheduler(const ContinuousBatchScheduler&) = delete;
    ContinuousBatchScheduler& operator=(const ContinuousBatchScheduler&) = delete;

    [[nodiscard]] RequestId submit(
        std::vector<std::int32_t> prompt,
        GenerationConfig config = {});
    [[nodiscard]] bool cancel(RequestId id);
    void step();
    void run_until_idle(std::int64_t maximum_steps = -1);

    [[nodiscard]] bool has_active_requests() const noexcept;
    [[nodiscard]] std::size_t active_request_count() const noexcept;
    [[nodiscard]] std::size_t pending_request_count() const noexcept;
    [[nodiscard]] RequestSnapshot request(RequestId id) const;
    [[nodiscard]] std::vector<RequestSnapshot> requests() const;
    [[nodiscard]] ContinuousBatchMetrics metrics() const noexcept;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace microllm::inference
