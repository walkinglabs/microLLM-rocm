#include <microllm/inference/scheduler.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <random>
#include <stdexcept>
#include <utility>

#include <microllm/inference/kv_cache.h>
#include <microllm/ops/ops.h>
#include <microllm/runtime/memory.h>
#include <microllm/runtime/runtime.h>

namespace microllm::inference {
namespace {

std::vector<DType> cache_policy(const model::TransformerModel& model,
                                const GenerationConfig& config) {
    auto policy = config.kv_cache_layer_dtypes;
    if (policy.empty()) {
        policy.assign(static_cast<std::size_t>(model.config().layers),
                      config.kv_cache_dtype);
    } else if (policy.size() != static_cast<std::size_t>(model.config().layers)) {
        throw std::invalid_argument(
            "scheduler KV cache policy must contain one dtype per layer");
    }
    return policy;
}

void validate_request(const model::TransformerModel& model,
                      const std::vector<std::int32_t>& prompt,
                      const GenerationConfig& config) {
    if (prompt.empty()) throw std::invalid_argument("scheduler prompt cannot be empty");
    if (config.max_new_tokens < 0) {
        throw std::invalid_argument("scheduler max_new_tokens cannot be negative");
    }
    if (static_cast<std::int64_t>(prompt.size()) + config.max_new_tokens >
        model.config().max_sequence_length) {
        throw std::invalid_argument("scheduler request exceeds model context");
    }
    if (config.temperature < 0.0F || !std::isfinite(config.temperature) ||
        config.top_k < 0 || config.top_k > model.config().vocabulary_size) {
        throw std::invalid_argument("scheduler sampling configuration is invalid");
    }
    for (const auto token : prompt) {
        if (token < 0 || token >= model.config().vocabulary_size) {
            throw std::out_of_range("scheduler prompt token is outside the vocabulary");
        }
    }
    auto stop_tokens = config.stop_tokens;
    std::sort(stop_tokens.begin(), stop_tokens.end());
    if (std::adjacent_find(stop_tokens.begin(), stop_tokens.end()) !=
        stop_tokens.end()) {
        throw std::invalid_argument("scheduler stop tokens must be unique");
    }
    for (const auto token : stop_tokens) {
        if (token < 0 || token >= model.config().vocabulary_size) {
            throw std::out_of_range("scheduler stop token is outside the vocabulary");
        }
    }
}

std::size_t cache_bytes(const KVCache& cache) {
    std::size_t bytes = 0;
    for (std::size_t layer = 0; layer < cache.layer_count(); ++layer) {
        for (const auto* tensor : {&cache.layer(layer).key,
                                   &cache.layer(layer).value}) {
            if (tensor->defined()) bytes += tensor->storage().num_bytes();
        }
    }
    return bytes;
}

bool is_terminal(RequestState state) {
    return state == RequestState::Completed || state == RequestState::Cancelled;
}

bool is_stop_token(const GenerationConfig& config, std::int32_t token) {
    return std::find(config.stop_tokens.begin(), config.stop_tokens.end(), token) !=
           config.stop_tokens.end();
}

}  // namespace

struct ReferenceScheduler::Impl {
    struct Request {
        RequestId id = 0;
        RequestState state = RequestState::PendingPrefill;
        CompletionReason completion_reason = CompletionReason::None;
        std::vector<std::int32_t> prompt;
        std::vector<std::int32_t> generated;
        GenerationConfig config;
        std::mt19937_64 random;
        std::unique_ptr<KVCache> cache;
        Tensor logits;
        std::int64_t arrival_step = 0;
        std::int64_t completion_step = -1;
    };

    explicit Impl(model::TransformerModel& value) : model(value) {}

    model::TransformerModel& model;
    std::vector<Request> requests;
    RequestId next_id = 1;
    SchedulerMetrics metrics;

    Request& find(RequestId id) {
        const auto found = std::find_if(requests.begin(), requests.end(),
                                        [id](const Request& request) {
                                            return request.id == id;
                                        });
        if (found == requests.end()) throw std::out_of_range("unknown scheduler request");
        return *found;
    }

    const Request& find(RequestId id) const {
        return const_cast<Impl*>(this)->find(id);
    }

    std::int32_t select(Request& request) {
        if (request.logits.device().is_hip() &&
            (request.config.temperature == 0.0F || request.config.top_k == 1)) {
            const auto selected = ops::argmax(request.logits).to_int32_vector().front();
            if (selected < 0) throw std::invalid_argument("scheduler logits are non-finite");
            return selected;
        }
        return sample_token(request.logits.to_vector(), request.config.temperature,
                            request.config.top_k, request.random);
    }

    void refresh_cache_metrics() {
        std::size_t active_bytes = 0;
        std::int64_t active = 0;
        for (const auto& request : requests) {
            if (is_terminal(request.state)) continue;
            ++active;
            if (request.cache) active_bytes += cache_bytes(*request.cache);
        }
        metrics.active_cache_bytes = active_bytes;
        metrics.peak_cache_bytes = std::max(metrics.peak_cache_bytes, active_bytes);
        metrics.peak_active_requests = std::max(metrics.peak_active_requests, active);
    }
};

ReferenceScheduler::ReferenceScheduler(model::TransformerModel& model)
    : impl_(std::make_unique<Impl>(model)) {}
ReferenceScheduler::~ReferenceScheduler() = default;
ReferenceScheduler::ReferenceScheduler(ReferenceScheduler&&) noexcept = default;
ReferenceScheduler& ReferenceScheduler::operator=(ReferenceScheduler&&) noexcept = default;

RequestId ReferenceScheduler::submit(std::vector<std::int32_t> prompt,
                                     GenerationConfig config) {
    validate_request(impl_->model, prompt, config);
    std::sort(config.stop_tokens.begin(), config.stop_tokens.end());
    auto policy = cache_policy(impl_->model, config);
    const auto id = impl_->next_id++;
    Impl::Request request;
    request.id = id;
    request.prompt = std::move(prompt);
    request.random = std::mt19937_64(config.seed);
    request.config = std::move(config);
    request.arrival_step = impl_->metrics.scheduler_steps;
    if (request.config.max_new_tokens == 0) {
        request.state = RequestState::Completed;
        request.completion_reason = CompletionReason::Length;
        request.completion_step = impl_->metrics.scheduler_steps;
        ++impl_->metrics.completed_requests;
    } else {
        request.cache = std::make_unique<KVCache>(
            std::move(policy),
            static_cast<std::int64_t>(request.prompt.size()) +
                request.config.max_new_tokens);
    }
    impl_->requests.push_back(std::move(request));
    ++impl_->metrics.submitted_requests;
    impl_->refresh_cache_metrics();
    return id;
}

bool ReferenceScheduler::cancel(RequestId id) {
    auto& request = impl_->find(id);
    if (is_terminal(request.state)) return false;
    request.state = RequestState::Cancelled;
    request.completion_reason = CompletionReason::Cancelled;
    request.completion_step = impl_->metrics.scheduler_steps;
    request.logits = {};
    request.cache.reset();
    ++impl_->metrics.cancelled_requests;
    impl_->refresh_cache_metrics();
    return true;
}

void ReferenceScheduler::step() {
    if (!has_active_requests()) return;
    ++impl_->metrics.scheduler_steps;
    for (auto& request : impl_->requests) {
        if (is_terminal(request.state)) continue;
        if (request.state == RequestState::PendingPrefill) {
            request.logits = impl_->model.forward_prefill_cached(
                Tensor::from_int32_vector(
                    request.prompt,
                    {1, static_cast<std::int64_t>(request.prompt.size())}),
                *request.cache);
            request.state = RequestState::Decoding;
            ++impl_->metrics.prefill_calls;
            impl_->refresh_cache_metrics();
        }
        const auto next = impl_->select(request);
        request.generated.push_back(next);
        const auto stopped = is_stop_token(request.config, next);
        if (stopped || static_cast<std::int64_t>(request.generated.size()) ==
                           request.config.max_new_tokens) {
            request.state = RequestState::Completed;
            request.completion_reason = stopped ? CompletionReason::StopToken
                                                : CompletionReason::Length;
            request.completion_step = impl_->metrics.scheduler_steps;
            ++impl_->metrics.completed_requests;
            if (stopped) ++impl_->metrics.stop_completed_requests;
            request.logits = {};
            request.cache.reset();
            continue;
        }
        request.logits = impl_->model.forward_cached(
            Tensor::from_int32_vector({next}, {1, 1}), *request.cache);
        ++impl_->metrics.decode_calls;
    }
    impl_->refresh_cache_metrics();
}

void ReferenceScheduler::run_until_idle(std::int64_t maximum_steps) {
    if (maximum_steps < -1) throw std::invalid_argument("maximum_steps is invalid");
    std::int64_t executed = 0;
    while (has_active_requests() && (maximum_steps < 0 || executed < maximum_steps)) {
        step();
        ++executed;
    }
    if (has_active_requests()) {
        throw std::runtime_error("scheduler did not become idle within maximum_steps");
    }
}

bool ReferenceScheduler::has_active_requests() const noexcept {
    return std::any_of(impl_->requests.begin(), impl_->requests.end(),
                       [](const Impl::Request& request) {
                           return !is_terminal(request.state);
                       });
}

std::size_t ReferenceScheduler::active_request_count() const noexcept {
    return static_cast<std::size_t>(std::count_if(
        impl_->requests.begin(), impl_->requests.end(),
        [](const Impl::Request& request) {
            return !is_terminal(request.state);
        }));
}

RequestSnapshot ReferenceScheduler::request(RequestId id) const {
    const auto& request = impl_->find(id);
    return {.id = request.id,
            .state = request.state,
            .completion_reason = request.completion_reason,
            .prompt = request.prompt,
            .generated = request.generated,
            .max_new_tokens = request.config.max_new_tokens,
            .arrival_step = request.arrival_step,
            .completion_step = request.completion_step,
            .cache_bytes = request.cache ? cache_bytes(*request.cache) : 0U};
}

std::vector<RequestSnapshot> ReferenceScheduler::requests() const {
    std::vector<RequestSnapshot> result;
    result.reserve(impl_->requests.size());
    for (const auto& request : impl_->requests) result.push_back(this->request(request.id));
    return result;
}

SchedulerMetrics ReferenceScheduler::metrics() const noexcept {
    return impl_->metrics;
}

struct AdmissionBatchScheduler::Impl {
    struct Request {
        RequestId id = 0;
        RequestState state = RequestState::PendingPrefill;
        std::vector<std::int32_t> prompt;
        std::vector<std::int32_t> generated;
        GenerationConfig config;
        CompletionReason completion_reason = CompletionReason::None;
        std::int64_t arrival_drain = 0;
        std::int64_t completion_drain = -1;
    };

    struct Key {
        std::size_t prompt_length = 0;
        std::int64_t max_new_tokens = 0;
        float temperature = 0.0F;
        std::int64_t top_k = 0;
        std::uint64_t seed = 0;
        std::vector<std::int32_t> stop_tokens;
        DType cache_dtype = DType::Float32;
        std::vector<DType> layer_dtypes;

        bool operator==(const Key&) const = default;
    };

    explicit Impl(model::TransformerModel& value) : model(value) {}

    model::TransformerModel& model;
    std::vector<Request> requests;
    RequestId next_id = 1;
    AdmissionBatchMetrics metrics;

    static Key key(const Request& request) {
        return {.prompt_length = request.prompt.size(),
                .max_new_tokens = request.config.max_new_tokens,
                .temperature = request.config.temperature,
                .top_k = request.config.top_k,
                .seed = request.config.seed,
                .stop_tokens = request.config.stop_tokens,
                .cache_dtype = request.config.kv_cache_dtype,
                .layer_dtypes = request.config.kv_cache_layer_dtypes};
    }

    Request& find(RequestId id) {
        const auto found = std::find_if(requests.begin(), requests.end(),
                                        [id](const Request& request) {
                                            return request.id == id;
                                        });
        if (found == requests.end()) throw std::out_of_range("unknown admission request");
        return *found;
    }

    const Request& find(RequestId id) const {
        return const_cast<Impl*>(this)->find(id);
    }
};

AdmissionBatchScheduler::AdmissionBatchScheduler(model::TransformerModel& model)
    : impl_(std::make_unique<Impl>(model)) {}
AdmissionBatchScheduler::~AdmissionBatchScheduler() = default;
AdmissionBatchScheduler::AdmissionBatchScheduler(AdmissionBatchScheduler&&) noexcept = default;
AdmissionBatchScheduler& AdmissionBatchScheduler::operator=(
    AdmissionBatchScheduler&&) noexcept = default;

RequestId AdmissionBatchScheduler::submit(std::vector<std::int32_t> prompt,
                                          GenerationConfig config) {
    validate_request(impl_->model, prompt, config);
    std::sort(config.stop_tokens.begin(), config.stop_tokens.end());
    (void)cache_policy(impl_->model, config);
    Impl::Request request;
    request.id = impl_->next_id++;
    request.prompt = std::move(prompt);
    request.config = std::move(config);
    request.arrival_drain = impl_->metrics.drain_calls;
    if (request.config.max_new_tokens == 0) {
        request.state = RequestState::Completed;
        request.completion_reason = CompletionReason::Length;
        request.completion_drain = impl_->metrics.drain_calls;
        ++impl_->metrics.completed_requests;
    }
    const auto id = request.id;
    impl_->requests.push_back(std::move(request));
    ++impl_->metrics.submitted_requests;
    return id;
}

bool AdmissionBatchScheduler::cancel(RequestId id) {
    auto& request = impl_->find(id);
    if (is_terminal(request.state)) return false;
    request.state = RequestState::Cancelled;
    request.completion_reason = CompletionReason::Cancelled;
    request.completion_drain = impl_->metrics.drain_calls;
    ++impl_->metrics.cancelled_requests;
    return true;
}

void AdmissionBatchScheduler::drain() {
    ++impl_->metrics.drain_calls;
    std::vector<std::vector<std::size_t>> groups;
    std::vector<Impl::Key> keys;
    for (std::size_t index = 0; index < impl_->requests.size(); ++index) {
        if (is_terminal(impl_->requests[index].state)) continue;
        const auto key = Impl::key(impl_->requests[index]);
        const auto found = std::find(keys.begin(), keys.end(), key);
        if (found == keys.end()) {
            keys.push_back(key);
            groups.push_back({index});
        } else {
            groups[static_cast<std::size_t>(std::distance(keys.begin(), found))]
                .push_back(index);
        }
    }
    for (const auto& group : groups) {
        std::vector<std::vector<std::int32_t>> prompts;
        prompts.reserve(group.size());
        for (const auto index : group) prompts.push_back(impl_->requests[index].prompt);
        const auto generated = generate_batch(
            impl_->model, prompts, impl_->requests[group.front()].config);
        if (generated.size() != group.size()) {
            throw std::runtime_error("admission batch returned the wrong row count");
        }
        ++impl_->metrics.batch_groups;
        impl_->metrics.maximum_batch_size = std::max(
            impl_->metrics.maximum_batch_size,
            static_cast<std::int64_t>(group.size()));
        if (group.size() == 1) {
            ++impl_->metrics.singleton_groups;
        } else {
            impl_->metrics.batched_requests += static_cast<std::int64_t>(group.size());
        }
        for (std::size_t row = 0; row < group.size(); ++row) {
            auto& request = impl_->requests[group[row]];
            request.generated.assign(
                generated[row].begin() +
                    static_cast<std::ptrdiff_t>(request.prompt.size()),
                generated[row].end());
            request.state = RequestState::Completed;
            const auto stopped = !request.generated.empty() &&
                                 is_stop_token(request.config,
                                               request.generated.back());
            request.completion_reason = stopped ? CompletionReason::StopToken
                                                : CompletionReason::Length;
            request.completion_drain = impl_->metrics.drain_calls;
            ++impl_->metrics.completed_requests;
            if (stopped) ++impl_->metrics.stop_completed_requests;
        }
    }
}

std::size_t AdmissionBatchScheduler::pending_request_count() const noexcept {
    return static_cast<std::size_t>(std::count_if(
        impl_->requests.begin(), impl_->requests.end(),
        [](const Impl::Request& request) {
            return !is_terminal(request.state);
        }));
}

RequestSnapshot AdmissionBatchScheduler::request(RequestId id) const {
    const auto& request = impl_->find(id);
    return {.id = request.id,
            .state = request.state,
            .completion_reason = request.completion_reason,
            .prompt = request.prompt,
            .generated = request.generated,
            .max_new_tokens = request.config.max_new_tokens,
            .arrival_step = request.arrival_drain,
            .completion_step = request.completion_drain,
            .cache_bytes = 0};
}

std::vector<RequestSnapshot> AdmissionBatchScheduler::requests() const {
    std::vector<RequestSnapshot> result;
    result.reserve(impl_->requests.size());
    for (const auto& request : impl_->requests) result.push_back(this->request(request.id));
    return result;
}

AdmissionBatchMetrics AdmissionBatchScheduler::metrics() const noexcept {
    return impl_->metrics;
}

struct ContinuousBatchScheduler::Impl {
    struct Request {
        RequestId id = 0;
        RequestState state = RequestState::PendingPrefill;
        CompletionReason completion_reason = CompletionReason::None;
        std::vector<std::int32_t> prompt;
        std::vector<std::int32_t> generated;
        GenerationConfig config;
        std::mt19937_64 random;
        std::int64_t arrival_step = 0;
        std::int64_t completion_step = -1;
        std::int64_t slot = -1;
    };

    static std::vector<DType> configured_policy(
        const model::TransformerModel& model,
        const ContinuousBatchConfig& config) {
        if (config.max_slots <= 0) {
            throw std::invalid_argument("continuous scheduler max_slots must be positive");
        }
        auto policy = config.kv_cache_layer_dtypes;
        if (policy.empty()) {
            policy.assign(static_cast<std::size_t>(model.config().layers),
                          config.kv_cache_dtype);
        } else if (policy.size() !=
                   static_cast<std::size_t>(model.config().layers)) {
            throw std::invalid_argument(
                "continuous scheduler cache policy must contain one dtype per layer");
        }
        for (const auto dtype : policy) {
            if (dtype != DType::Float32 && dtype != DType::BFloat16) {
                throw std::invalid_argument(
                    "continuous scheduler cache dtype must be float32 or bfloat16");
            }
        }
        return policy;
    }

    Impl(model::TransformerModel& value, ContinuousBatchConfig settings)
        : model(value),
          config(std::move(settings)),
          policy(configured_policy(value, config)),
          cache(policy, value.config().max_sequence_length, config.max_slots),
          slots(static_cast<std::size_t>(config.max_slots), -1),
          slot_ever_used(static_cast<std::size_t>(config.max_slots), false) {}

    model::TransformerModel& model;
    ContinuousBatchConfig config;
    std::vector<DType> policy;
    KVCache cache;
    Tensor slot_logits;
    std::vector<Request> requests;
    std::vector<std::int64_t> slots;
    std::vector<bool> slot_ever_used;
    RequestId next_id = 1;
    ContinuousBatchMetrics metrics;

    std::size_t find_index(RequestId id) const {
        const auto found = std::find_if(
            requests.begin(), requests.end(),
            [id](const Request& request) { return request.id == id; });
        if (found == requests.end()) {
            throw std::out_of_range("unknown continuous scheduler request");
        }
        return static_cast<std::size_t>(std::distance(requests.begin(), found));
    }

    Request& find(RequestId id) { return requests[find_index(id)]; }
    const Request& find(RequestId id) const { return requests[find_index(id)]; }

    void validate_policy(const GenerationConfig& generation) const {
        if (cache_policy(model, generation) != policy) {
            throw std::invalid_argument(
                "request KV cache policy does not match continuous scheduler");
        }
    }

    std::size_t row_capacity_bytes() const {
        return config.max_slots > 0
                   ? cache_bytes(cache) /
                         static_cast<std::size_t>(config.max_slots)
                   : 0U;
    }

    std::size_t active_prefix_bytes() const {
        std::size_t bytes = 0;
        std::int64_t active_tokens = 0;
        for (const auto position : cache.row_positions()) {
            active_tokens += position;
        }
        for (std::size_t layer = 0; layer < cache.layer_count(); ++layer) {
            const auto& state = cache.layer(layer);
            if (!state.key.defined()) continue;
            const auto per_tensor = static_cast<std::size_t>(
                active_tokens * state.key.shape()[1] * state.key.shape()[3]) *
                dtype_size(state.key.dtype());
            bytes += per_tensor * 2U;
        }
        return bytes;
    }

    void refresh_metrics() {
        const auto occupied = static_cast<std::int64_t>(std::count_if(
            slots.begin(), slots.end(),
            [](std::int64_t value) { return value >= 0; }));
        metrics.occupied_slots = occupied;
        metrics.peak_occupied_slots =
            std::max(metrics.peak_occupied_slots, occupied);
        metrics.allocated_cache_bytes = cache_bytes(cache);
        metrics.active_cache_bytes = active_prefix_bytes();
        metrics.peak_active_cache_bytes =
            std::max(metrics.peak_active_cache_bytes,
                     metrics.active_cache_bytes);
        if (metrics.scheduler_steps > 0) {
            metrics.slot_utilization =
                static_cast<double>(metrics.occupied_slot_steps) /
                (static_cast<double>(metrics.scheduler_steps) *
                 static_cast<double>(config.max_slots));
        }
    }

    void copy_logits_to_slot(const Tensor& logits, std::int64_t slot) {
        if (logits.dtype() != DType::Float32 || logits.ndim() != 3 ||
            logits.shape()[0] != 1 || logits.shape()[1] != 1 ||
            logits.shape()[2] != model.config().vocabulary_size ||
            slot < 0 || slot >= config.max_slots) {
            throw std::invalid_argument("continuous scheduler received incompatible logits");
        }
        if (!slot_logits.defined()) {
            slot_logits = Tensor(
                {config.max_slots, 1, model.config().vocabulary_size},
                DType::Float32, logits.device());
        }
        if (slot_logits.device() != logits.device()) {
            throw std::invalid_argument("continuous scheduler logits device changed");
        }
        runtime::copy_bytes(
            static_cast<float*>(slot_logits.data()) +
                slot * slot_logits.stride(0),
            slot_logits.device(), logits.data(), logits.device(),
            static_cast<std::size_t>(model.config().vocabulary_size) *
                sizeof(float));
    }

    void admit_pending() {
        for (std::int64_t slot = 0; slot < config.max_slots; ++slot) {
            if (slots[static_cast<std::size_t>(slot)] >= 0) continue;
            const auto found = std::find_if(
                requests.begin(), requests.end(), [](const Request& request) {
                    return request.state == RequestState::PendingPrefill;
                });
            if (found == requests.end()) break;
            const auto index = static_cast<std::int64_t>(
                std::distance(requests.begin(), found));
            auto& request = *found;
            try {
                const auto logits = model.forward_prefill_cached_row(
                    Tensor::from_int32_vector(
                        request.prompt,
                        {1, static_cast<std::int64_t>(request.prompt.size())}),
                    cache, slot);
                copy_logits_to_slot(logits, slot);
            } catch (...) {
                cache.reset_row(slot);
                throw;
            }
            if (slot_ever_used[static_cast<std::size_t>(slot)]) {
                ++metrics.slot_refills;
            }
            slot_ever_used[static_cast<std::size_t>(slot)] = true;
            slots[static_cast<std::size_t>(slot)] = index;
            request.slot = slot;
            request.state = RequestState::Decoding;
            ++metrics.slot_admissions;
            ++metrics.row_prefill_calls;
        }
    }

    std::vector<std::int32_t> select_tokens() {
        if (!slot_logits.defined()) {
            throw std::logic_error("continuous scheduler has no slot logits");
        }
        ++metrics.selection_calls;
        const auto all_greedy = std::all_of(
            slots.begin(), slots.end(), [this](std::int64_t index) {
                if (index < 0) return true;
                const auto& generation =
                    requests[static_cast<std::size_t>(index)].config;
                return generation.temperature == 0.0F || generation.top_k == 1;
            });
        std::vector<std::int32_t> selected(
            static_cast<std::size_t>(config.max_slots), -1);
        if (all_greedy) {
            selected = ops::argmax_last_dim(slot_logits).to_int32_vector();
            if (selected.size() != static_cast<std::size_t>(config.max_slots)) {
                throw std::logic_error("continuous scheduler argmax row count changed");
            }
            return selected;
        }
        for (std::int64_t slot = 0; slot < config.max_slots; ++slot) {
            const auto index = slots[static_cast<std::size_t>(slot)];
            if (index < 0) continue;
            auto& request = requests[static_cast<std::size_t>(index)];
            const auto row = slot_logits.slice(0, slot, slot + 1).to_vector();
            selected[static_cast<std::size_t>(slot)] = sample_token(
                row, request.config.temperature, request.config.top_k,
                request.random);
        }
        return selected;
    }

    void release_slot(Request& request) {
        if (request.slot < 0) return;
        const auto slot = request.slot;
        cache.reset_row(slot);
        slots[static_cast<std::size_t>(slot)] = -1;
        request.slot = -1;
    }

    void complete(Request& request, CompletionReason reason) {
        request.state = RequestState::Completed;
        request.completion_reason = reason;
        request.completion_step = metrics.scheduler_steps;
        ++metrics.completed_requests;
        if (reason == CompletionReason::StopToken) {
            ++metrics.stop_completed_requests;
        }
        release_slot(request);
    }
};

ContinuousBatchScheduler::ContinuousBatchScheduler(
    model::TransformerModel& model, ContinuousBatchConfig config)
    : impl_(std::make_unique<Impl>(model, std::move(config))) {}
ContinuousBatchScheduler::~ContinuousBatchScheduler() = default;
ContinuousBatchScheduler::ContinuousBatchScheduler(
    ContinuousBatchScheduler&&) noexcept = default;
ContinuousBatchScheduler& ContinuousBatchScheduler::operator=(
    ContinuousBatchScheduler&&) noexcept = default;

RequestId ContinuousBatchScheduler::submit(
    std::vector<std::int32_t> prompt, GenerationConfig config) {
    validate_request(impl_->model, prompt, config);
    std::sort(config.stop_tokens.begin(), config.stop_tokens.end());
    impl_->validate_policy(config);
    Impl::Request request;
    request.id = impl_->next_id++;
    request.prompt = std::move(prompt);
    request.random = std::mt19937_64(config.seed);
    request.config = std::move(config);
    request.arrival_step = impl_->metrics.scheduler_steps;
    if (request.config.max_new_tokens == 0) {
        request.state = RequestState::Completed;
        request.completion_reason = CompletionReason::Length;
        request.completion_step = impl_->metrics.scheduler_steps;
        ++impl_->metrics.completed_requests;
    }
    const auto id = request.id;
    impl_->requests.push_back(std::move(request));
    ++impl_->metrics.submitted_requests;
    impl_->refresh_metrics();
    return id;
}

bool ContinuousBatchScheduler::cancel(RequestId id) {
    auto& request = impl_->find(id);
    if (is_terminal(request.state)) return false;
    impl_->release_slot(request);
    request.state = RequestState::Cancelled;
    request.completion_reason = CompletionReason::Cancelled;
    request.completion_step = impl_->metrics.scheduler_steps;
    ++impl_->metrics.cancelled_requests;
    impl_->refresh_metrics();
    return true;
}

void ContinuousBatchScheduler::step() {
    if (!has_active_requests()) return;
    ++impl_->metrics.scheduler_steps;
    impl_->admit_pending();
    impl_->refresh_metrics();
    const auto occupied = impl_->metrics.occupied_slots;
    if (occupied == 0) return;
    impl_->metrics.occupied_slot_steps += occupied;
    const auto selected = impl_->select_tokens();
    std::vector<std::int32_t> next_tokens(
        static_cast<std::size_t>(impl_->config.max_slots), 0);
    std::vector<bool> survivors(
        static_cast<std::size_t>(impl_->config.max_slots), false);
    for (std::int64_t slot = 0; slot < impl_->config.max_slots; ++slot) {
        const auto index = impl_->slots[static_cast<std::size_t>(slot)];
        if (index < 0) continue;
        auto& request = impl_->requests[static_cast<std::size_t>(index)];
        const auto token = selected[static_cast<std::size_t>(slot)];
        if (token < 0) {
            throw std::invalid_argument("continuous scheduler logits are non-finite");
        }
        request.generated.push_back(token);
        if (is_stop_token(request.config, token)) {
            impl_->complete(request, CompletionReason::StopToken);
        } else if (static_cast<std::int64_t>(request.generated.size()) ==
                   request.config.max_new_tokens) {
            impl_->complete(request, CompletionReason::Length);
        } else {
            survivors[static_cast<std::size_t>(slot)] = true;
            next_tokens[static_cast<std::size_t>(slot)] = token;
        }
    }
    const auto survivor_count = static_cast<std::int64_t>(std::count(
        survivors.begin(), survivors.end(), true));
    if (survivor_count > 0) {
        std::vector<std::int64_t> active_rows;
        std::vector<std::int32_t> active_tokens;
        active_rows.reserve(static_cast<std::size_t>(survivor_count));
        active_tokens.reserve(static_cast<std::size_t>(survivor_count));
        for (std::int64_t slot = 0; slot < impl_->config.max_slots; ++slot) {
            if (!survivors[static_cast<std::size_t>(slot)]) continue;
            active_rows.push_back(slot);
            active_tokens.push_back(next_tokens[static_cast<std::size_t>(slot)]);
        }
        const auto active_positions_uniform = std::all_of(
            active_rows.begin() + 1, active_rows.end(),
            [&impl = *impl_, first = impl_->cache.row_position(active_rows.front())](
                std::int64_t row) {
                return impl.cache.row_position(row) == first;
            });
        const auto full_uniform = survivor_count == impl_->config.max_slots &&
                                  active_positions_uniform;
        if (full_uniform) {
            impl_->slot_logits = impl_->model.forward_cached_rows(
                Tensor::from_int32_vector(
                    next_tokens, {impl_->config.max_slots, 1}),
                impl_->cache);
            ++impl_->metrics.uniform_batch_decode_calls;
        } else {
            const auto active_logits = impl_->model.forward_cached_active_rows(
                Tensor::from_int32_vector(
                    active_tokens, {survivor_count, 1}),
                impl_->cache, active_rows);
            for (std::size_t index = 0; index < active_rows.size(); ++index) {
                impl_->copy_logits_to_slot(
                    active_logits.slice(
                        0, static_cast<std::int64_t>(index),
                        static_cast<std::int64_t>(index + 1)),
                    active_rows[index]);
            }
            ++impl_->metrics.compacted_batch_decode_calls;
            if (!active_positions_uniform) {
                ++impl_->metrics.divergent_batch_decode_calls;
            }
        }
        ++impl_->metrics.batch_decode_calls;
        impl_->metrics.logical_decode_rows += survivor_count;
        impl_->metrics.inactive_rows_skipped +=
            impl_->config.max_slots - survivor_count;
    }
    impl_->refresh_metrics();
}

void ContinuousBatchScheduler::run_until_idle(std::int64_t maximum_steps) {
    if (maximum_steps < -1) throw std::invalid_argument("maximum_steps is invalid");
    std::int64_t executed = 0;
    while (has_active_requests() &&
           (maximum_steps < 0 || executed < maximum_steps)) {
        step();
        ++executed;
    }
    if (has_active_requests()) {
        throw std::runtime_error(
            "continuous scheduler did not become idle within maximum_steps");
    }
}

bool ContinuousBatchScheduler::has_active_requests() const noexcept {
    return std::any_of(impl_->requests.begin(), impl_->requests.end(),
                       [](const Impl::Request& request) {
                           return !is_terminal(request.state);
                       });
}

std::size_t ContinuousBatchScheduler::active_request_count() const noexcept {
    return static_cast<std::size_t>(std::count_if(
        impl_->requests.begin(), impl_->requests.end(),
        [](const Impl::Request& request) {
            return !is_terminal(request.state);
        }));
}

std::size_t ContinuousBatchScheduler::pending_request_count() const noexcept {
    return static_cast<std::size_t>(std::count_if(
        impl_->requests.begin(), impl_->requests.end(),
        [](const Impl::Request& request) {
            return request.state == RequestState::PendingPrefill;
        }));
}

RequestSnapshot ContinuousBatchScheduler::request(RequestId id) const {
    const auto& request = impl_->find(id);
    return {.id = request.id,
            .state = request.state,
            .completion_reason = request.completion_reason,
            .prompt = request.prompt,
            .generated = request.generated,
            .max_new_tokens = request.config.max_new_tokens,
            .arrival_step = request.arrival_step,
            .completion_step = request.completion_step,
            .cache_bytes = request.slot >= 0 ? impl_->row_capacity_bytes() : 0U,
            .slot = request.slot};
}

std::vector<RequestSnapshot> ContinuousBatchScheduler::requests() const {
    std::vector<RequestSnapshot> result;
    result.reserve(impl_->requests.size());
    for (const auto& request : impl_->requests) {
        result.push_back(this->request(request.id));
    }
    return result;
}

ContinuousBatchMetrics ContinuousBatchScheduler::metrics() const noexcept {
    return impl_->metrics;
}

}  // namespace microllm::inference
