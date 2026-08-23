#include <microllm/ops/tuning.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <set>
#include <stdexcept>
#include <utility>

#include <microllm/runtime/memory.h>
#include <microllm/runtime/runtime.h>

namespace microllm::ops {
namespace {

std::pair<float, double> default_tolerances(DType dtype) {
    switch (dtype) {
        case DType::Float32: return {2.0e-4F, 5.0e-5};
        case DType::Float16: return {3.0e-2F, 1.0e-2};
        case DType::BFloat16: return {2.0e-1F, 5.0e-2};
        default:
            throw std::invalid_argument(
                "matmul autotune supports FP32, FP16 and BF16 operands");
    }
}

void validate_options(const MatmulAutotuneOptions& options) {
    if (options.warmup < 0 || options.repetitions <= 0 ||
        !std::isfinite(options.maximum_absolute_tolerance) ||
        !std::isfinite(options.rms_tolerance) || options.candidates.empty()) {
        throw std::invalid_argument("matmul autotune options are invalid");
    }
    std::set<MatmulImplementation> unique;
    for (const auto candidate : options.candidates) {
        if (candidate == MatmulImplementation::Auto ||
            !unique.insert(candidate).second) {
            throw std::invalid_argument(
                "matmul autotune candidates must be unique concrete implementations");
        }
    }
}

std::pair<float, double> compare_complete(
    const std::vector<float>& actual, const std::vector<float>& reference,
    bool& finite) {
    if (actual.size() != reference.size() || actual.empty()) {
        throw std::runtime_error("matmul autotune output shape changed");
    }
    float maximum = 0.0F;
    double squared = 0.0;
    finite = true;
    for (std::size_t index = 0; index < actual.size(); ++index) {
        if (!std::isfinite(actual[index]) || !std::isfinite(reference[index])) {
            finite = false;
            continue;
        }
        const auto difference = std::abs(actual[index] - reference[index]);
        maximum = std::max(maximum, difference);
        squared += static_cast<double>(difference) * difference;
    }
    return {maximum, std::sqrt(squared / static_cast<double>(actual.size()))};
}

double percentile(std::vector<double> values, double fraction) {
    if (values.empty()) throw std::invalid_argument("cannot summarize empty timings");
    std::sort(values.begin(), values.end());
    const auto index = static_cast<std::size_t>(std::ceil(
        fraction * static_cast<double>(values.size()))) - 1U;
    return values[std::min(index, values.size() - 1U)];
}

Tensor run_candidate(const Tensor& left, const Tensor& right,
                     MatmulImplementation implementation,
                     bool transpose_left, bool transpose_right,
                     const OpContext& context) {
    return matmul_with_implementation(
        left, right, implementation, transpose_left, transpose_right, context);
}

}  // namespace

MatmulAutotuneReport autotune_matmul(
    const Tensor& left, const Tensor& right,
    bool transpose_left, bool transpose_right,
    const MatmulAutotuneOptions& options) {
    validate_options(options);
    if (!left.device().is_hip() || right.device() != left.device()) {
        throw std::invalid_argument("matmul autotune requires operands on one HIP device");
    }
    const auto automatic = default_tolerances(left.dtype());
    const auto maximum_tolerance = options.maximum_absolute_tolerance < 0.0F
                                       ? automatic.first
                                       : options.maximum_absolute_tolerance;
    const auto rms_tolerance = options.rms_tolerance < 0.0F
                                   ? automatic.second
                                   : static_cast<double>(options.rms_tolerance);
    runtime::synchronize(left.device());
    runtime::Event start(left.device());
    runtime::Event finish(left.device());
    Storage workspace(options.workspace_limit, left.device());
    OpContext context;
    context.workspace = workspace.data();
    context.workspace_bytes = workspace.num_bytes();
    context.mode = options.mode;

    const auto reference_tensor = run_candidate(
        left, right, MatmulImplementation::Readable,
        transpose_left, transpose_right, context);
    runtime::synchronize(left.device());
    const auto reference = reference_tensor.to_vector();
    MatmulAutotuneReport report;
    report.key = make_matmul_tuning_key(
        left, right, transpose_left, transpose_right, context);
    report.reference_elements = static_cast<std::int64_t>(reference.size());
    report.maximum_absolute_tolerance = maximum_tolerance;
    report.rms_tolerance = rms_tolerance;

    for (const auto implementation : options.candidates) {
        MatmulAutotuneCandidate candidate;
        candidate.implementation = implementation;
        try {
            if (implementation == MatmulImplementation::HipBLASLt &&
                !hipblaslt_available()) {
                throw std::runtime_error("hipBLASLt is unavailable");
            }
            const auto checked = run_candidate(
                left, right, implementation,
                transpose_left, transpose_right, context);
            runtime::synchronize(left.device());
            const auto actual = checked.to_vector();
            const auto error = compare_complete(actual, reference, candidate.finite);
            candidate.maximum_absolute_error = error.first;
            candidate.rms_error = error.second;
            candidate.supported = true;
            candidate.correctness_passed = candidate.finite &&
                candidate.maximum_absolute_error <= maximum_tolerance &&
                candidate.rms_error <= rms_tolerance;
            if (!candidate.correctness_passed) {
                candidate.failure = "complete-output correctness gate failed";
                report.candidates.push_back(std::move(candidate));
                continue;
            }
            Tensor output;
            for (int iteration = 0; iteration < options.warmup; ++iteration) {
                output = run_candidate(
                    left, right, implementation,
                    transpose_left, transpose_right, context);
            }
            runtime::synchronize(left.device());
            std::vector<double> event_times;
            std::vector<double> wall_times;
            event_times.reserve(static_cast<std::size_t>(options.repetitions));
            wall_times.reserve(static_cast<std::size_t>(options.repetitions));
            for (int iteration = 0; iteration < options.repetitions; ++iteration) {
                const auto wall_start = std::chrono::steady_clock::now();
                start.record_default_stream();
                output = run_candidate(
                    left, right, implementation,
                    transpose_left, transpose_right, context);
                finish.record_default_stream();
                finish.synchronize();
                const auto wall_finish = std::chrono::steady_clock::now();
                event_times.push_back(finish.elapsed_ms_since(start));
                wall_times.push_back(std::chrono::duration<double, std::milli>(
                    wall_finish - wall_start).count());
            }
            candidate.event_ms_p50 = percentile(event_times, 0.50);
            candidate.event_ms_p95 = percentile(event_times, 0.95);
            candidate.wall_ms_p50 = percentile(wall_times, 0.50);
            candidate.wall_ms_p95 = percentile(wall_times, 0.95);
        } catch (const std::exception& error) {
            candidate.failure = error.what();
        }
        report.candidates.push_back(std::move(candidate));
    }

    const MatmulAutotuneCandidate* best = nullptr;
    for (const auto& candidate : report.candidates) {
        if (!candidate.supported || !candidate.correctness_passed ||
            !(candidate.event_ms_p50 > 0.0) ||
            !(candidate.event_ms_p95 > 0.0)) continue;
        if (best == nullptr ||
            std::pair(candidate.event_ms_p50, candidate.event_ms_p95) <
                std::pair(best->event_ms_p50, best->event_ms_p95)) {
            best = &candidate;
        }
    }
    if (best == nullptr) {
        throw std::runtime_error("no matmul candidate passed correctness");
    }
    report.recommended = best->implementation;
    return report;
}

void register_matmul_autotune_winner(const MatmulAutotuneReport& report) {
    if (report.recommended == MatmulImplementation::Auto) {
        throw std::invalid_argument("matmul autotune report has no recommendation");
    }
    const auto found = std::find_if(
        report.candidates.begin(), report.candidates.end(),
        [&](const auto& candidate) {
            return candidate.implementation == report.recommended;
        });
    if (found == report.candidates.end() || !found->supported ||
        !found->correctness_passed || !found->finite ||
        !(found->event_ms_p50 > 0.0) || !(found->event_ms_p95 > 0.0)) {
        throw std::invalid_argument(
            "matmul autotune recommendation lacks correctness and timing evidence");
    }
    register_matmul_implementation(report.key, report.recommended);
}

}  // namespace microllm::ops
