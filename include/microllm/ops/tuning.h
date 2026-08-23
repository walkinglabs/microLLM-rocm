#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include <microllm/ops/ops.h>

namespace microllm::ops {

struct MatmulAutotuneOptions {
    int warmup = 3;
    int repetitions = 10;
    // Negative values select dtype-aware defaults. Zero requests bit-exact output.
    float maximum_absolute_tolerance = -1.0F;
    float rms_tolerance = -1.0F;
    std::size_t workspace_limit = 0;
    OpMode mode = OpMode::Unspecified;
    std::vector<MatmulImplementation> candidates{
        MatmulImplementation::Readable, MatmulImplementation::HipBLASLt};
};

struct MatmulAutotuneCandidate {
    MatmulImplementation implementation = MatmulImplementation::Auto;
    bool supported = false;
    bool correctness_passed = false;
    bool finite = false;
    float maximum_absolute_error = 0.0F;
    double rms_error = 0.0;
    double event_ms_p50 = 0.0;
    double event_ms_p95 = 0.0;
    double wall_ms_p50 = 0.0;
    double wall_ms_p95 = 0.0;
    std::string failure;
};

struct MatmulAutotuneReport {
    MatmulTuningKey key;
    std::int64_t reference_elements = 0;
    float maximum_absolute_tolerance = 0.0F;
    double rms_tolerance = 0.0;
    std::vector<MatmulAutotuneCandidate> candidates;
    MatmulImplementation recommended = MatmulImplementation::Auto;
};

// Does not mutate the live registry. A caller must run its end-to-end regression
// and explicitly accept the report with register_matmul_autotune_winner().
[[nodiscard]] MatmulAutotuneReport autotune_matmul(
    const Tensor& left, const Tensor& right,
    bool transpose_left = false, bool transpose_right = false,
    const MatmulAutotuneOptions& options = {});

void register_matmul_autotune_winner(const MatmulAutotuneReport& report);

struct AdamWAutotuneOptions {
    int warmup = 3;
    int repetitions = 10;
    // Negative values select FP32 defaults. Zero requests bit-exact state.
    float maximum_absolute_tolerance = -1.0F;
    float rms_tolerance = -1.0F;
    float learning_rate = 0.01F;
    float beta1 = 0.9F;
    float beta2 = 0.99F;
    float epsilon = 1.0e-8F;
    float weight_decay = 0.1F;
    float first_correction = 0.1F;
    float second_correction = 0.01F;
    OpMode mode = OpMode::Unspecified;
    std::vector<AdamWImplementation> candidates{
        AdamWImplementation::Scalar, AdamWImplementation::Vectorized};
};

struct AdamWStateError {
    float maximum_absolute_error = 0.0F;
    double rms_error = 0.0;
};

struct AdamWAutotuneCandidate {
    AdamWImplementation implementation = AdamWImplementation::Auto;
    bool supported = false;
    bool correctness_passed = false;
    bool finite = false;
    AdamWStateError parameter;
    AdamWStateError first_moment;
    AdamWStateError second_moment;
    AdamWStateError bf16_mirror;
    double event_ms_p50 = 0.0;
    double event_ms_p95 = 0.0;
    double wall_ms_p50 = 0.0;
    double wall_ms_p95 = 0.0;
    std::string failure;
};

struct AdamWAutotuneReport {
    AdamWTuningKey key;
    std::int64_t reference_elements = 0;
    float maximum_absolute_tolerance = 0.0F;
    double rms_tolerance = 0.0;
    std::vector<AdamWAutotuneCandidate> candidates;
    AdamWImplementation recommended = AdamWImplementation::Auto;
};

// Does not mutate the caller's state or the live registry. The caller must run
// its end-to-end regression and explicitly accept a result.
[[nodiscard]] AdamWAutotuneReport autotune_adamw(
    const Tensor& parameter, const Tensor& gradient,
    const Tensor& first_moment, const Tensor& second_moment,
    const Tensor* bf16_mirror = nullptr,
    const AdamWAutotuneOptions& options = {});

void register_adamw_autotune_winner(const AdamWAutotuneReport& report);

}  // namespace microllm::ops
