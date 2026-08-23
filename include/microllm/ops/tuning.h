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

}  // namespace microllm::ops
