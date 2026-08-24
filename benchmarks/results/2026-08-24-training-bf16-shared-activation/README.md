# Training BF16 shared-activation evidence

Experiment 212 shares one FP32-to-BF16 activation cast across Q/K/V or gate/up forward
projections while leaving each backward edge independent.

## Evidence layout

- `combined-pilot-*`: both boundaries, three fresh processes per model/policy;
- `combined-formal-*`: both boundaries, five fresh processes per model/policy;
- `qkv-*`: QKV-only three-process isolation and diagnostics;
- `gate-up-*`: gate/up-only three-process isolation and diagnostics;
- `*-kernel-stats.csv` and `profile-summary.json`: two-model structural profile;
- `summary.json`: compact comparison of all three policies;
- `coverage-summary.json` and `verification.json`: final regression evidence.

## Result

| Policy | Qwen | DeepSeek | Decision |
|---|---:|---:|---|
| QKV + gate/up, five runs | 1.0066× | 1.0179× | reject |
| QKV only, three runs | 0.9804× | 1.0039× | reject |
| gate/up only, three runs | 0.9911× | 1.0012× | reject |

The complete policy removes exactly 216/252 cast launches from the profiled three-step
Qwen/DeepSeek workloads. Total Kernel time improves only 1.0116×/1.0095×, and the official
model gate fails. Model, optimizer and CLI routes are absent from retained source. The tensor
and Autograd primitives remain because CPU, HIP and PyTorch cover every output and gradient.
