# Current training local-saturation audit

Experiment 250 combines the current retained B1T512 profile with six adjacent
closed training optimization tracks.

| Model | Kernel/step | GEMM | AdamW | Cast | Free cast-deletion ceiling |
|---|---:|---:|---:|---:|---:|
| Qwen2.5-0.5B | 31.327 ms | 58.56% | 13.22% | 3.21% | 1.0332× |
| DeepSeek Distill 1.5B | 71.873 ms | 63.43% | 18.16% | 2.70% | 1.0277× |

Grouped weight gradient, packed weight gradient, exact solution indices,
optimizer-only Graph routing, BF16 gate/up long trajectory and its workspace
follow-up all have measured rejection gates. The largest remaining non-GEMM,
non-AdamW category is still below a 1.051×/1.039× perfect-deletion ceiling.

This closes local default-policy retuning, not training optimization. The next
work must introduce a different custom-kernel/graph-wide architecture or move to
the production data-parallel reducer. It must not reopen a closed local route
without a new backend, hardware matrix or rebuttal contract.

