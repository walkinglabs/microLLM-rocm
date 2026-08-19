# 2026-08-19 — branch verification summary

Branch: `feat/bootstrap-engine-m0-n0`

## Passing gates

| Gate | Result |
|---|---:|
| normal CPU evidence | 99/99 |
| CPU ASan/UBSan (dynamic bindings excluded) | 97/97 |
| MI300X/gfx942 HIP label | 23/23 |
| two-rank RCCL label | 7/7 |
| PyTorch 2.13 CPU Custom Op and correctness oracle | 2/2 |
| committed JSON/JSONL parser | all records valid |
| N0–N8 and PA0–PA2 artifact presence | pass |

The sanitized build excludes C/Python dynamic binding targets because loading an
ASan-instrumented shared library into a non-ASan C/Python process requires runtime
preload ordering. Those bindings pass separate normal CPU and HIP integration tests.

## Measured framework milestones

- Model-S: 15,586,176 parameters; CPU/HIP forward, CPU 3-step training, 10-step
  TinyStories HIP text smoke;
- Model-M: 31,334,912 parameters; one complete MI300X train step, 518,798,856 peak
  engine-owned HIP bytes;
- tiny SFT response-only loss: 1.88494 to 0.0106737;
- full tiny GQA Transformer forward/backward graph: CPU/HIP loss and every parameter
  gradient agree, with zero host/device transfers during graph execution;
- PyTorch CPU oracle: every public math operator, backward family, valid output shape,
  24 invalid shape/dtype contracts, SGD, two-step AdamW moments, and the full tiny GQA
  Transformer logits/loss/all-parameter gradients agree;
- coverage audit: all 30 Tensor APIs, 29 graph/Value APIs, and 25 discovered test files
  have explicit gates and CMake/CTest registration;
- Model-S measured generation: CPU 9.33, readable HIP 55.86, Auto hipBLASLt 187.10
  tokens/s for the recorded one-token experiment;
- two-rank parameters identical; single/two-rank maximum difference 1.49012e-08;
- 1MB two-rank all-reduce: 6.676ms at 64 buckets versus 0.22454ms at one bucket;
- synthetic compute/RCCL overlap improvement: 30–33% across three runs.

## Explicitly incomplete or externally blocked

- full TinyStories train/validation reference curve and checkpoints;
- Model-S instruction-corpus SFT quality report;
- PyTorch ROCm Custom Op: matching wheel fails with Bus error on import;
- Radeon hardware validation: no Radeon device available;
- four-rank RCCL: current 64MB `/dev/shm` causes shared-memory ENOSPC;
- backward-ready bucket overlap: current overlap experiment uses independent compute;
- comprehensive PyTorch/llama.cpp performance comparison;
- BF16/FP16 kernels and mixed-precision optimizer state.

These items remain outside release claims even though their interfaces or workflows
may exist. The repository is a tested development branch, not yet a full reference-
trained or Radeon-released project.
