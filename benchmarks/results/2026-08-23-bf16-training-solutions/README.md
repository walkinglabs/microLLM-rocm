# BF16 training solution-index screen

Experiment 160 screens hipBLASLt solution indices for the eight unique BF16 Linear
forward shapes in Qwen2.5-0.5B and DeepSeek Distill Qwen 1.5B at T512.

- `raw.jsonl`: eight shapes × three fresh tuner processes; each process screens 64
  heuristic solutions with complete output before Event/wall timing;
- `summary.json`: common-passing candidates and cross-process median winners;
- `training-baseline.jsonl`: no explicit solution registrations;
- `training-all-shapes.jsonl`: four exact registrations per model;
- `training-selective.jsonl`: gate/up registration removed;
- `training-comparison.json`: same-revision three-process model decision;
- `verification.json`: machine-readable rejection contract.

All 1,536 candidate evaluations are finite and pass complete output. Operator medians
improve 1.031×–1.189×, but only Qwen down chooses the same best index in all three
processes. Neither all-shape nor selective registration reaches the 1.05 end-to-end gate
on both models, so no solution is installed as a default or persisted.

The CLI registration is an explicit research seam:

```bash
./build/hip-release/apps/microllm_hf_train_step ... \
  --bf16-algorithms 512:896:896:98676,512:896:128:98590
```

It is process-local and not an environment-safe package policy.
