# Experiment 041 evidence

`raw.jsonl` contains one same-window Experiment 040 control and three completed Qwen island
processes. `early-stop.json` records why DeepSeek was interrupted. `profile-summary.json`
and the two CSV files are aggregates from a one-step `rocprofv3` trace; the large raw trace
is intentionally not committed.

The candidate was compiled and tested, then deleted after the gate failed. Its contract was:

```text
FP32 input → one BF16 cast → BF16 gate/up → BF16 SwiGLU → BF16 down input → FP32 output
backward → FP32 input and FP32 master-weight gradients
```

The crucial comparison is the same-window `18.892/18.685 = 1.011×`, not the invalid ratio
against the earlier `151.69 token/s` publication. Shared-resource drift is evidence about
measurement validity, not evidence for or against the candidate.
