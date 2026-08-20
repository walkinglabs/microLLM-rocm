# 2026-08-20 — dedicated 0→1 optimization journal

## Goal

Create a separate, living record for the performance campaign from the measured FP32
baseline to selected-matrix PyTorch parity. Normal chronological development records
describe what landed; this journal must also preserve experiments that were discarded,
crashed or invalidated.

The experiment-loop presentation is inspired by
[karpathy/autoresearch](https://github.com/karpathy/autoresearch): fixed comparison,
one primary variable, keep/discard decisions, an append-only result table and a
running-best chart. The metrics, scripts, SVGs and content are repository-owned.

## Delivered structure

```text
docs/optimization-log/
  README.md
  BLOG.zh-CN.md
  PROGRAM.md
  PLAN.md
  SCHEMA.md
  results.tsv
  steps/00..12
  experiments/TEMPLATE.md
  scripts/render_progress.py
  scripts/validate_log.py
  assets/progress.svg
  assets/bottleneck-map.svg
```

## Baseline

```text
Qwen train ratio             0.142235
Qwen generate ratio          0.267461
DeepSeek train ratio         0.220921
DeepSeek generate ratio      0.160553
geometric score              0.191660
selected-matrix target       1.000000
```

## Evidence rules

- planned work is kept in `steps/`, never inserted into measured results;
- `results.tsv` contains baseline/keep/discard/crash/invalid rows;
- score is recomputed from four ratios by the validator;
- local Markdown links and SVG XML are checked;
- SVG assets must exactly match regenerated content;
- low-precision tracks cannot extend the FP32 running-best line;
- a rounded blog table never replaces raw JSONL/profiler evidence.

## Current state

Only experiment 000 exists. The progress figure deliberately shows one baseline point
and a large gap to parity; planned roadmap boxes are labeled as plans, not results.
