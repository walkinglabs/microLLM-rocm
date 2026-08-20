# Experiment 008 — workspace-aware hipBLASLt algorithm cache

Status: `discard`

## Why this differs from discarded Experiment 007

Experiment 007 retained host descriptor/layout objects and regressed. This experiment
does not retain those objects. hipBLASLt documents `hipblasLtMatmulAlgo_t` as trivially
serializable and says `algo=null` performs an implicit heuristic query. We cache only
that algorithm value after one explicit query.

## Hypothesis

Exact-shape algorithm reuse avoids repeated implicit heuristic selection while keeping
the existing descriptor lifetime. Repeated short projection GEMMs should improve if
implicit selection is material.

## Scope

- key: dtype, physical shapes, transpose flags and available workspace bytes;
- one requested heuristic, matching the current default estimated-best behavior;
- unavailable heuristic is cached as fallback and continues with `algo=null`;
- FP8 dynamic-scale path unchanged;
- no descriptor cache, Kernel change, model change or workload change.

## Required gates

- [x] identical key miss then hit
- [x] workspace and transpose keys remain distinct
- [x] unavailable path is explicit and resettable
- [x] FP32/FP16 and NN/NT/TN/TT numerical gates
- [x] exact external model tokens/loss/update
- [x] three-process baseline/candidate fixed matrix

## Candidate

- local descriptors/layouts kept their old lifetime;
- on an exact-key miss, one workspace-limited heuristic was requested;
- only the trivially serializable `hipblasLtMatmulAlgo_t` was retained;
- cached fallback used `algo=null` when no heuristic was available;
- focused tests proved hit/miss, transpose, dtype, workspace and reset behavior.

## Why one process was not enough

The first candidate process scored `1.682214`, below the historical `1.700597` best.
However, a contemporary unmodified run showed Qwen generation at 118.24 token/s while
the candidate showed 126.93. That contradicted a simple “candidate is slower” reading.

The experiment was expanded to three independent processes for baseline and candidate;
each process still used two warm-ups and five measured iterations.

## Three-process medians

| Workload | Baseline samples | Baseline median | Candidate samples | Candidate median | Change |
|---|---|---:|---|---:|---:|
| Qwen train | 107.08 / 104.99 / 110.75 | 107.08 | 112.91 / 106.29 / 107.39 | 107.39 | +0.3% |
| Qwen generate | 134.87 / 118.24 / 134.96 | 134.87 | 126.93 / 122.58 / 122.58 | 122.58 | -9.1% |
| DeepSeek train | 69.77 / 68.77 / 68.41 | 68.77 | 67.31 / 69.02 / 65.99 | 67.31 | -2.1% |
| DeepSeek generate | 48.93 / 49.05 / 51.37 | 49.05 | 48.93 / 51.29 / 48.94 | 48.94 | -0.2% |

```text
baseline median score   1.695566
candidate median score  1.646877
relative change            -2.9%
```

Qwen generation is a clear median regression. The candidate source was removed and the
framework restored exactly to the Experiment 006 code.

## Measurement protocol finding

The result changes interpretation depending on which single baseline process is chosen.
From this point, candidates whose claimed gain is below 10% require at least three
independent process repeats and use the per-workload median. Historical rows remain
untouched; the stronger rule applies prospectively.

## Evidence

All six raw runs and the generated median summary are in
[008-data](008-data/README.md). Baseline run 1 is Experiment 006's committed final raw
file; it is linked rather than duplicated.

## Results

Falsified. Avoiding implicit heuristic query did not improve the repeated-process
end-to-end objective.

## Decision

`discard`. No algorithm cache remains in framework source.
