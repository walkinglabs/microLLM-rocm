# Experiment 009 repeated and context evidence

| File | Meaning |
|---|---|
| `candidate-run-{1,2,3}.jsonl` | three fused candidate process runs |
| `median-summary.json` | baseline/candidate medians and scored PyTorch ratios |
| `post-candidate-baseline-infer.jsonl` | unmodified inference after candidates |
| `baseline-context-curve.jsonl` | interleaved Qwen 1/32/128/512 baseline points |
| `candidate-context-curve.jsonl` | matching fused points |
| `after-*-stats.csv` | fused Qwen decode rocprof compact tables |

Baseline three-process source is Experiment 008's repeated evidence and Experiment
006's final raw row. The direct profiler before is Experiment 006's `after-*` data.

The one-token context point regresses 7.8%; 32/128/512 improve 18.5%/18.5%/57.9%.
This failure is deliberately retained alongside the kept median result.
