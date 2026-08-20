# Experiment 008 repeated-process evidence

| File | Meaning |
|---|---|
| `candidate-run-{1,2,3}.jsonl` | three algorithm-cache process runs |
| `baseline-run-{2,3}.jsonl` | two contemporary unmodified process runs |
| `candidate-run-1-comparison.jsonl` | first candidate versus fixed PyTorch data |
| `median-summary.json` | generated per-workload samples, medians and A/B ratios |

Baseline run 1 is Experiment 006's [`microllm.jsonl`](../006-data/microllm.jsonl).

The Qwen generation median regressed 9.1%; candidate median score was `1.646877` versus
baseline median `1.695566`. Candidate code was removed.
