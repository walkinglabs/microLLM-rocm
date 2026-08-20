# Experiment 012 repeated argmax evidence

| File | Meaning |
|---|---|
| `candidate-run-{1,2,3}.jsonl` | three two-stage argmax inference processes |
| `median-summary.json` | running-best versus candidate medians and score |
| `post-candidate-baseline-infer.jsonl` | unmodified inference after candidates |
| `after-*-stats.csv` | two-stage Qwen decode profiler tables |

The direct profiler before is Experiment 009. Argmax Kernel time falls 96.7%; robust
Qwen/DeepSeek generation medians improve 3.6%/0.6%.
