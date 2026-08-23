# Paired GQA K/V repeat evidence

Experiment 168 maps K[B,KV,T,D] and V[B,T,KV,D] to repeated H heads in one HIP
Kernel, and reduces their two gradient layouts in one Kernel. Separate Storage/output
contracts are unchanged.

- `training.jsonl` / `summary.json`: separate/paired × Qwen/DeepSeek × three fresh T512
  processes;
- `*-diagnostics.json`: both policies retain zero strided-copy state;
- `separate-kernel-stats.csv` / `paired-kernel-stats.csv`: complete Qwen rocprofv3 data;
- `profile-summary.json`: repeat and total-Kernel attribution;
- `coverage-summary.json`: post-change CPU coverage;
- `verification.json`: default-off decision and regression evidence.

The profiler mechanism improves: repeat-family calls fall `432→216`, their time falls
`2.105→1.330 ms`, dispatches fall `6,907→6,689`, and total Kernel time falls 1.18%.
The full model rejects it: Qwen is `0.9758×` and DeepSeek `1.0084×`, with no peak or
allocation benefit. Paired operators remain explicit/tested; production and CLI defaults
are false.
