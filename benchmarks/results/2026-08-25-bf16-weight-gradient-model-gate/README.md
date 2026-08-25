# Gate/up-only BF16 weight-gradient model gate

Experiment 246 runs one current binary with only
`--bf16-gate-up-weight-gradient` changed. Each Qwen/DeepSeek policy has three
fresh processes; order reverses every second pair.

| Model | Baseline tok/s | Candidate tok/s | Speedup | Peak ratio | Routed dW |
|---|---:|---:|---:|---:|---:|
| Qwen2.5-0.5B | 15,631.52 | 15,964.51 | 1.0213× | 1.000 | 48 |
| DeepSeek Distill 1.5B | 6,500.38 | 6,915.39 | 1.0638× | 1.000 | 56 |

The warm-up update intentionally uses each policy, so the first measured loss is
not expected to be bit-identical. First-loss relative differences are 0.0712% and
0.0088%; two-step final-loss differences are 0.0201% and 0.0035%. The observed
non-FFN parameter remains equal.

Candidate logical allocations increase by 192/224 over two steps, while peak
engine bytes remain unchanged. All six gates pass. The route remains explicit and
default-off until a longer loss/parameter trajectory passes.

`training.jsonl` contains all 12 performance processes. The two diagnostics files
prove exact route counts and zero strided-copy bytes.

