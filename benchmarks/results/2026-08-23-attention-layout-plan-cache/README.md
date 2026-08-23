# Interleaved Attention plan-cache evidence

Experiment 166 caches only immutable hipBLASLt description/layout objects for exact
`{mode,H,T,D,device}` keys. Tensor pointers, algorithms and workspace remain per call.

- `operator-raw.jsonl` / `operator-summary.json`: four shapes × cached/uncached × three
  fresh processes, 20 repetitions after three warm-ups;
- `training.jsonl` / `summary.json`: Qwen/DeepSeek T512 cached/uncached × three fresh
  training processes;
- `route-smoke/`: one-step route proof showing exact model hit/miss counts;
- `*-diagnostics.json`: both policies preserve zero strided-copy state;
- `coverage-summary.json`: post-change CPU coverage;
- `verification.json`: final default-off decision and regression gates.

The operator wall medians improve 1.053×–1.185×, including Qwen/DeepSeek T512 at
1.0667×/1.0689×. The full-model medians do not pass the declared 1.01 gate:
Qwen is `0.9902×`; DeepSeek is `1.0005×`. Peak, allocations, loss guard and observed
parameter are unchanged.

The API and explicit benchmark/CLI control remain for diagnosis, but production and CLI
defaults are false. A one-step route smoke proves exactly three misses followed by 69 Qwen
or 81 DeepSeek hits; the uncached route stays at zero cache state.
