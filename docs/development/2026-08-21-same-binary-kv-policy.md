# 2026-08-21 — same-binary KV-policy comparison

A new benchmark runner pairs uniform BF16 and per-layer strict Cache policies using one Release
binary. Fresh-process order alternates per run, and summaries compute medians only from complete
policy pairs.

The 72-record Qwen/DeepSeek matrix invalidates the earlier cross-window claim that strict DeepSeek
T2048 B8 end-to-end is 13.4% slower. In the same window its prepare ratio is 0.994x, end-to-end is
1.011x and decode is 1.034x. The strict policy remains explicit because its layer selection is
checkpoint-specific and its Cache is 3.57% larger than uniform BF16—not because a long-batch
slowdown was proven.

See [Experiment 069](../optimization-log/experiments/069-same-binary-kv-policy.md) and the
[72 raw records](../optimization-log/experiments/069-data/).

Final gates: full CPU/HIP 268/268, ASan/UBSan 184/184 and PyTorch-enabled CPU 189/189.
