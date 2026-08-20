# Step 04 — preallocated device KV Cache and direct GQA

Status: `complete` — Experiment 004, `keep`

## Hypothesis

CPU cache concatenation, repeated allocation and physical GQA expansion explain a
large portion of the 3.7–6.2x generation gap.

## Design

```text
K/V storage [B, kv_heads, max_sequence, head_dim]
logical length
device append at position
query_head → kv_head mapping inside Attention
no expanded K/V Tensor
```

Prefill and decode APIs must be separate. Cache owns Storage; views do not own or free it.

## Required tests

- every prefix against full forward for MHA and GQA;
- position 0, last valid position and overflow;
- reset and reuse;
- multiple layers and future batch dimension;
- cache address stability;
- measured decode transfer counters;
- exact Qwen/DeepSeek greedy tokens.

## Falsification

If host copies and reallocations disappear but generation remains slow, RMSNorm/output
projection/launch overhead is the stronger explanation.

## Keep gate

- no Tensor payload host roundtrip during measured decode;
- no whole-cache allocation per token;
- no physical GQA expansion;
- context 1/32/128/512 curves retained.

## Measured result

```text
Qwen generate             57.32 → 85.64 token/s
DeepSeek generate         18.60 → 35.79 token/s
four-workload score       0.885816 → 1.167931
profiled Qwen hipMemcpy calls 2712 → 600
profiled copyBuffer calls      2269 → 253
```

The first candidate allocated the model's theoretical maximum context and raised
DeepSeek peak memory to 14.63 GB. It was rejected. The kept design reserves the known
request bound (`prompt + max_new_tokens`), preserving stable addresses without that cost.

See [Experiment 004](../experiments/004-device-kv-cache.md) and its retained context
curve for the complete evidence.
