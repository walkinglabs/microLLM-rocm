# Step 04 — preallocated device KV Cache and direct GQA

Status: `planned`

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
