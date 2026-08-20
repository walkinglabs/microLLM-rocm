# Step 05 — device greedy argmax and token loop

Status: `complete for greedy` — Experiment 005, `keep`

## Hypothesis

Copying the full 151,936-element logits vector to CPU every token adds avoidable latency
and synchronization.

## Design

- block-parallel argmax with deterministic tie rule;
- stochastic top-k/temperature keeps the existing CPU reference in this experiment;
- next token remains on GPU for Embedding;
- copy final generated ID sequence only when returning text;
- optional scalar EOS check separated from full-logit copy.

## Required tests

- hand logits, equal maxima and non-finite rejection;
- vocabulary 32/8192/151936;
- greedy/top-1 exact equality;
- top-k fixed-seed equality remains covered by the unchanged CPU path;
- generated sequence bounds and EOS;
- profiler proves full logits D2H is gone.

## Falsification

If D2H disappears but generation throughput is unchanged, full-logit sampling was not a
material bottleneck at the current token count; retain only if complexity is low.

## Keep gate

All exact greedy tokens pass, no full-vocabulary D2H per token, and both official
generation rows are non-regressing.

## Measured result

```text
Qwen generate             85.64 → 93.34 token/s
DeepSeek generate         35.79 → 38.99 token/s
four-workload score       1.167931 → 1.219170
profiled generated D2H records     9 → 1
```

The device result is a `[1,1]` int32 Tensor. The C++ API copies that one scalar to append
to its returned token vector, while the same device Tensor feeds the next Embedding.
Stochastic device top-k remains separate future work and is not claimed here.
