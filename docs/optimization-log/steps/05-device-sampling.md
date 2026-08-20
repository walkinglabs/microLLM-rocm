# Step 05 — device argmax, top-k and token loop

Status: `planned`

## Hypothesis

Copying the full 151,936-element logits vector to CPU every token adds avoidable latency
and synchronization.

## Design

- block-parallel argmax with deterministic tie rule;
- device top-k candidate selection;
- temperature and RNG state with explicit seed;
- next token remains on GPU for Embedding;
- copy final generated ID sequence only when returning text;
- optional scalar EOS check separated from full-logit copy.

## Required tests

- hand logits, equal maxima and non-finite rejection;
- vocabulary 32/8192/151936;
- greedy/top-1 exact equality;
- top-k fixed-seed equality within the agreed RNG contract;
- generated sequence bounds and EOS;
- profiler proves full logits D2H is gone.

## Falsification

If D2H disappears but generation throughput is unchanged, full-logit sampling was not a
material bottleneck at the current token count; retain only if complexity is low.

## Keep gate

All exact greedy tokens pass, no full-vocabulary D2H per token, and both official
generation rows are non-regressing.
