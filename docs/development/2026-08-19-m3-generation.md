# 2026-08-19 — M3 cache-backed token generation

## Contract

Generate autoregressively from the real per-layer KV cache. Support deterministic
greedy decoding and seeded temperature/top-k sampling. Validate all sampling and
context inputs before model execution.

## Implementation

- prompt tokens are fed through cached forward once each;
- greedy mode is selected by temperature zero or top-k one;
- sampled mode uses stable exponentials over the selected top-k candidates;
- a caller seed owns all sampling randomness;
- prompt IDs, temperature, top-k, generated length, context capacity, and finite
  logits are checked;
- returned tokens retain the full prompt followed by the requested continuation.

## Verification

Four focused tests pass:

- greedy and top-one select the maximum logit;
- identical seeds produce identical 20-token top-k samples;
- cache-backed generation returns the requested length and only valid IDs;
- empty prompt, bad temperature, context overflow, and invalid token IDs fail.

## Boundary

These tests establish decoding semantics and reproducibility on randomly initialized
tiny CPU models. A trained-text generation report, multiple seeds, stop tokens,
batched decoding, and HIP KV cache remain subsequent work.
