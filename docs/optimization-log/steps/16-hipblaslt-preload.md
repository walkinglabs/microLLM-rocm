# Step 16 — cold-start preload counterexample

Status: complete

## Evidence

- 18 fresh processes, zero warm-up, one T512 prefill;
- FP32 first forward is already 3582/3564 ms;
- BF16 lazy first forward is 5030/4968 ms;
- BF16 all-kernel preload first forward is 17190/17123 ms;
- preload slows forward by 3.417×/3.447× and process wall by 3.140×/2.938×;
- complete logits pass the retained BF16 Max/RMS gate;
- engine peak memory is unchanged.

## Decision

Reject all-kernel preload and do not wrap ordinary full-forward warm-up in a new API. Keep lazy
loading plus exact targeted prewarm. Any next startup candidate must reduce selected-library work,
not merely move or broaden it.
