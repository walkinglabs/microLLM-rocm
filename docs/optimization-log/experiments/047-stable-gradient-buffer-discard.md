# Experiment 047 — stable leaf gradient buffers (discarded)

## Hypothesis

If leaf gradient Storage survives `zero_grad()`, a later multi-tensor optimizer can retain
a device pointer table across steps and replace 339 AdamW launches with one launch.

## Candidate

The candidate separated “buffer exists” from “gradient is valid”. Non-leaf gradients kept
their old lifetime. Each leaf owned a persistent FP32 buffer; the first contribution of a
new backward used device-native `copy_`, later branches used `add_`. CPU/HIP tests covered:

- repeated backward and branch accumulation;
- address equality across `zero_grad()` and `set_grad()`;
- in-place storage identity;
- no optimizer Tensor payload transfer.

All 18 targeted tests passed after one test-detected device-pointer API correction.

## Protocol correction

The first performance run used two warm-ups and five measured steps. Its microLLM median
was 755.14 token/s, but the retained Experiment 044 baseline used one warm-up and two
measured steps. That division was rejected rather than reported.

The candidate was rerun with the exact old protocol: Qwen2.5-0.5B, BF16 Linear/FP32
master, batch 1, context 128, three fresh processes with alternating framework order.

## Result

| Metric | Experiment 044 | Candidate | Change |
|---|---:|---:|---:|
| microLLM throughput | 802.70 tok/s | 757.48 tok/s | **−5.63%** |
| peak engine memory | 10.450 GB | 9.906 GB | −5.21% |
| allocation calls | 3,296 | 3,294 | −2 |
| optimizer payload H2D/D2H | 0 / 0 | 0 / 0 | unchanged |

![Stable gradient buffer discard](../assets/stable-gradient-buffer-discard.svg)

The memory result is useful, but the keep gate does not trade an unexplained >5% speed
regression for it. Copying the first contribution adds a full gradient memory pass and a
launch for each leaf. The candidate was removed.

## Revised design

Stable gradient addresses are not actually required if descriptors are current at each
launch. The next candidate will pass bounded groups of pointers and sizes as ordinary HIP
Kernel arguments. A chunk of 16 tensors stays below the Kernel argument limit, needs no
device pointer table, accepts fresh gradient addresses, and still reduces roughly 339
AdamW launches to about 22.

## Falsification

The chunked candidate must match the scalar AdamW state/parameter result, keep payload
H2D/D2H at zero, reduce actual AdamW dispatches, and improve the formal Qwen/DeepSeek
shape medians. Launch reduction alone is insufficient.
