# Experiment 031 — single-representation BF16 FFN model inference

Status: `partial keep` — self-baseline speed/memory pass; selected PyTorch BF16 gate fails

## Question

Experiment 015 failed because it kept two permanent copies of selected weights and cast
every Linear input. Experiment 030 proved a continuous FFN operator island. The next
question is stricter:

> Can official Qwen and DeepSeek replace each FFN FP32 weight with one BF16 weight,
> preserve useful outputs, reduce resident memory and improve whole-model inference?

## Design in plain language

Imagine that the model owns a shelf of books. The rejected design put a thick FP32 book
and a thin BF16 copy on the shelf forever. The new API makes a complete thin copy first,
checks that copying succeeded, then removes the thick FFN books. After preparation there
is only one FFN copy.

```text
load ordinary FP32 model
        ↓
transactionally cast every FFN weight
        ↓ success                     ↓ failure
replace FP32 with BF16                keep original FP32 model unchanged
        ↓
graph-free forward_inference / forward_cached only
```

`prepare_bf16_ffn_inference()` is deliberately one-way. It freezes the converted weights,
rejects autograd `forward`, rejects another weight load, and keeps `state_dict()` as an
FP32 export snapshot. `to(device)` preserves the prepared dtype. This prevents a caller
from accidentally training a half-converted model.

Preparation is transactional, so it temporarily holds both representations. The duplicate
is not persistent, but its peak is reported separately.

## Fixed protocol

```text
GPU             AMD Instinct MI300X VF
architecture    gfx942:sramecc+:xnack-
ROCm/HIP        7.13.99004
models          pinned Qwen2.5-0.5B and DeepSeek-R1-Distill-Qwen-1.5B revisions
processes       3 per model and policy
prefill         2 warm-up, 5 measured full forwards
decode          2 warm-up, 5 measured generations
tokens          Qwen 4 new tokens; DeepSeek 8 new tokens
references      retained microLLM FP32 and Python/Transformers full BF16
```

The microLLM policy is mixed: attention, embedding, Norm and output remain FP32; only FFN
weights and compatible activations are BF16. PyTorch uses full-model BF16. They answer a
useful competitive question but are not identical dtype policies.

## Results

| Model | Decode vs micro FP32 | Prefill vs micro FP32 | Decode vs PyTorch BF16 | Prefill vs PyTorch BF16 | Engine current |
|---|---:|---:|---:|---:|---:|
| Qwen2.5-0.5B | 1.115× | 1.112× | 1.172× | 0.741× | 68.3% of FP32 |
| DeepSeek Distill 1.5B | 1.051× | 1.053× | 0.520× | 0.681× | 67.5% of FP32 |

![Official-model BF16 FFN result](../assets/bf16-model-inference.svg)

All 18 records reproduce the fixed expected greedy token IDs. Maximum full-vocabulary
logit differences against microLLM FP32 run 1 are `0.15114` for Qwen and `0.11045` for
DeepSeek. Those are BF16 policy differences, not FP32 equality claims.

Absolute timings were lower than an earlier session, indicating shared-GPU/process-state
variation. The keep decision therefore uses paired three-process medians from this one
fixed matrix, not historical high points.

## Weight and memory evidence

| Model | FP32 logical weights | Persistent prepared weights | Preparation peak | Converted tensors |
|---|---:|---:|---:|---:|
| Qwen | 1,976,131,072 B | 1,348,558,336 B | 2,603,703,808 B | 72 |
| DeepSeek | 7,108,352,000 B | 4,796,241,920 B | 9,420,462,080 B | 84 |

The preparation peak equals original FP32 weights plus the new BF16 FFN copy. After the
transaction commits, current bytes equal the single prepared representation. Generation
peak is reset and reported separately by the CLI.

## Correctness gates

```text
CPU CTest                 161/161 pass
ASan/UBSan CTest          159/159 pass
HIP CTest                  62/62 pass
Python/PyTorch oracle        4/4 pass
official raw rows           18/18 exact expected tokens
```

The small full-model oracle rebuilds the mixed BF16 FFN path in Python `torch`. Dedicated
tests also cover MHA/GQA graph-free inference, CPU/HIP values, zero payload transfers,
single-representation state, FP32 snapshot export and illegal training/reload operations.

## Decision

Keep the model API and policy because both official models improve against the retained
microLLM FP32 path, exact tokens pass, and persistent engine memory falls about 32%.

Do not claim selected-matrix PyTorch BF16 parity. Three of four competitive ratios are
below 1.0: both prefill rows and DeepSeek decode. The next optimization target is therefore
the BF16 full-sequence/DeepSeek timeline, not another dtype switch.

This experiment remains outside the FP32 `results.tsv`; the FP32 running best stays
`2.478439`.
