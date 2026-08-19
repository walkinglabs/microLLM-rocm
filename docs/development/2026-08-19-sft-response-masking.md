# 2026-08-19 — SFT response masking

## Contract

Train on prompt/response sequences while excluding prompt-only prediction positions
from loss. Ignored targets must contribute neither loss nor logit gradient on CPU or
HIP. Response tokens remain ordinary next-token targets.

## Implementation

- `make_sft_batch` concatenates prompt/response and shifts inputs/targets;
- target positions predicting prompt tokens use sentinel `-100`;
- CPU and readable HIP cross entropy skip ignored rows and average valid rows;
- Autograd sets every ignored-row logit gradient exactly zero;
- all-ignored batches fail rather than divide by zero;
- a text helper formats `User:` / `Assistant:` using the byte tokenizer.

## Tiny SFT evidence

One prompt `{1,2,3}` and response `{4,5}` produces targets
`{-100,-100,4,5}`. A tiny Transformer trained for 30 steps:

```text
step=1  response_loss=1.88494
step=10 response_loss=0.439716
step=20 response_loss=0.0549371
step=30 response_loss=0.0106737
```

CPU mask/loss/backward tests and HIP forward conformance pass. This proves SFT
mechanics, not instruction-following quality or a Model-S SFT result.
