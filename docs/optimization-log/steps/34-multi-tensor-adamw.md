# Step 34 — multi-tensor AdamW

Status: complete, model route rejected, primitive retained

## Question

Can one descriptor-driven Kernel replace 290/339 per-parameter AdamW launches and improve both
official B1/T512 models without weakening the complete-state gate?

## Contract

- compare every parameter, first moment, second moment and BF16 mirror within `2e-6`;
- allow a missing gradient to leave that complete Tensor state unchanged;
- upload exactly one metadata table per optimizer step and no value payload D2H;
- require both model medians to reach `1.01×` and isolated AdamW Kernel time to reach `1.10×`;
- cap added peak memory at 1%; keep ordinary AdamW as the default until all gates pass.

## Decision

Qwen passes at `1.0573×`, but DeepSeek reaches only `1.0094×`. Isolated AdamW is
`1.4699×/1.0828×`. Remove the temporary model/CLI route. Retain the independently tested
primitive because it removes 867/1,014 launches and gives the next scheduler a measured base.
