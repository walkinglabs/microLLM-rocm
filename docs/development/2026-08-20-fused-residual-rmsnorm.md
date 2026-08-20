# 2026-08-20 — fused cached residual and RMSNorm

## Design

`ops::add_rms_norm(left, right, weight)` returns two FP32 tensors:

```text
sum        = left + right
normalized = rms_norm(sum, weight)
```

On CPU this is the readable composition. On HIP, one block per row writes the sum,
reduces its squared values, then writes the normalized output. Cached Transformer
inference uses the pair after attention; the autograd training path is unchanged.

## Evidence

- independently composed CPU and PyTorch results match both outputs;
- HIP matches CPU and performs no host payload transfer;
- cached MHA/GQA logits and official greedy tokens remain exact;
- CPU `157/157`, sanitizer `155/155`, HIP `56/56`, Python oracle `4/4` pass;
- DeepSeek profile removes 532 Kernel launches and lowers `hipLaunchKernel` API time
  from 65.26 to 59.70 ms.

Three inference processes produce a mixed result: Qwen +8.9%, DeepSeek -4.2%. The fixed
four-workload score rises `1.784147 → 1.803226`, so the candidate passes the written gate,
but the DeepSeek regression remains an open failure rather than being averaged away.

Full data: `docs/optimization-log/experiments/017-fused-residual-rmsnorm.md`.
