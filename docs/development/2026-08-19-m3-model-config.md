# 2026-08-19 — M3 executable model configurations

## Contract

Replace hand-maintained parameter arithmetic with validated executable configuration.
Report parameter count separately from weight bytes at a named dtype.

## Results

```text
Model-S: 15,586,176 parameters
         62,344,704 FP32 weight bytes
         31,172,352 FP16/BF16 weight bytes

Model-M: 31,334,912 parameters
         125,339,648 FP32 weight bytes
         62,669,824 FP16/BF16 weight bytes
```

The count includes token embedding, Q/K/V/O projections, SwiGLU gate/up/down
projections, two RMSNorm weights per layer, final RMSNorm, and an untied output head.
GQA configurations reduce K/V projection counts according to `kv_heads`.

## Verification

Four configuration tests cover exact Model-S and Model-M budgets, byte footprints,
GQA reduction, invalid head divisibility, and even RoPE head dimension. The
`microllm_model_info` executable prints the same source-of-truth values.
