# 2026-08-19 — M3 trainable Decoder Transformer

## Contract

Compose a single Decoder-only architecture from the readable Autograd operations.
Support MHA and GQA, causal masking, RoPE, RMSNorm, SwiGLU, residual connections,
untied or tied output weights, deterministic initialization, stable parameter names,
and exact agreement with the executable parameter budget.

## Structure

```text
token embedding
→ repeated blocks:
    RMSNorm
    Q/K/V projections
    RoPE
    GQA K/V head repeat when configured
    scaled causal attention
    output projection + residual
    RMSNorm
    SwiGLU gate/up/down + residual
→ final RMSNorm
→ tied or untied vocabulary projection
```

## Verification

- constructed tiny GQA parameters equal `ModelConfig::parameter_count` exactly;
- parameter names are unique and every parameter requires gradients;
- forward logits have `B x T x vocabulary` shape and finite values;
- language-model loss produces finite, correctly shaped gradients for every parameter;
- changing future tokens leaves all earlier-prefix logits exactly unchanged;
- invalid token rank and over-length sequence fail visibly.

## Failure found

The first complete backward failed on `reshape → transpose → backward`: transpose
returned a non-contiguous gradient view and reshape correctly rejected a zero-copy
reshape. The shared reshape backward rule now materializes logical gradient order
before restoring the parent shape. A focused regression test preserves this case.

## Boundary

This milestone proves structure and gradient connectivity on tiny CPU float32 data.
It does not yet claim that Model-S has trained, that HIP backward exists, or that KV
cache logits match the full path.
