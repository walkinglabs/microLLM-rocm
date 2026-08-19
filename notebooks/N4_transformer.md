# N4 — 小型 Transformer 怎样逐层形成

## 预测 shape

For `B=1,T=4,D=8,H=2,KVH=1`, write the shapes of:

```text
embedding
Q
K/V before and after GQA repeat
attention scores
context
logits
```

Predict which logits may change if only token positions 2–3 change.

## 从局部模型到 Attention

The engine composes:

```text
Embedding
→ [RMSNorm → Q/K/V → RoPE → causal Attention → residual]
→ [RMSNorm → SwiGLU FFN → residual]
→ final RMSNorm → vocabulary projection
```

MHA uses `heads == kv_heads`. GQA stores fewer K/V heads and differentiably repeats
them for query groups; repeated gradients sum back to the original K/V head.

## 任务契约

```text
One Decoder-only architecture; no model-family switchboard.
No bias; RMSNorm, RoPE, SwiGLU, causal mask.
Stable named parameters and deterministic seed.
Actual parameter count must equal executable ModelConfig count.
Every parameter must receive finite gradient on a tiny batch.
Future tokens must not affect earlier logits.
```

## 运行结构和因果测试

```bash
ctest --test-dir build --output-on-failure -R TransformerModelTest
./build/examples/microllm_model_info
```

Model-S is exactly 15,586,176 parameters and 62,344,704 FP32 weight bytes. Model-M is
31,334,912 parameters and 125,339,648 FP32 weight bytes.

## Tiny overfit

```bash
./build/examples/microllm_tiny_overfit
```

Observed trajectory:

```text
step 1:  loss 1.81171
step 10: loss 0.468095
step 20: loss 0.0731425
step 40: loss 0.00673309
```

## 必交失败

Despite low loss, greedy output is:

```text
expected  0,1,2,3,0,1,2,3
observed  0,1,2,3,0,3,0,1
```

It succeeds inside trained context-four positions and fails beyond them. Competing
explanations are position-specific memorization and cached/full divergence. N5 tests
cache logits directly and keeps this generation failure visible.

## 下一步

N5 connects Model-S smoke training, real K/V cache, deterministic generation, and the
boundary between generated-data evidence and a real-corpus report.
