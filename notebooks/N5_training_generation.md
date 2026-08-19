# N5 — 训练、生成与统一评测

## 不要先挑一条漂亮生成

A training loss alone cannot establish validation quality. Before running, record:

```text
dataset/version/license
split boundary
tokenizer and vocabulary
model config and seed
batch/context/dtype
prompt set and generation seeds
```

The current repository has a byte tokenizer and generated sequence dataset. The real
Model-S corpus registry entry remains `planned`; therefore no real-corpus quality or
SFT claim is made.

## Model-S executable evidence

```bash
./build/examples/microllm_model_s_smoke
./build/examples/microllm_model_s_train_smoke
```

Observed three-step CPU smoke:

```text
11.2473 → 5.39479 → 1.98712
```

This proves all 15.6M parameters, gradients, AdamW moments, and updates connect. It is
not a pretraining curve.

## HIP forward and training

```bash
ctest --test-dir build-hip -L hip --output-on-failure
./build-hip/examples/microllm_model_s_hip_smoke
./build-hip/examples/microllm_tiny_hip_train
```

On MI300X/gfx942, Model-S CPU/HIP maximum logit error was `4.05312e-06`. Tiny HIP
training loss went from `2.21512` to `1.11681` in five steps. Nonlinear backward and
AdamW still cross the host boundary; N6 profiles this cost.

## KV Cache correctness

For MHA and GQA, every cached prefix logit is compared with full-prefix logits:

```bash
ctest --test-dir build --output-on-failure -R CachedLogits
```

Tolerance is `2e-5`; cache stores actual per-layer projected K/V, not token IDs.

## Generation

The generator supports greedy, temperature, top-k, and fixed seed:

```bash
ctest --test-dir build --output-on-failure -R 'SamplingTest|GeneratorTest'
```

## 预训练与 SFT 的边界

SFT uses the same next-token loss over formatted prompt/response sequences; it does
not need another optimizer. What is still missing is a licensed/versioned corpus,
document-boundary split, response masking policy, reference training run, and fixed
evaluation prompts. Until those artifacts exist, SFT remains an interface capability,
not a measured result.

## 下一步

N6 asks where time goes. It separates Event kernel time, synchronized wall time,
host copies, allocations, and complete tokens/s before accepting an optimization.
