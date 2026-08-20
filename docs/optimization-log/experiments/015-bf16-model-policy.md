# Experiment 015 — BF16 model policy and FP32 master gradients

Status: `partial keep` — autograd primitive kept; model policy discarded

## Question

Experiment 014 found two M=1 Linear shapes where BF16 mixed GEMM was faster.
Would an exact shape allow-list plus cached BF16 weights make official Qwen and
DeepSeek decoding faster without changing generated tokens?

## Candidate

The experimental model build kept all public tensors and outputs in FP32. It
cached a BF16 copy only for `(K,N)=(896,896)` and `(1536,8960)`, then used the
mixed GEMM path for those exact shapes. It did not silently convert other
Linear layers.

The independent framework change was a `bf16_matmul` autograd primitive:

```text
forward:   round both operands to BF16 → FP32 accumulation/output
backward:  use FP32 master operands and FP32 upstream gradient
```

This straight-through boundary is useful for later mixed-precision training,
but it does not claim that an entire model now trains in BF16.

## Extra shape gate

| Shape M×K×N | BF16/FP32 | Result |
|---|---:|---|
| 1×384×832 | 0.845× | slower |
| 1×832×384 | 0.972× | slower |
| 1×896×128 | 0.899× | slower |
| 1×4864×896 | unavailable | hipBLASLt status 6 |
| 1×1536×256 | 0.898× | slower |
| 1×8960×1536 | unavailable | hipBLASLt status 6 |

This already falsified a broad “all projections should use BF16” rule.

## Official-model result

Three independent processes were measured. Every process used 2 warm-ups and
5 timed decode steps; the table compares process medians with the retained FP32
Experiment 012 medians.

![Rejected BF16 model policy](../assets/bf16-model-policy.svg)

| Model | FP32 token/s | Candidate token/s | Ratio | Extra engine memory |
|---|---:|---:|---:|---:|
| Qwen2.5-0.5B | 147.41 | 125.29 | 0.850× | +73.5 MiB |
| DeepSeek Distill 1.5B | 53.36 | 51.85 | 0.972× | +1.44 GiB |

Both models produced the same greedy token IDs as FP32. Correct tokens are
necessary, but the speed and memory gates both failed. Caching extra BF16
weights while repeatedly casting FP32 activations did not create an end-to-end
win.

## Verification of the retained part

- CPU debug: `154/154` pass;
- ASan/UBSan: `152/152` pass;
- MI300X/gfx942 HIP: `55/55` pass;
- Python/PyTorch forward and backward oracle: `4/4` pass;
- BF16 forward is checked against independently rounded PyTorch operands;
- left/right gradients are checked against FP32 master-weight gradients;
- graph coverage manifest requires the BF16 gradient case.

## Decision

Keep the tested operator and autograd building block. Remove the model precision
enum, cached-weight policy and CLI option. A later model-level attempt must keep
activations BF16 across compatible operator chains and avoid persistent duplicate
weights; simply wrapping a few FP32 Linear calls is not enough.

Raw evidence is in [015-data](015-data/README.md).
