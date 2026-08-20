# 2026-08-20 — official HF single-GPU memory/performance matrix

## Goal

Extend the built-in tiny/Model-S/Model-M ladder with common official small-model
checkpoints without making large external files a repository or CI dependency.

The chosen architecture-compatible targets are:

- Qwen2.5-0.5B, 494,032,768 parameters and 290 checkpoint Tensors;
- DeepSeek-R1-Distill-Qwen-1.5B, 1,777,088,000 parameters and 339 Tensors.

The second target is the dense Qwen distillation model, not flagship DeepSeek MLA/MoE.

## Availability contract

The runner receives paths through a local JSON manifest. Missing config, weights,
vocab, or merges produces an `unavailable` row and an `incomplete` summary. The command
fails unless the caller explicitly uses `--allow-unavailable`; even then no row changes
to `pass`.

This negative path is registered in CPU CTest with the repository example manifest.

## MI300X result

All four requested real-checkpoint measurements passed:

| Model | Mode | Work | Throughput | Peak engine GiB |
|---|---|---:|---:|---:|
| Qwen2.5-0.5B | inference | 2-token prefill, 4-token decode | 19.524 decode token/s | 3.681 |
| Qwen2.5-0.5B | train | 3 predicted tokens | 1.571 token/s | 8.901 |
| DeepSeek Distill 1.5B | inference | 12-token prefill, 8-token decode | 10.252 decode token/s | 13.240 |
| DeepSeek Distill 1.5B | train | 3 predicted tokens | 1.350 token/s | 26.514 |

Inference exact-token gates:

```text
Qwen       [0,358,2776,264]
DeepSeek   [40,1184,311,8253,279,2629,315,220]
```

Training gates:

```text
Qwen loss                         6.83602953
DeepSeek loss                    11.039709091
Qwen AdamW payload H2D/D2H        0 / 0
DeepSeek AdamW payload H2D/D2H    0 / 0
both observed parameters changed true
```

## Important boundary

This run uses FP32 compute after decoding BF16 source weights. The result does not prove
BF16/FP8 whole-model performance, long-context throughput, model quality, or a speedup
over PyTorch. Each mode is a single short run, not a repeated performance distribution.

Peak bytes are engine-owned allocations. The runner now samples them after generation,
so cache activity contributes to the inference peak. Vendor/driver-private allocations
still require an external sampler for board-level memory claims.
