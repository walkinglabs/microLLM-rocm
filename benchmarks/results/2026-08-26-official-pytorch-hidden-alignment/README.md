# Official Qwen/DeepSeek hidden-state alignment

The C++ engine and PyTorch load the same pinned safetensors, consume the same four token
IDs, run FP32 eager Attention, and export embedding, every decoder block, final norm and
last-token logits. Large synchronous engine traces are deleted after comparison; the
repository retains complete per-stage metrics.

| model | stages | first nonzero | maximum relative-L2 | logits Max | logits RMS |
|---|---:|---|---:|---:|---:|
| Qwen2.5-0.5B | 27 | `blocks.0` | 2.89e-5 at `blocks.21` | 8.01e-5 | 1.01e-5 |
| DeepSeek-R1-Distill-Qwen-1.5B | 31 | `blocks.0` | 2.85e-6 at `blocks.0` | 2.48e-5 | 4.19e-6 |

Both embedding outputs are bit-exact. Every expected layer is present with an identical
shape. This is synchronous numerical evidence, not a performance trace.

`raw.jsonl` stores one model record each. `summary.json` stores the combined contract.
