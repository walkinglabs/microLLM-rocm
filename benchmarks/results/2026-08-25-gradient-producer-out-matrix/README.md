# Caller-owned weight-gradient producer matrix

Experiment 260 measures `input^T @ output_gradient` for four real Model-S shapes
and one tiny counterexample. The baseline allocates a producer Tensor and performs
the leaf add; the candidate writes the caller target directly. Each shape uses
three fresh processes, 5 warm-ups, 40 repetitions, and alternating operation and
shape order.

| Shape | Event speedup | Wall speedup | Allocation calls |
|---|---:|---:|---:|
| Model-S head T32 | 1.873x | 1.612x | 1 → 0 |
| Model-S FFN T32 | 1.260x | 1.181x | 1 → 0 |
| Model-S Attention T32 | 1.179x | 1.122x | 1 → 0 |
| Model-S head T512 | 1.426x | 1.363x | 1 → 0 |
| tiny counterexample | 1.178x | 1.101x | 1 → 0 |

All 15 complete outputs are bit-exact. CPU, HIP, and PyTorch operator parity pass.
Every shape clears the 1.05 Event and wall gate, so exact producer shapes advance
to a scoped Autograd test. No model or DDP route is created by this experiment.
