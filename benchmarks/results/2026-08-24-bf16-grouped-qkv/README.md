# BF16 grouped QKV probe and model gate

Experiment 190 runs on gfx942 MI300X virtual functions with HIP runtime/driver
`71399004` and hipBLASLt `1.3.0`.

The experiment first profiles `load + 1 prefill` and `load + 6 prefills`, then subtracts
the first trace and divides by five. This removes one-time weight loading/preparation from
the hot-path conclusion. Qwen/DeepSeek T512 prefill spends 53.6%/61.9% of incremental
Kernel time in hipBLASLt GEMMs; DeepSeek's 84 Q/K/V projection GEMMs are the largest
single exact family.

The grouped probe uses the three existing BF16 weights and one shared BF16 activation.
Direct grouped FP32 output is unsupported in all six control processes. Grouped BF16
output is supported, so the real candidate includes three BF16-to-FP32 casts.

| Model | Pointer-stable Event | Wall | Reinitialized Event | Max/RMS projection error |
|---|---:|---:|---:|---:|
| Qwen | 1.881× | 1.553× | 0.908× | 0.000224 / 0.000062 |
| DeepSeek | 1.225× | 1.174× | 0.815× | 0.000244 / 0.000108 |

Reinitializing the grouped descriptor on every block is slower. The production candidate
therefore caches one initialized plan per exact tensor-pointer set. The existing QKV Arena
provides stable activation/output addresses; each block's distinct weight pointers select
its own plan.

The complete-model gate keeps the accepted BF16 FFN Arena baseline and changes only QKV
Arena plus grouped plans:

| Model | Speedup | Max/RMS logits | Peak ratio | Allocations |
|---|---:|---:|---:|---:|
| Qwen2.5-0.5B | 1.0317× | 0.09360 / 0.01978 | 1.0034 | 2895→2415 |
| DeepSeek-R1-Distill-Qwen-1.5B | 1.0015× | 0.06300 / 0.02044 | 1.0017 | 3375→2815 |

All outputs are finite, top tokens match, and the errors stay within the declared BF16
Max/RMS envelope. DeepSeek misses the joint 1.01 performance gate, so the exact registry,
plan cache, probe and CLI remain explicit while the model default stays unchanged.

Files:

- `operator-raw.jsonl`, `operator-summary.json`: 12 fresh capability/timing processes;
- `model-raw.jsonl`, `model-summary.json`: 12 fresh complete-model processes;
- `*-profile-delta.json`: derived per-prefill Kernel attribution;
- one/six-step Kernel CSVs: raw profiler inputs for each model;
- `verification.json`: build and regression evidence.
