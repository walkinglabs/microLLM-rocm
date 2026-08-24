# Multi-tensor AdamW evidence

Experiment 211 asks whether one descriptor-driven HIP launch can replace one AdamW launch per
parameter Tensor. The retained public primitive owns a stable device block map, uploads only a
small pointer descriptor table each step, and updates FP32 parameters, gradients, both moments,
and optional BF16 mirrors.

## Files

- `sync-*`: first three-process candidate with a blocking descriptor copy;
- `async-pilot-*`: three-process pinned/async refinement;
- `training.jsonl` and `summary.json`: final five-process, alternating-order model gate;
- `*-kernel-stats.csv`: two models × per/multi structural profiles;
- `profile-summary.json`: compact dispatch and isolated AdamW time;
- `coverage-summary.json` and `verification.json`: final regression evidence.

## Final result

| Model | Per Tensor | Multi Tensor | Speedup | Peak added | AdamW Kernel speedup |
|---|---:|---:|---:|---:|---:|
| Qwen2.5-0.5B | 14,742.46 | 15,587.30 tok/s | 1.0573× | 7,735,744 B | 1.4699× |
| DeepSeek Distill 1.5B | 6,226.66 | 6,285.24 tok/s | 1.0094× | 27,785,984 B | 1.0828× |

The temporary model/CLI route is rejected because DeepSeek misses the predeclared `1.01×`
end-to-end gate and its isolated AdamW Kernel misses `1.10×`. The public primitive, pinned
descriptor lifetime, native-stream async copy, complete-state HIP test and profiler evidence are
retained for a future block scheduler. Ordinary AdamW remains unchanged.
