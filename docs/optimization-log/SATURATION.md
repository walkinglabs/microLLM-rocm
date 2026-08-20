# Local optimization saturation audit

Date: 2026-08-20 · retained FP32 score: `2.478439`

This does **not** say the framework cannot become faster. It says the current class of
small, local FP32 M=1 decode edits has reached a measured stopping point. Continuing to
permute the same knobs would repeat rejected experiments rather than produce research.

## Search spaces closed by evidence

| Search space | Measured conclusion |
|---|---|
| allocator retirement batch | 8 works, 16 wins, 24 and 32 lose |
| cached Attention block size | 64/128-thread specialization loses to 256 |
| cached Attention query staging | shared-memory query lowers DeepSeek and score |
| local bias fusion | hipBLASLt epilogue and V-bias/store both lose end-to-end |
| Q/K/V grouping | FP32 M=1 GroupedGemm exposes no heuristic |
| hipBLASLt host caching | descriptor and algorithm caches lose repeated matrices |
| explicit GEMM solution | stable micro gain does not survive DeepSeek model gate |
| local gradient reuse | copy-on-write condition never reduces measured allocations |
| device-selection API cache | API count improves, all uninstrumented workloads regress |
| cross-block residual/Norm | 28 fewer launches, Qwen regression lowers score |

## Retained local architecture

- parallel CE and RMSNorm;
- transpose-aware hipBLASLt GEMM;
- device KV cache, direct GQA and fused cached Attention;
- fused Q/K bias+RoPE and paired K/V store;
- two-stage large-vocabulary argmax;
- steady-state exact-size pool with 16-block shared retirement Events;
- width-aware fused residual-Norm.

## Why the next work is architectural

The remaining trace is dominated by repeated GEMM and whole-model scheduling. The next
valid tracks change a larger contract and therefore require new baselines instead of
extending the existing local curve:

1. BF16 activation islands with no permanent FP32+BF16 weight duplicate;
2. packed gate/up or QKV weights loaded directly, not cached as a second model copy;
3. HIP Graph capture with stable addresses and explicit eager fallback;
4. prefill/training Attention forward+backward fusion;
5. autograd liveness planning rather than local `use_count` guesses;
6. separate long-context and batch>1 matrices;
7. multi-GPU overlap measured independently from this single-GPU curve.

Each item must start with a new task contract, correctness oracle and track-specific
figure. The FP32 M=1 running best remains frozen until a candidate passes the same fixed
matrix.
