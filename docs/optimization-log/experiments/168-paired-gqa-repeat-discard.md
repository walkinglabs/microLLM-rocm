# Experiment 168 — halve repeat Kernels, regress Qwen

## Candidate

GQA repeats each KV head into several query heads. The BTHD graph needs two different
physical outputs:

```text
K [B,KV,T,D] → [B,H,T,D]
V [B,T,KV,D] → [B,T,H,D]
```

The old path launches one `repeat_interleave` for each Tensor in forward and again while
recomputing backward inputs. It then launches two separate reductions for dK and dV.

One paired forward Kernel computes `(b,h,t,d)` once, writes both layouts and uses
`kv=h/repeats`. One paired backward Kernel computes `(b,kv,t,d)` once, sums the same repeat
range for both gradients and writes their different layouts. It does not combine Storage,
change summation order or alter GQA math.

## Correctness gates

- CPU paired results equal two separate repeat/reduce calls;
- PyTorch independently repeats K on dim1, V on dim2 and groups both gradients;
- HIP compares all four outputs exactly with zero payload transfer;
- invalid B/H/T/D/repeat contracts fail;
- T256 saved Attention and complete CPU/HIP Transformer gradients still pass;
- same-binary false restores the exact separate route.

## Same-binary official T512 result

Both layout fusions stay true; rejected plan/alpha policies stay false. Only
`--attention-paired-gqa-repeat` changes.

| Model | Separate | Paired | Speedup | Peak ratio | Loss relative diff | Parameter equal |
|---|---:|---:|---:|---:|---:|---|
| Qwen2.5-0.5B | 15,292.18 | 14,922.01 tok/s | 0.9758× | 1.000 | 0.0187% | yes |
| DeepSeek Distill 1.5B | 6,291.09 | 6,343.67 tok/s | 1.0084× | 1.000 | 0 | yes |

Allocation counts are equal because the same two output Tensors still exist. Both
strided-copy diagnostics remain zero.

Qwen rocprofv3 proves the local mechanism:

```text
repeat forward  288 calls / 1.312 ms → paired 144 / 0.835 ms
repeat backward 144 calls / 0.793 ms → paired  72 / 0.496 ms
all dispatches  6,907 → 6,689
all Kernel time 110.668 → 109.360 ms
```

Fewer/faster repeat Kernels still do not survive uninstrumented Qwen scheduling. This is a
direct counterexample to using profiler totals alone as the merge decision.

![Paired GQA repeat discarded](../assets/paired-gqa-repeat-discard.svg)

## Decision

Reject the production policy and default false. Keep the paired forward/backward operators
as tested diagnostic primitives. A future GQA optimization must avoid the repeated output
Tensors themselves or change the GEMM batch mapping; combining the same memory traffic into
one thread is closed by this result.

Raw evidence is in
[`benchmarks/results/2026-08-23-paired-gqa-repeat/`](../../../benchmarks/results/2026-08-23-paired-gqa-repeat/).
