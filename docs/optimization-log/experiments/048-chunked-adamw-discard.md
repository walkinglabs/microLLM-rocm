# Experiment 048 — chunked AdamW (discarded)

## Hypothesis

Experiment 046 found 339 AdamW launches per DeepSeek step. Passing current pointers in
bounded Kernel arguments could reduce launch count without persistent gradient buffers or
Tensor payload copies.

## Correctness contract

The candidate accepted at most 16 `{parameter, gradient, first moment, second moment,
optional BF16 mirror}` records. A 33-Tensor HIP test crossed the capacity boundary and
matched the scalar CPU path for parameters, both moments and mirrors. It also proved zero
payload H2D/D2H. The training CLI exposed Tensor, scalar and group counts.

## First failure: grouping every Tensor

Qwen has 290 parameter tensors. All-Tensor grouping reduced that to 19 launches, but the
first `1×128` process fell from the retained 802.70 to 463.00 token/s (`0.577×`). Peak
memory was unchanged. The remaining processes were stopped.

The large Kernel-argument record and per-block Tensor mapping were now paid by every block
of every large matrix. Fewer launches made the actual work slower.

## Revised candidate: group only small Tensor updates

Safetensors shape inspection showed:

```text
Qwen:     121 tensors ≤ 4096 elements, 169 larger tensors
DeepSeek: 141 tensors ≤ 4096 elements, 198 larger tensors
```

Large tensors returned to the original scalar Kernel. The 121 Qwen small tensors became
eight groups, reducing optimizer dispatches from 290 to 177.

| Shape | Before | Small-group candidate | Speedup | Peak ratio |
|---|---:|---:|---:|---:|
| 1×3 | 33.45 | 33.05 tok/s | 0.988× | 1.000× |
| 2×3 | 64.57 | 66.30 tok/s | 1.027× | 1.000× |
| 1×32 | 304.38 | 311.22 tok/s | 1.022× | 1.000× |
| 1×128 | 802.70 | 806.79 tok/s | 1.005× | 1.000× |

![Chunked AdamW discard](../assets/chunked-adamw-discard.svg)

No shape reaches the 1.05 keep gate. One regresses 1.2%; memory never changes. Reducing
39% of dispatches is real, but it is not an end-to-end optimization. The implementation
was removed.

## Revised explanation

The 32.94% AdamW share in the DeepSeek process trace is dominated by bytes moved through
large parameters and moment arrays, not by launches for tiny Norm/bias tensors. The next
candidate must first benchmark vectorized contiguous loads/stores on exact large optimizer
shapes. A faster micro Kernel is still insufficient unless the official shape matrix moves.
