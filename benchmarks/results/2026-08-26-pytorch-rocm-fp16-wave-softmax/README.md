# FP16-only wave-reduction typed Softmax

This experiment turns the rejected broad wave candidate into an explicit compile-time
dtype policy. Cached FP16 widths 2048–8192 use wave-level max/sum reduction; BF16
instantiates the same cached Kernel with the retained shared-tree reduction.

The six-process, two-order ten-case matrix passes every precision, pointer, ownership
and zero-peak-extra gate. At width4096:

| route | Event gain vs cached | wall gain vs cached | current/PyTorch Event |
|---|---:|---:|---:|
| BF16 fallback | 1.002× | 1.004× | 0.539× |
| FP16 wave | 1.077× | 1.080× | 0.615× |

The FP16-only predicate is accepted because both affected metrics exceed 1.05 while
the BF16 fallback remains within ±5% process noise and is a compile-time false branch.
This is not wide-row parity: FP16 remains below PyTorch and BF16 intentionally keeps
the previous reduction.

`raw.jsonl` and `summary.json` retain every measured process and resource field.
