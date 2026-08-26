# PyTorch ROCm cached typed Softmax

This is the same six-process, two-order FP16/BF16 matrix as the serial and block
baselines. Widths from 2048 through 8192 may use block-local FP32 shared memory to
retain each exponential between denominator reduction and final output conversion.
No Tensor-shaped engine allocation is introduced; widths outside the bounded range
keep the ordinary block or serial implementation.

All 10 precision, pointer, ownership and zero-peak-extra gates pass. At width4096:

| dtype | block Event | cached Event | Event gain | wall gain | cached/PyTorch Event |
|---|---:|---:|---:|---:|---:|
| BF16 | 10.828 μs | 8.701 μs | 1.244× | 1.226× | 0.550× |
| FP16 | 9.919 μs | 8.148 μs | 1.217× | 1.193× | 0.576× |

The bounded cache is accepted, but wide-row parity is not closed. The next candidate
must target the two full-block reductions and their barriers without changing the
FP32 accumulation order contract or allocating a global scratch tensor.

`raw.jsonl` preserves every process. `summary.json` contains correctness, ownership,
memory and PyTorch ratios.
