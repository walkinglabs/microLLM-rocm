# Rejected FP16 fast-exp typed Softmax

The candidate changes only the FP16 cached/wave width range from precise device
`expf` to the HIP fast exponential intrinsic. The same six-process, two-order matrix
passes all ten precision, pointer, ownership and zero-peak-extra rows; FP16 maximum
error remains 1.19e-7 in the declared cases.

At width4096, relative to the retained FP16 wave baseline:

| metric | gain | keep gate |
|---|---:|---:|
| Event | 1.045× | 1.05× |
| wall | 1.034× | 1.05× |

Both performance metrics miss the gate, so the approximate intrinsic is removed even
though the measured low-precision outputs pass. The retained implementation continues
to use `expf`. A later candidate must change scheduling or occupancy rather than spend
the approximation budget on a sub-threshold gain.

`raw.jsonl` and `summary.json` retain the complete rejected evidence.
