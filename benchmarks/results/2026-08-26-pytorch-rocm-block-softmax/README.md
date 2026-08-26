# PyTorch ROCm block-parallel typed Softmax

Six fresh MI300X processes measure both call orders for FP16/BF16 and widths
1/17/128/1024/4096. Each worker performs five warm-ups followed by 25 Event/wall
repetitions. The caller owns both tensors and the native Stream.

All 10 precision, pointer, ownership and zero-temporary gates pass. Relative to the
serial baseline, median microLLM Event time changes as follows:

| dtype | w128 | w1024 | w4096 |
|---|---:|---:|---:|
| BF16 | 15.680× | 99.945× | 148.896× |
| FP16 | 13.297× | 103.214× | 145.826× |

The block path reaches 1.213×/1.252× PyTorch at width128 and 1.114×/1.103× at
width1024 for BF16/FP16. Width4096 improves by roughly 146×–149× but remains only
0.430×/0.464× PyTorch. The accepted change is therefore shape-aware: widths at or
below 32 retain the serial row path; larger widths use one block per row. The next
experiment may cache FP32 exponentials for wide rows, but cannot add a Tensor-shaped
allocation or weaken the current precision gate.

`raw.jsonl` contains all per-process results. `summary.json` contains the six-process
medians and correctness/resource contract.
