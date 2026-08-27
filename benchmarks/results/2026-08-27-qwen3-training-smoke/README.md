# Official Qwen3-0.6B training-step smoke

This is the first official Qwen3 training execution evidence. A temporary local manifest adds only
`training.learning_rate=1e-5`; model revision, weights and token seed remain unchanged.

Both runs use B1/T32, zero warm-up, one measured step and fresh microLLM/PyTorch processes.

## FP32

- both frameworks execute forward, loss, backward and AdamW update;
- loss: `2.37676549 / 2.376765728`, absolute difference `2.38e-7`;
- observed final-norm parameter after update differs by `2.57e-10`;
- peak: `9.778 / 12.102 GB` microLLM/PyTorch;
- single-process throughput ratio: `1.119x` (smoke only, not a performance claim).

## BF16 forward with FP32 masters

- both frameworks execute and update a parameter;
- microLLM retains 196 BF16 training mirrors over 880,803,840 bytes;
- loss: `2.377071381 / 2.367111206`, absolute difference `0.009960`;
- microLLM reaches only `0.5969x` the matched PyTorch throughput and `0.7900x` its own FP32 smoke;
- microLLM peak is `1.0901x` its FP32 smoke because FP32 masters, gradients and AdamW state remain.

The FP32 row is an execution/alignment smoke. The BF16 row is an execution smoke with an explicit
loss and performance gap. Neither row proves complete parameter/gradient alignment, multi-step loss
trajectory or repeated performance. Those require the next parameter-signature and trajectory gate.

Files:

- `fp32-raw.jsonl` / `fp32-summary.json`;
- `bf16-raw.jsonl` / `bf16-summary.json`;
- `summary.json`: compact decision.
