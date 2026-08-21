# Readable fused Attention is not a FlashAttention backend

The installed ROCm environment exposes rocWMMA building blocks but no drop-in Composable Kernel or
FMHA runtime. Routing long prefill to the existing score-in-shared-memory reference kernel reduced
Qwen T512 B1 throughput to 0.360x of the hipBLASLt path while saving only 1.7% peak memory.

The route was reverted before any T2048 claim. A future online Attention implementation must use
matrix fragments and an explicit numerical contract; it cannot simply relabel the readable kernel.
See [Experiment 080](../optimization-log/experiments/080-readable-fused-attention-discard.md).
