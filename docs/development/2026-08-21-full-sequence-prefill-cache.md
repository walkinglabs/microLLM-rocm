# 2026-08-21 — full-sequence prefill-to-cache

Added a B=1 full-sequence cache prefill API. It computes each layer once, writes K/V into
capacity-strided Storage per head, advances cache position by T and returns last-token logits.
The CLI defaults to this path and retains explicit token replay only as a research control.
The public generator uses the same API and transfers the whole prompt in one H2D call.

CPU/HIP tests cover full-vs-prefill logits, continuation logits, cache position/Storage,
invalid batch atomicity and zero payload transfers. Two defects were caught before performance:
capacity-stride corruption and unnecessary full [T,V] output memory.

Qwen/DeepSeek T1024 prepare is 71/109ms instead of historical 38/55s warm-up totals. A
same-window profiled Qwen T512 control improves prepare 275x and Kernel time 112x.

See [Experiment 062](../optimization-log/experiments/062-full-sequence-prefill-to-cache.md)
and [raw evidence](../optimization-log/experiments/062-data/).
