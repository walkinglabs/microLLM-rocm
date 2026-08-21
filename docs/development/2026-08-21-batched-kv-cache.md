# 2026-08-21 — batch-aware KV cache

KV cache, prefix population, step stores, cached GQA Attention, model prefill/step and
device row argmax now support a shared batch dimension. B1 remains the default constructor
contract. CPU/HIP tests cover independent rows, continuation logits, Storage shape and
zero payload transfers.

The official 48-record matrix passes. Qwen/DeepSeek B8 reaches 721/495 tok/s with
98.1%/99.5% scaling efficiency and exact tokens. microLLM's FP32 cache remains 2.057x the
PyTorch BF16 cache and becomes the next dtype experiment.

Final gates: full CPU/HIP 260/260, ASan/UBSan 179/179, and PyTorch-enabled CPU
184/184. The current machine does not provide `clang-format`, so formatting was checked
with `git diff --check` and compiler warnings; that missing optional tool is not reported
as a passed gate.

See [Experiment 064](../optimization-log/experiments/064-batched-kv-cache.md) and
[raw evidence](../optimization-log/experiments/064-data/).
