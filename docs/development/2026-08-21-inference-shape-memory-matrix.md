# Inference shape and memory matrix

This node turns inference benchmarking from one short prompt into an executable matrix.

## Scope

- contexts from 1 through 4096, plus values immediately around 32/128/512 boundaries;
- batches 1/2/4/8/16 and odd batch 3;
- output lengths 1/8/32;
- prefill, cached decode, and uncached decode;
- microLLM/PyTorch ROCm fresh-process pairing;
- FP32/BF16 and mixed-layer KV Cache policy reporting.

## New evidence contracts

Every successful cached row must prove that active tokens equal
`context + decode_tokens`. The steady-decode runner seeds the first input outside timing, then
executes exactly one model forward for every measured token. This prevents a one-token case from
measuring only the argmax already produced by prefill. microLLM can reserve the largest output
length in a sweep, while the current PyTorch DynamicCache reports its active prefix. The runner
records this policy instead of comparing unlabelled byte counts.

Summary rows now include batch scaling/efficiency, context-relative throughput and latency,
output-length efficiency, peak bytes per request, incremental peak above resident weights, active
and allocated KV bytes per request, KV share of incremental peak, throughput per peak GiB, and
fresh-process P50/P95 latency. The process percentile is not a claim about token-level production
tail latency.

## Fast executable gates

The tiny CPU matrix executes 66 context/batch/dtype combinations. The tiny HIP matrix executes 88
combinations and compares generated tokens with CPU. Contexts include 31/32/33, 63/64/65, and
127/128. The CPU gate additionally checks active view bytes, allocated Storage bytes, position
growth, and stable backing addresses after an append.

The official checkpoint matrix remains a slow/manual evidence job. OOM, unsupported shapes, and
cross-framework token divergence are retained as results.

The first frozen 72-record sweep used an unspecified build type, so it is accepted only for
semantics, KV bytes, peak memory and output-length coverage. A separate frozen Release/gfx942 N8
matrix supplies performance: Qwen meets PyTorch at all six shapes; DeepSeek meets it at T8/T512
but remains at 0.866x/0.671x for T2048 B1/B8. DeepSeek Release also retains its T2048 token
divergence instead of hiding it behind throughput.

See [the beginner-facing guide](../dev/inference-matrix.zh-CN.md) and
[Experiment 085](../optimization-log/experiments/085-inference-shape-memory-matrix.md).
