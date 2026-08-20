# Experiment 032 — enable the retained allocator at the prefill boundary

Status: `keep` — prefill improves on both models; decode and exact tokens do not regress

## Failure

Experiment 031 measured full-sequence inference before enabling the exact-size HIP pool.
The pool was enabled only after generation warm-up, so decode used it but prefill still paid
repeated backend allocation/free costs.

This is a lifecycle bug, not a Kernel math bug:

```text
old
load → BF16 prepare → prefill warm-up → prefill measure → decode warm-up → enable pool

new
load → BF16 prepare → prefill warm-up → enable pool → reset counters → prefill measure
                                                    ↓
                                             decode reuses same pool
```

The CLI also gains `--workload prefill|decode|both`, so profiler runs do not need to execute
an unrelated workload first.

## Contract

- one timing/synchronization boundary after prefill warm-up;
- enable only the already-tested default-Stream exact-size allocator;
- reset peak/call counters after warm-up;
- do not change Tensor values, dtype policy, model weights or generation settings;
- keep decode within 5% of Experiment 031;
- use three independent processes per model/policy.

## Results

| Model/policy | Exp031 prefill | Exp032 prefill | Speedup | Decode change |
|---|---:|---:|---:|---:|
| Qwen FP32 | 96.85 tok/s | 158.45 tok/s | 1.636× | +0.04% |
| Qwen BF16 FFN | 107.67 tok/s | 176.74 tok/s | 1.642× | +0.60% |
| DeepSeek FP32 | 408.44 tok/s | 627.57 tok/s | 1.537× | +0.01% |
| DeepSeek BF16 FFN | 430.17 tok/s | 660.47 tok/s | 1.535× | +0.37% |

Every number is the median of three independent processes. Each process uses two warm-ups
and five measured iterations. All twelve rows reproduce the expected greedy token IDs and
the BF16 full-vocabulary differences remain unchanged.

Against the Experiment 031 PyTorch full-BF16 medians:

| Model | Decode ratio | Prefill ratio |
|---|---:|---:|
| Qwen | 1.179× | 1.216× |
| DeepSeek | 0.522× | 1.046× |

![Prefill allocator before/after](../assets/bf16-prefill-allocator.svg)

Three of four selected BF16 performance rows now pass. DeepSeek decode remains a stable,
large failure and becomes the only next target in this track.

## Correctness and scope

```text
CPU CTest          161/161 pass
ASan/UBSan         159/159 pass
HIP CTest           62/62 pass
official rows       12/12 exact expected tokens
```

This change optimizes both FP32 and BF16 prefill, but this report stays in the BF16 official
model track because that is where the missing allocator boundary was discovered. It does
not alter the historical FP32 `results.tsv` protocol or score.

## Decision

Keep the workload split and allocator boundary. The hypothesis is supported by large,
repeatable prefill gains on both model widths while decode stays flat. The next experiment
must profile DeepSeek decode only; another prefill optimization would no longer address the
remaining competitive failure.
