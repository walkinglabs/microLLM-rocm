# Experiment 013 — grouped Q/K/V decode projections

Status: `discard` — backend unavailable for tested decode shapes

## Observed bottleneck

Projection GEMMs are roughly 60% of Qwen decode Kernel time. Cached Attention submits
Q, K and V as three independent hipBLASLt calls even though they share the same input.

## Hypothesis

hipBLASLt extension `GroupedGemm` can submit the three independent outputs together
without packing or duplicating weights. Reducing projection submission/launch overhead
should improve decode while preserving external weight layout.

## Scope

- FP32 cached inference only;
- separate Q/K/V weight Storage and separate output Tensors;
- dimensions may differ for GQA K/V;
- bias remains the tested standalone add path;
- unsupported grouped heuristic falls back to three existing GEMMs;
- training, FP8, checkpoint/state mapping and model parameters unchanged.

## Required gates

- [x] varying output widths and shared input probe
- [x] equal output width control probe
- [x] fallback count visible
- [x] CPU/three-GEMM numerical parity
- [ ] grouped launch observed — no heuristic returned

## Probe implementation

The candidate used the extension descriptor API without packed weights:

- three independent weight pointers;
- one shared input pointer;
- three independent output Tensors;
- column-major reinterpretation identical to the proven regular GEMM path;
- one requested `GroupedGemm` heuristic, workspace limit zero;
- fallback to three existing hipBLASLt GEMMs when unavailable.

Two MI300X probes were executed:

```text
M=1 K=128, N={128,64,64}  grouped_launches=0 fallbacks=1
M=1 K=128, N={128,128,128} grouped_launches=0 fallbacks=1
```

Both numerical fallbacks matched the three CPU references, but neither grouped shape
returned a usable heuristic. The issue is not only varying GQA widths; the equal-width
control also fails for this FP32 decode configuration.

## Decision boundary

Running official model benchmarks would only measure the unchanged fallback, so they
were intentionally not run. Candidate API, extension dependency, model branch and tests
were removed. A future BF16 or larger-M prefill track may probe GroupedGemm again because
backend availability depends on dtype and shape.

## Results

Unsupported for the tested FP32 M=1 MI300X shapes.

## Decision

`discard`. No grouped-QKV source remains.
