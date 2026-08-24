# Step 13 — BF16 grouped QKV

Status: complete

## Contract

Use fresh phase-delta profiling to select one current inference family. Probe BF16 GroupedGemm
with real Q/K/V widths, complete outputs and reinitialization cost. Integrate only through stable
caller-owned buffers, then require both official models to pass correctness, performance and
memory gates.

## Result

- incremental GEMM share: 53.6% Qwen, 61.9% DeepSeek;
- pointer-stable grouped operator: 1.881×/1.225× Event;
- per-call descriptor initialization: 0.908×/0.815× counterexample;
- complete-model: 1.0317× Qwen, 1.0015× DeepSeek;
- complete logits stay within BF16 Max/RMS limits and top tokens match;
- peak ratios: 1.0034/1.0017.

## Decision

Keep explicit exact registry and cached primitive. Reject the cross-model default. This exact
two-checkpoint hypothesis is closed until a model-independent shape rule gains more evidence.
