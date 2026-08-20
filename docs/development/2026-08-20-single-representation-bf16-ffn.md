# 2026-08-20 — single-representation BF16 FFN inference

The model now has a graph-free full-sequence `forward_inference()` and a one-way
`prepare_bf16_ffn_inference()` API. Preparation transactionally converts exactly three FFN
weights per layer and replaces their FP32 `Value`s with frozen BF16 values. No persistent
FP32 FFN copy remains.

Training forward and weight reload reject a prepared model. State export returns an FP32
snapshot, device migration retains BF16, and cached/full inference share the continuous
BF16 FFN operator.

Official Qwen and DeepSeek exact tokens pass. Against the retained microLLM FP32 path,
decode improves `1.115×/1.051×`, prefill improves `1.112×/1.053×`, and current engine
memory falls to `68.3%/67.5%`. Against full-model PyTorch BF16, only Qwen decode exceeds
1.0; the remaining three rows are the explicit next bottleneck.

Full verification: CPU `161/161`, sanitizer `159/159`, HIP `62/62`, PyTorch oracle `4/4`.
The 18 official rows, preparation peaks, summary and generated chart are in Experiment 031.
