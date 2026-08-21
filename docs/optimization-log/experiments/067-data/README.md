# Experiment 067 evidence

- `search/`: DeepSeek T512 B1 layer-policy search plus T32 rebuttal cases.
- `layer0-full/`: a full matrix showing that fixing T512 with layer 0 breaks T32 B1.
- `layer1-precision/`: the selected one-FP32-layer policy; 12/12 complete-logit rows pass.
- `formal-release/`: 72 process records for layer 1 FP32 and all other layers BF16.
- `comparison.json`: selected policy against Experiment 065 uniform BF16.

The policy is an explicit precision/performance trade-off, not an automatic model-name rule.
