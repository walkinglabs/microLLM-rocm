# Longer BF16 gate/up weight-gradient trajectory

Experiment 247 repeats baseline/candidate for 20 measured B1T512 steps, three
fresh processes per model and policy. Run 1 also exports every gate/up FP32 master
parameter, compares complete safetensors, then deletes the temporary snapshots.

| Model | Throughput | Peak | Max loss relative diff | Parameter Max | Parameter RMS |
|---|---:|---:|---:|---:|---:|
| Qwen2.5-0.5B | 1.0006× | 1.000× | 15.705× | 1.407e-4 | 1.218e-5 |
| DeepSeek Distill 1.5B | 1.0528× | 1.000× | 7.70e-4 | 6.235e-5 | 8.299e-7 |

Qwen's relative loss ratio is amplified when baseline loss approaches zero, but
the maximum absolute loss difference is also `1.242e-3`. More importantly, Qwen
does not clear the 1.01 throughput gate and both models exceed the predeclared
parameter-Max gate. Qwen also exceeds the parameter-RMS gate.

Only peak memory passes for both models. The Autograd/CLI model route and its
candidate runners are removed. The CPU/HIP/PyTorch-aligned
`bf16_weight_gradient` operator, six-shape benchmark, stepwise loss output and
complete safetensors comparison remain reusable.

`trajectory.jsonl` retains all 240 loss values and performance records. Multi-GB
parameter snapshots are intentionally absent after complete comparison.

