# FP32 gate/up weight-gradient solution evidence

Experiment 219 screens exact hipBLASLt solution indices for the FP32 gate/up weight-gradient GEMM
at rows 512.

## Operator screen

Three fresh tuner processes per model evaluate 64 complete outputs each (384 evaluations total).

| Model | Shape M×K×N | Selected common index | Median | Minimum |
|---|---|---:|---:|---:|
| Qwen | 896×512×4864 | 289155 | 1.077× | 1.070× |
| DeepSeek | 1536×512×8960 | 284846 | 1.133× | 1.114× |

All candidates are finite and pass Max `1e-4` / RMS `1e-5` complete-output gates.

## Model rebuttal

The explicit CLI seam registers one exact rank-2 key. It proves 144/168 hits for one warm-up plus
two measured steps. Three fresh processes per policy/model produce:

| Model | Baseline | Candidate | Speedup | Peak | Decision |
|---|---:|---:|---:|---:|---|
| Qwen | 15,493.55 | 15,377.81 tok/s | 0.993× | 1.000× | reject |
| DeepSeek | 6,488.55 | 6,461.88 tok/s | 0.996× | 1.000× | reject |

`raw.jsonl` and `summary.json` contain the operator screen. `model-pilot/` contains the 12 model
processes. No solution is installed as a default; the CLI flag remains an explicit diagnostic seam.
`verification.json` records the complete release gates.
