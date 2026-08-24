# BTHD BF16 Q/K initial gate and profile

The initial three-process T512 window is retained because DeepSeek reached only
`1.0068x`, below the `1.01x` keep gate. Qwen reached `1.0227x`; both models were
bit-exact with unchanged engine peak.

The phase-differential profile uses separate one-forward and six-forward
processes for both policies and models. It removes setup/load by subtraction:

| Model | Cast calls | Cast time | Total Kernel speedup |
|---|---:|---:|---:|
| Qwen | 144 -> 96 | 0.639 -> 0.334 ms | 1.0787x |
| DeepSeek | 168 -> 112 | 0.927 -> 0.514 ms | 1.0600x |

The exact 48/56-call reduction proves that two Q/K casts per block disappeared.
Because the end-to-end gain is small relative to process variation, the formal
gate was expanded to five processes rather than accepting profiler time alone.

Files: `raw.jsonl`, `summary.json`, `profile-summary.json`, and per-policy
phase-delta CSV/JSON under `profile/`.
