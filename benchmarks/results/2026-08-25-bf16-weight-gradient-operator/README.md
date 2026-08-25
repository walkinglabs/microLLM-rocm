# Cast-inclusive BF16 weight-gradient matrix

Experiment 245 measures `input^T @ output_gradient` for six real B1T512
Qwen/DeepSeek Linear shapes. Candidate Event time includes input cast+transpose,
output-gradient cast and BF16×BF16→FP32 GEMM.

| Model | Family | Shape K×M×N | Median speedup | Minimum | Decision |
|---|---|---:|---:|---:|---|
| Qwen | query | 896×512×896 | 0.718× | 0.718× | reject |
| Qwen | KV | 896×512×128 | 0.821× | 0.813× | reject |
| Qwen | gate/up | 896×512×4864 | 1.459× | 1.446× | model gate |
| DeepSeek | query | 1536×512×1536 | 0.976× | 0.965× | reject |
| DeepSeek | KV | 1536×512×256 | 0.816× | 0.813× | reject |
| DeepSeek | gate/up | 1536×512×8960 | 1.890× | 1.823× | model gate |

All 18 processes produce finite complete outputs. Deterministic BF16 CPU sample
Max error is at most `5.22e-8`. Complete-output FP32-baseline Max/RMS is retained
as the deliberate precision change, not hidden behind a loose equality claim.

The public `bf16_weight_gradient` operator has CPU, HIP and PyTorch BF16-semantic
alignment. The Autograd model route is explicit and default-off; only gate/up can
use it. Query/KV counterexamples prevent a universal low-precision gradient policy.

