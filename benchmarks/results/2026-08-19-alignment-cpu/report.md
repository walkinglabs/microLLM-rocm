# microLLM / PyTorch alignment report

- Run ID: `2026-08-19-alignment-cpu`
- Status: **pass**
- Value checkpoints: 45
- Passed checkpoints: 45
- Failed or missing checkpoints: 0
- Tolerance: `atol=3e-05`, `rtol=3e-05`

## Largest numerical differences

| Kind | Name | # | Status | Max abs | Max rel | Cosine |
|---|---|---:|---|---:|---:|---:|
| operator | `swiglu` | 0 | pass | 8.34465e-07 | 3.00431e-06 | 1.00000000 |
| layer | `model.blocks.0` | 0 | pass | 5.96046e-07 | 5.39734e-07 | 1.00000000 |
| operator | `add` | 1 | pass | 5.96046e-07 | 5.39734e-07 | 1.00000000 |
| operator | `matmul` | 8 | pass | 5.96046e-07 | 3.81433e-06 | 1.00000000 |
| operator | `reshape` | 7 | pass | 5.96046e-07 | 3.81433e-06 | 1.00000000 |
| operator | `matmul` | 3 | pass | 4.76837e-07 | 3.47661e-06 | 1.00000000 |
| operator | `matmul` | 6 | pass | 4.76837e-07 | 3.02764e-06 | 1.00000000 |
| operator | `matmul` | 7 | pass | 4.76837e-07 | 2.01499e-06 | 1.00000000 |
| layer | `model.final_norm` | 0 | pass | 3.57628e-07 | 4.65582e-07 | 1.00000000 |
| model | `model.forward` | 0 | pass | 3.57628e-07 | 6.80368e-07 | 1.00000000 |
| operator | `matmul` | 9 | pass | 3.57628e-07 | 6.80368e-07 | 1.00000000 |
| operator | `reshape` | 8 | pass | 3.57628e-07 | 4.65582e-07 | 1.00000000 |
| operator | `reshape` | 9 | pass | 3.57628e-07 | 6.80368e-07 | 1.00000000 |
| operator | `rms_norm` | 2 | pass | 3.57628e-07 | 4.65582e-07 | 1.00000000 |
| output | `model.logits` | 0 | pass | 3.57628e-07 | 6.80368e-07 | 1.00000000 |
| operator | `contiguous` | 0 | pass | 2.38419e-07 | 4.33973e-06 | 1.00000000 |
| operator | `matmul` | 1 | pass | 2.38419e-07 | 4.08961e-07 | 1.00000000 |
| operator | `matmul` | 4 | pass | 2.38419e-07 | 4.33973e-06 | 1.00000000 |
| operator | `repeat_interleave` | 0 | pass | 2.38419e-07 | 3.43674e-07 | 1.00000000 |
| operator | `reshape` | 2 | pass | 2.38419e-07 | 4.08961e-07 | 1.00000000 |

## Operator timing

| Kind | Name | # | microLLM median ms | PyTorch median ms | PyTorch/microLLM |
|---|---|---:|---:|---:|---:|
| operator | `matmul` | 8 | 0.0064825 | 0.003214 | 0.4958 |
| operator | `matmul` | 6 | 0.006273 | 0.004486 | 0.7151 |
| operator | `matmul` | 7 | 0.0061605 | 0.0028215 | 0.458 |
| operator | `matmul` | 0 | 0.004475 | 0.0032815 | 0.7333 |
| operator | `matmul` | 5 | 0.004367 | 0.010134 | 2.321 |
| operator | `matmul` | 9 | 0.0042615 | 0.00316 | 0.7415 |
| operator | `matmul` | 4 | 0.0042535 | 0.0136855 | 3.217 |
| operator | `matmul` | 3 | 0.0042005 | 0.0109385 | 2.604 |
| operator | `swiglu` | 0 | 0.0038505 | 0.0090725 | 2.356 |
| operator | `matmul` | 1 | 0.003224 | 0.0028975 | 0.8987 |
| operator | `matmul` | 2 | 0.0032 | 0.0028325 | 0.8852 |
| operator | `repeat_interleave` | 1 | 0.003137 | 0.005048 | 1.609 |
| operator | `repeat_interleave` | 0 | 0.0030675 | 0.009248 | 3.015 |
| operator | `add` | 1 | 0.003005 | 0.0026615 | 0.8857 |
| operator | `embedding` | 0 | 0.0029585 | 0.0190805 | 6.449 |
| operator | `rope` | 0 | 0.0029025 | 0.101278 | 34.89 |
| operator | `add` | 0 | 0.0028275 | 0.005008 | 1.771 |
| operator | `rms_norm` | 2 | 0.0027665 | 0.018187 | 6.574 |
| operator | `rms_norm` | 1 | 0.0026355 | 0.043976 | 16.69 |
| operator | `rms_norm` | 0 | 0.0026235 | 0.0166885 | 6.361 |
| operator | `causal_softmax` | 0 | 0.0025965 | 17.8667 | 6881 |
| operator | `scale` | 0 | 0.0022575 | 0.0067205 | 2.977 |
| operator | `rope` | 1 | 0.002221 | 0.0567535 | 25.55 |
| operator | `contiguous` | 0 | 0.0022175 | 0.006978 | 3.147 |
| operator | `reshape` | 2 | 0.0007915 | 0.002093 | 2.644 |
| operator | `reshape` | 1 | 0.0006175 | 0.002456 | 3.977 |
| operator | `reshape` | 3 | 0.000602 | 0.0020125 | 3.343 |
| operator | `transpose` | 1 | 0.000583 | 0.0019645 | 3.37 |
| operator | `transpose` | 3 | 0.000579 | 0.0017925 | 3.096 |
| operator | `transpose` | 4 | 0.0005785 | 0.003197 | 5.526 |

## Layer and model timing

| Kind | Name | # | microLLM median ms | PyTorch median ms | PyTorch/microLLM |
|---|---|---:|---:|---:|---:|
| model | `model.forward` | 0 | 0.175438 | 22.2531 | 126.8 |
| layer | `model.blocks.0` | 0 | 0.150972 | 22.1733 | 146.9 |
| layer | `model.embedding` | 0 | 0.005489 | 0.0201535 | 3.672 |
| layer | `model.final_norm` | 0 | 0.0044995 | 0.019869 | 4.416 |

Positive PyTorch/microLLM values greater than 1 mean microLLM was faster for that measured checkpoint.
