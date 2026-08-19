# microLLM / PyTorch alignment report

- Run ID: `2026-08-19-alignment-hip`
- Status: **pass**
- Value checkpoints: 45
- Passed checkpoints: 45
- Failed or missing checkpoints: 0
- Tolerance: `atol=0.002`, `rtol=0.002`

## Largest numerical differences

| Kind | Name | # | Status | Max abs | Max rel | Cosine |
|---|---|---:|---|---:|---:|---:|
| operator | `matmul` | 3 | pass | 9.53674e-07 | 1.95559e-06 | 1.00000000 |
| operator | `swiglu` | 0 | pass | 5.96046e-07 | 2.55467e-06 | 1.00000000 |
| layer | `model.blocks.0` | 0 | pass | 4.76837e-07 | 5.39734e-07 | 1.00000000 |
| operator | `add` | 1 | pass | 4.76837e-07 | 5.39734e-07 | 1.00000000 |
| operator | `matmul` | 7 | pass | 4.76837e-07 | 2.59424e-06 | 1.00000000 |
| operator | `matmul` | 8 | pass | 4.76837e-07 | 2.86075e-06 | 1.00000000 |
| operator | `reshape` | 7 | pass | 4.76837e-07 | 2.86075e-06 | 1.00000000 |
| operator | `scale` | 0 | pass | 4.76837e-07 | 1.95559e-06 | 1.00000000 |
| operator | `contiguous` | 0 | pass | 3.57628e-07 | 4.24932e-06 | 1.00000000 |
| operator | `matmul` | 4 | pass | 3.57628e-07 | 4.24932e-06 | 1.00000000 |
| operator | `reshape` | 4 | pass | 3.57628e-07 | 4.24932e-06 | 1.00000000 |
| operator | `transpose` | 4 | pass | 3.57628e-07 | 4.24932e-06 | 1.00000000 |
| model | `model.forward` | 0 | pass | 2.38419e-07 | 7.69773e-07 | 1.00000000 |
| operator | `matmul` | 0 | pass | 2.38419e-07 | 4.54047e-07 | 1.00000000 |
| operator | `matmul` | 2 | pass | 2.38419e-07 | 3.75469e-07 | 1.00000000 |
| operator | `matmul` | 6 | pass | 2.38419e-07 | 7.92952e-07 | 1.00000000 |
| operator | `matmul` | 9 | pass | 2.38419e-07 | 7.69773e-07 | 1.00000000 |
| operator | `repeat_interleave` | 1 | pass | 2.38419e-07 | 3.75469e-07 | 1.00000000 |
| operator | `reshape` | 0 | pass | 2.38419e-07 | 1.7924e-07 | 1.00000000 |
| operator | `reshape` | 1 | pass | 2.38419e-07 | 4.54047e-07 | 1.00000000 |

## Operator timing

| Kind | Name | # | microLLM median ms | PyTorch median ms | PyTorch/microLLM |
|---|---|---:|---:|---:|---:|
| operator | `matmul` | 8 | 0.020828 | 0.0029245 | 0.1404 |
| operator | `rms_norm` | 1 | 0.020736 | 0.0521755 | 2.516 |
| operator | `rms_norm` | 2 | 0.020035 | 0.018347 | 0.9157 |
| operator | `matmul` | 6 | 0.0199955 | 0.004522 | 0.2262 |
| operator | `causal_softmax` | 0 | 0.0199455 | 25.9325 | 1300 |
| operator | `matmul` | 0 | 0.0192245 | 0.003168 | 0.1648 |
| operator | `matmul` | 1 | 0.018885 | 0.00287 | 0.152 |
| operator | `rms_norm` | 0 | 0.018781 | 0.0167995 | 0.8945 |
| operator | `rope` | 0 | 0.0187635 | 0.105764 | 5.637 |
| operator | `matmul` | 5 | 0.0186235 | 0.013294 | 0.7138 |
| operator | `matmul` | 7 | 0.0185555 | 0.0028525 | 0.1537 |
| operator | `repeat_interleave` | 0 | 0.0184485 | 0.0094855 | 0.5142 |
| operator | `contiguous` | 0 | 0.018054 | 0.0093305 | 0.5168 |
| operator | `matmul` | 9 | 0.0178305 | 0.0030235 | 0.1696 |
| operator | `matmul` | 3 | 0.017592 | 0.011517 | 0.6547 |
| operator | `swiglu` | 0 | 0.0174065 | 0.0100385 | 0.5767 |
| operator | `add` | 1 | 0.0173355 | 0.002834 | 0.1635 |
| operator | `rope` | 1 | 0.017318 | 0.058107 | 3.355 |
| operator | `scale` | 0 | 0.017184 | 0.007132 | 0.415 |
| operator | `matmul` | 2 | 0.0171675 | 0.0027635 | 0.161 |
| operator | `matmul` | 4 | 0.0171415 | 0.0183195 | 1.069 |
| operator | `embedding` | 0 | 0.0170525 | 0.019764 | 1.159 |
| operator | `repeat_interleave` | 1 | 0.017016 | 0.005191 | 0.3051 |
| operator | `add` | 0 | 0.0148865 | 0.0067895 | 0.4561 |
| operator | `transpose` | 0 | 0.000366 | 0.0028575 | 7.807 |
| operator | `reshape` | 1 | 0.00027 | 0.0026385 | 9.772 |
| operator | `reshape` | 4 | 0.000269 | 0.0035205 | 13.09 |
| operator | `reshape` | 7 | 0.000261 | 0.002485 | 9.521 |
| operator | `transpose` | 1 | 0.0002555 | 0.0019975 | 7.818 |
| operator | `reshape` | 2 | 0.000254 | 0.002109 | 8.303 |

## Layer and model timing

| Kind | Name | # | microLLM median ms | PyTorch median ms | PyTorch/microLLM |
|---|---|---:|---:|---:|---:|
| model | `model.forward` | 0 | 0.2529 | 27.9275 | 110.4 |
| layer | `model.blocks.0` | 0 | 0.193864 | 27.8652 | 143.7 |
| layer | `model.final_norm` | 0 | 0.0230945 | 0.0170085 | 0.7365 |
| layer | `model.embedding` | 0 | 0.017521 | 0.0186185 | 1.063 |

Positive PyTorch/microLLM values greater than 1 mean microLLM was faster for that measured checkpoint.
