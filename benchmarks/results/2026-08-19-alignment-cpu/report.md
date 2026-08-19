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
| operator | `matmul` | 8 | 0.006394 | 0.003258 | 0.5095 |
| operator | `matmul` | 6 | 0.0061515 | 0.0045225 | 0.7352 |
| operator | `matmul` | 7 | 0.0060885 | 0.002906 | 0.4773 |
| operator | `matmul` | 3 | 0.004431 | 0.0106485 | 2.403 |
| operator | `matmul` | 0 | 0.004353 | 0.003363 | 0.7726 |
| operator | `matmul` | 5 | 0.0042855 | 0.013141 | 3.066 |
| operator | `matmul` | 4 | 0.004228 | 0.018862 | 4.461 |
| operator | `matmul` | 9 | 0.0042265 | 0.0033725 | 0.7979 |
| operator | `swiglu` | 0 | 0.0037895 | 0.009504 | 2.508 |
| operator | `matmul` | 2 | 0.003183 | 0.002884 | 0.9061 |
| operator | `repeat_interleave` | 1 | 0.0031245 | 0.0052585 | 1.683 |
| operator | `matmul` | 1 | 0.00312 | 0.002969 | 0.9516 |
| operator | `repeat_interleave` | 0 | 0.003095 | 0.009885 | 3.194 |
| operator | `add` | 1 | 0.003009 | 0.0027875 | 0.9264 |
| operator | `embedding` | 0 | 0.0029755 | 0.020134 | 6.767 |
| operator | `rope` | 0 | 0.002846 | 0.105463 | 37.06 |
| operator | `add` | 0 | 0.0028145 | 0.0065755 | 2.336 |
| operator | `rms_norm` | 2 | 0.0027465 | 0.019276 | 7.018 |
| operator | `rms_norm` | 1 | 0.002649 | 0.051533 | 19.45 |
| operator | `rms_norm` | 0 | 0.0026235 | 0.01724 | 6.571 |
| operator | `causal_softmax` | 0 | 0.002577 | 20.3782 | 7908 |
| operator | `rope` | 1 | 0.0022735 | 0.0588455 | 25.88 |
| operator | `scale` | 0 | 0.002266 | 0.0070865 | 3.127 |
| operator | `contiguous` | 0 | 0.002219 | 0.008922 | 4.021 |
| operator | `reshape` | 2 | 0.0007885 | 0.0021925 | 2.781 |
| operator | `reshape` | 3 | 0.000611 | 0.002084 | 3.411 |
| operator | `reshape` | 1 | 0.00061 | 0.002656 | 4.354 |
| operator | `transpose` | 3 | 0.000598 | 0.0018675 | 3.123 |
| operator | `transpose` | 2 | 0.0005885 | 0.0017135 | 2.912 |
| operator | `transpose` | 1 | 0.0005785 | 0.0020235 | 3.498 |

## Layer and model timing

| Kind | Name | # | microLLM median ms | PyTorch median ms | PyTorch/microLLM |
|---|---|---:|---:|---:|---:|
| model | `model.forward` | 0 | 0.158591 | 20.8477 | 131.5 |
| layer | `model.blocks.0` | 0 | 0.136479 | 20.7709 | 152.2 |
| layer | `model.embedding` | 0 | 0.004969 | 0.020312 | 4.088 |
| layer | `model.final_norm` | 0 | 0.004224 | 0.0202 | 4.782 |

Positive PyTorch/microLLM values greater than 1 mean microLLM was faster for that measured checkpoint.
