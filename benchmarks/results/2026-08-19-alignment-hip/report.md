# microLLM / PyTorch alignment report

- Run ID: `tiny-alignment-hip`
- Status: **pass**
- Value checkpoints: 58
- Passed checkpoints: 58
- Failed or missing checkpoints: 0
- Tolerance: `atol=0.002`, `rtol=0.002`

## Largest numerical differences

| Kind | Name | # | Status | Max abs | Max rel | Cosine |
|---|---|---:|---|---:|---:|---:|
| parameter | `gradient.token_embedding.weight` | 0 | pass | 3.33786e-06 | 1.67743e-05 | 1.00000000 |
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

## Operator timing

| Kind | Name | # | microLLM median ms | PyTorch median ms | PyTorch/microLLM |
|---|---|---:|---:|---:|---:|
| operator | `matmul` | 8 | 0.025857 | 0.0021655 | 0.08375 |
| operator | `rms_norm` | 0 | 0.02412 | 0.012904 | 0.535 |
| operator | `rms_norm` | 1 | 0.023728 | 0.046514 | 1.96 |
| operator | `causal_softmax` | 0 | 0.0230705 | 25.3075 | 1097 |
| operator | `repeat_interleave` | 0 | 0.0228985 | 0.008492 | 0.3709 |
| operator | `rms_norm` | 2 | 0.0226155 | 0.0147495 | 0.6522 |
| operator | `matmul` | 6 | 0.022118 | 0.0036425 | 0.1647 |
| operator | `rope` | 0 | 0.021859 | 0.090171 | 4.125 |
| operator | `matmul` | 5 | 0.0209145 | 0.0117315 | 0.5609 |
| operator | `rope` | 1 | 0.0206715 | 0.0418465 | 2.024 |
| operator | `matmul` | 1 | 0.020655 | 0.001992 | 0.09644 |
| operator | `matmul` | 0 | 0.020336 | 0.0023085 | 0.1135 |
| operator | `matmul` | 7 | 0.020313 | 0.001957 | 0.09634 |
| operator | `matmul` | 2 | 0.020281 | 0.0019295 | 0.09514 |
| operator | `matmul` | 9 | 0.020086 | 0.00237 | 0.118 |
| operator | `repeat_interleave` | 1 | 0.019262 | 0.0037715 | 0.1958 |
| operator | `matmul` | 3 | 0.0191905 | 0.0095945 | 0.5 |
| operator | `embedding` | 0 | 0.0189095 | 0.017727 | 0.9375 |
| operator | `contiguous` | 0 | 0.018461 | 0.008171 | 0.4426 |
| operator | `swiglu` | 0 | 0.018197 | 0.0083705 | 0.46 |
| operator | `matmul` | 4 | 0.017791 | 0.0162665 | 0.9143 |
| operator | `add` | 1 | 0.016606 | 0.001944 | 0.1171 |
| operator | `add` | 0 | 0.016584 | 0.005901 | 0.3558 |
| operator | `scale` | 0 | 0.0164055 | 0.0059025 | 0.3598 |
| operator | `transpose` | 0 | 0.0003215 | 0.002285 | 7.107 |
| operator | `reshape` | 4 | 0.0003025 | 0.00281 | 9.289 |
| operator | `reshape` | 1 | 0.000283 | 0.0021135 | 7.468 |
| operator | `transpose` | 1 | 0.0002685 | 0.001516 | 5.646 |
| operator | `reshape` | 6 | 0.000268 | 0.001435 | 5.354 |
| operator | `reshape` | 8 | 0.0002675 | 0.0011435 | 4.275 |

## Layer and model timing

| Kind | Name | # | microLLM median ms | PyTorch median ms | PyTorch/microLLM |
|---|---|---:|---:|---:|---:|
| model | `model.forward` | 0 | 0.304914 | 33.2697 | 109.1 |
| layer | `model.blocks.0` | 0 | 0.219614 | 33.1989 | 151.2 |
| layer | `model.final_norm` | 0 | 0.026087 | 0.0195735 | 0.7503 |
| layer | `model.embedding` | 0 | 0.018585 | 0.019008 | 1.023 |

## Backward timing

| Kind | Name | # | microLLM median ms | PyTorch median ms | PyTorch/microLLM |
|---|---|---:|---:|---:|---:|
| model | `model.backward` | 0 | 0.650085 | 45.634 | 70.2 |

Positive PyTorch/microLLM values greater than 1 mean microLLM was faster for that measured checkpoint.
