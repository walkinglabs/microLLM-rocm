# microLLM / PyTorch alignment report

- Run ID: `tiny-alignment-cpu`
- Status: **pass**
- Value checkpoints: 58
- Passed checkpoints: 58
- Failed or missing checkpoints: 0
- Tolerance: `atol=3e-05`, `rtol=3e-05`

## Largest numerical differences

| Kind | Name | # | Status | Max abs | Max rel | Cosine |
|---|---|---:|---|---:|---:|---:|
| parameter | `gradient.token_embedding.weight` | 0 | pass | 1.43051e-06 | 7.59777e-05 | 1.00000000 |
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

## Operator timing

| Kind | Name | # | microLLM median ms | PyTorch median ms | PyTorch/microLLM |
|---|---|---:|---:|---:|---:|
| operator | `matmul` | 8 | 0.007206 | 0.003019 | 0.419 |
| operator | `matmul` | 6 | 0.0068885 | 0.0042505 | 0.617 |
| operator | `matmul` | 7 | 0.0068635 | 0.002923 | 0.4259 |
| operator | `matmul` | 0 | 0.00473 | 0.003221 | 0.681 |
| operator | `matmul` | 5 | 0.0046225 | 0.0093035 | 2.013 |
| operator | `matmul` | 4 | 0.004617 | 0.0130735 | 2.832 |
| operator | `matmul` | 9 | 0.004577 | 0.0029265 | 0.6394 |
| operator | `matmul` | 3 | 0.004556 | 0.010597 | 2.326 |
| operator | `swiglu` | 0 | 0.0043315 | 0.008565 | 1.977 |
| operator | `matmul` | 1 | 0.0036815 | 0.002816 | 0.7649 |
| operator | `matmul` | 2 | 0.003509 | 0.0027795 | 0.7921 |
| operator | `embedding` | 0 | 0.003308 | 0.019966 | 6.036 |
| operator | `repeat_interleave` | 1 | 0.003293 | 0.0050815 | 1.543 |
| operator | `repeat_interleave` | 0 | 0.003164 | 0.009014 | 2.849 |
| operator | `add` | 0 | 0.003156 | 0.006481 | 2.054 |
| operator | `add` | 1 | 0.0031085 | 0.002747 | 0.8837 |
| operator | `rope` | 0 | 0.0030155 | 0.0965235 | 32.01 |
| operator | `rms_norm` | 1 | 0.002935 | 0.040563 | 13.82 |
| operator | `rms_norm` | 0 | 0.0028985 | 0.0168775 | 5.823 |
| operator | `rms_norm` | 2 | 0.002874 | 0.016876 | 5.872 |
| operator | `causal_softmax` | 0 | 0.0027335 | 16.3758 | 5991 |
| operator | `scale` | 0 | 0.0024315 | 0.0066605 | 2.739 |
| operator | `rope` | 1 | 0.0023285 | 0.0556665 | 23.91 |
| operator | `contiguous` | 0 | 0.0022945 | 0.0068495 | 2.985 |
| operator | `reshape` | 2 | 0.000762 | 0.002055 | 2.697 |
| operator | `reshape` | 3 | 0.0006615 | 0.002006 | 3.033 |
| operator | `reshape` | 1 | 0.000647 | 0.0023995 | 3.709 |
| operator | `reshape` | 5 | 0.000636 | 0.0024915 | 3.917 |
| operator | `transpose` | 1 | 0.00063 | 0.0019 | 3.016 |
| operator | `reshape` | 4 | 0.000627 | 0.0027085 | 4.32 |

## Layer and model timing

| Kind | Name | # | microLLM median ms | PyTorch median ms | PyTorch/microLLM |
|---|---|---:|---:|---:|---:|
| model | `model.forward` | 0 | 0.159135 | 17.1808 | 108 |
| layer | `model.blocks.0` | 0 | 0.136464 | 17.124 | 125.5 |
| layer | `model.embedding` | 0 | 0.005 | 0.013326 | 2.665 |
| layer | `model.final_norm` | 0 | 0.00411 | 0.01327 | 3.229 |

## Backward timing

| Kind | Name | # | microLLM median ms | PyTorch median ms | PyTorch/microLLM |
|---|---|---:|---:|---:|---:|
| model | `model.backward` | 0 | 0.267479 | 25.3006 | 94.59 |

Positive PyTorch/microLLM values greater than 1 mean microLLM was faster for that measured checkpoint.
