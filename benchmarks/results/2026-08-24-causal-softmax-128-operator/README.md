# 128-thread causal-softmax operator matrix

Three fresh processes per case/policy, with three warm-up and twenty measured
HIP Event iterations, compare the existing 256-thread Auto route against the
explicit 128-thread primitive.

| Family | T256 | T512 | T1024 |
|---|---:|---:|---:|
| Qwen heads=14 | 1.0168x | 1.0255x | 1.0127x |
| DeepSeek heads=12 | 1.0063x | 1.0071x | 1.0214x |

All complete outputs pass with maximum error `1.86e-9`; no timed payload
transfers occur. Only 4/6 rows clear the fixed 1.01 gate, including a DeepSeek
T512 failure, so the model gate is deliberately not run. Auto remains at 256
threads and the 128-thread path is retained only as an explicit research primitive.
