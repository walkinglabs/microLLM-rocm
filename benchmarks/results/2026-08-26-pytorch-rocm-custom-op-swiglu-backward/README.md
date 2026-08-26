# SwiGLU backward vector4 rejection

Six fresh MI300X processes compare the retained scalar fused producer, an aligned float4
candidate, and a readable native PyTorch formula at 4K/64K/1M/16M FP32 elements. Every row checks
both complete gradients; maximum error versus native is `1.19e-7`, and vector versus scalar is at
most `2.98e-8`.

Vector/scalar Event medians are `0.971×`, `1.039×`, `1.003×`, and `0.946×`: no 1M/16M row clears
the 1.05 admission gate. The vector implementation, public research selector, dedicated test and
runner were removed after measurement. Raw evidence and the schema summary remain here.

The more useful result is that scalar fused backward is already `2.07×–2.82×` faster than the
readable native formula and uses one-third less measured peak. Therefore the slow end-to-end
Autograd path is not explained by scalar backward arithmetic. The next experiment targets the
forced materialization of `sum()`'s zero-stride output gradient.

