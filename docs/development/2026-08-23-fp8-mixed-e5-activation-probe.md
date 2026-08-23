# Mixed E5M2 activation and E4M3 weight probe

E4M3 has more mantissa precision; E5M2 has much wider range. Dynamic Tensor-amax prevents E4M3
overflow, so E5M2 is not assumed better. Before adding a model switch, the operator path must prove
that MI300 and the installed hipBLASLt execute the mixed pair.

CPU evidence quantizes activations as E5M2-FNUZ and weights as E4M3-FNUZ, then proves
`fp8_matmul` equals an explicit dequantize-plus-matmul reference. It also verifies the corrected
E5M2 dynamic scale uses 57,344 rather than E4M3's 240.

The 128x128 MI300 probe records:

```text
mixed_e5e4_native_shapes = 1
mixed_e5e4_fallback_calls = 0
hot H2D/D2H = 0/0
```

The result is an executed capability, not a model precision claim. The next node may expose E5M2
activations while keeping E4M3 weights and the retained O-projection scope. Official complete-logit
comparison decides whether the extra exponent bit is worth losing one mantissa bit.
