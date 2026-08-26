# Rejected Softmax-out Autograd fallthrough

The candidate moves the inference-only requires-grad check into the backend and registers
an Autograd fallthrough for `softmax_out`. Correctness, exact returned pointers and all
zero-peak rows pass on six MI300X processes.

Relative to the retained explicit Autograd kernel at width4096:

- FP16 Custom Event changes 5.043→5.004 μs, about 1.008×;
- BF16 changes 5.191→5.203 μs, about 0.998×.

Neither dtype passes the 1.05 gate. The fallthrough is removed and the centralized
`softmax_out_autograd` rejection remains. This bounds Autograd dispatch as a non-hotspot.
