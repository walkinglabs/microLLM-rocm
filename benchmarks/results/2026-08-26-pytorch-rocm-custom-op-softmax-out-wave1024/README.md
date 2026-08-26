# Caller-owned Custom Op after BF16 wave1024

This is the same six-process native-out/Custom-out matrix after enabling the scoped
BF16 1024-thread wave path. All 10 precision/pointer rows and zero-peak gates pass.

- BF16 width4096 Custom Event falls from 8.758 to 5.191 μs, about 1.687×;
- native-out ratio improves from 0.467× to 0.804×;
- FP16 width4096 remains 0.821×;
- width1024 FP16/BF16 remains above native out at 1.086×/1.085×.

The result is scoped evidence for BF16 cached wide rows, not a general wave policy.
