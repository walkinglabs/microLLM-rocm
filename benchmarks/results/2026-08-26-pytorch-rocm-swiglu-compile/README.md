# torch.compile SwiGLU rejection

Eight fresh MI300X processes rotate native eager, custom eager, custom compiled and manual fused
FP32 F+B at 64K/1M. The runner applies the explicit AMDSMI-zero fallback only when HIP runtime
reports a device, because this pre-release Torch build otherwise exposes `is_available=True` with
an empty device-property cache.

Compilation succeeds after the fallback, but steady state regresses: compiled/eager Event is
`0.584×–0.610×`, compiled/native is `0.462×–0.476×`, and manual/compiled is
`7.696×–8.635×`. Median cold start is 1160.3 ms for the first shape and 55.8 ms for the second.
Gradient Max stays `4.77e-7`; compiled sum changes reduction order at 1M and records an explicit
loss delta of `0.00390625`.

The recommendation is rejected. An opaque Custom Op remains a dispatch boundary, so compilation
does not fuse away its callback/submission cost. Raw and summary artifacts are retained.

