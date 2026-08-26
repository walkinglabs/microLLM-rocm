# C++ Autograd SwiGLU result

The optional adapter now registers SwiGLU's Autograd dispatch in C++ with
`torch::autograd::Function`, while add/multiply keep their lightweight Python formulas. FP32 uses
the existing general/scalar-seed fused HIP producers. FP16/BF16 use explicit no-grad, in-place ATen
formulas to avoid Python callback and intermediate-lifetime inflation. FakeTensor pointerless
execution returns the Meta-shaped output so the forward fullgraph contract remains valid.

Six fresh processes repeat the 15-case matrix. FP32 forward+backward reaches `1.144×` native Torch
at 64K and `1.136×` at 1M, with only 1,536 bytes measured temporary peak. FP16/BF16 improve from
the Python path's roughly `0.61×–0.65×` to `0.77×–0.81×`; their measured peak now equals native.
All dtype-specific loss/gradient gates pass.

`comparison.json` records the complete Python→C++ speed, peak and native gates. C++ Autograd is the
recommended default adapter path. Low-precision fused backward remains a separate opportunity.

