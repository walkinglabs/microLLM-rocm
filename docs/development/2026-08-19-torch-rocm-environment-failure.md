# 2026-08-19 — PyTorch ROCm environment failure

The AMD gfx94X staging index offers Torch `2.11.0+rocm7.13.0rc2`, matching the local
ROCm development runtime. It was installed into an isolated temporary Python 3.13
environment. Importing `torch` terminated with a Bus error before Torch version/device
queries or the microLLM binding build could run.

This failure does not test the Custom Op ROCm source. It establishes only that this
temporary wheel environment is unusable. The already isolated Torch 2.13 CPU wheel
continues to compile and pass dispatcher add/multiply tests.

Required follow-up: use an AMD-supported PyTorch ROCm environment where a basic
`import torch; torch.cuda.is_available()` succeeds, then build `TorchOps.Basic` and run
the non-default current-stream case.
