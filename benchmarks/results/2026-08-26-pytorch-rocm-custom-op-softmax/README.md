# Initial C++ PyTorch Custom Op Softmax

This six-process FP16/BF16 matrix records the first functional C++ Custom Op. It passes
all 10 precision/output/peak gates, but the Autograd dispatch kernel unconditionally
entered `Function::apply` even when the input did not require gradients.

FP16 width4096 measured 6.640 μs Event and only 0.700× native Torch. This result is the
control for the retained inference gate; it is not the current adapter performance.
`raw.jsonl` and `summary.json` preserve every process.
