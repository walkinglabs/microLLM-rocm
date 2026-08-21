# Experiment 068 evidence

- `reference/`: same-binary strict mixed policy using the retained per-head prefix copies.
- `paired/`: same binary, same GPU and same protocol, but the one FP32 layer uses a paired Kernel.
- `precision/`: DeepSeek T512 B1 complete-logit control for the paired candidate.
- `comparison.json`: three-process median and discard decision.

The experimental route, Kernel, public switch and tests were removed after the primary prepare and
end-to-end metrics regressed.
