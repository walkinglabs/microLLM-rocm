# Ranked Model-S checkpoint smoke

Experiment 273 runs from clean revision `526c95b`. Two ranks train Model-S T32
for one step and publish a rank0-only checkpoint. Fresh ranks restore for one
more step; a control group trains uninterrupted for two steps.

All interrupted, resumed-final and uninterrupted-final checkpoints are
187,042,096 bytes. The final checkpoint bytes are exactly equal. Every successful
group compares all 57 tensors and 15,586,176 values across ranks with Max/RMS zero.

| Measurement | Result |
|---|---:|
| rank0 write | 1,022–1,068 ms |
| maximum rank1 wait | 1,069 ms |
| maximum checkpoint verify/read | 532 ms |
| maximum load + restore on a rank | 740 ms |

Only rank0 writes. The shared ownership failure gate remains the tiny injected
case: rank0 returns 1 and its waiting peer is terminated with -15. This avoids
creating a deliberately failed 187 MB payload while exercising the same barrier,
write and marker layer.

All checkpoint, safetensors, ready, temporary and communicator files are deleted.
These single-run numbers are resource evidence for the current storage environment,
not a portable I/O benchmark.
