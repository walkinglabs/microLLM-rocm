# 128-thread causal-softmax counterexample

The candidate changes only the cooperative row block from 256 to 128 threads
for sequences 256 through 1024. Dynamic reduction stride makes both powers of
two valid while preserving the default 256-thread tree.

Complete CPU/HIP output gates pass, but the three-process operator matrix clears
the 1.01 speed gate in only four of six rows. DeepSeek T512 reaches 1.0071x.
The planned model experiment is therefore cancelled, the CLI/model route is
removed, and only an explicit operator implementation remains.

See [Experiment 207](../optimization-log/experiments/207-causal-softmax-128-discard.md).
