# Exact matmul registry key evidence

`cpu-key.log`检查host环境、逻辑M/K/N、dtype、转置、stride、mode与workspace，并验证无效key、
Auto实现和非连续Tensor被拒绝。

`hip-isolation.log`只注册FP32 NN 64³，然后证明FP16、TT、Training mode和不同workspace不会命中。
它还检查gfx architecture、HIP runtime/driver和hipBLASLt版本都进入真实key。

这只是进程内exact registry的证据。持久cache、自动候选计时与correctness-before-timing仍是下一
节点，不能把本目录解释成自动调优已经完成。
