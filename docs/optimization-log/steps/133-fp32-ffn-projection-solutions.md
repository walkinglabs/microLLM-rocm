# Step 133 — FP32 FFN gate/up row-invariant solutions

Status: planned

Experiment 316证明FFN norm exact，而gate和up两个独立投影都从同一输入开始漂移。它们共享真实
descriptor family：

```text
M = 2048 / 4096 / 8192 / 16384
K = 1536
N = 8960
transpose = NN
dtype = FP32
```

下一节点枚举四个M在当前gfx942、ROCm和hipBLASLt环境下的共同solution。每个候选顺序通过：支持与
workspace合同 → 完整CPU/default reference → 相同row跨M bitwise → 同M重复row bitwise → HIP Event/wall。

只有在四个M都不低于0.95×的exact候选才进入gate/up独立scope的完整模型反驳。index保持version-local
和显式；若没有候选同时通过，就关闭vendor-solution路线，转向一个可读保序FP32 projection Kernel。
