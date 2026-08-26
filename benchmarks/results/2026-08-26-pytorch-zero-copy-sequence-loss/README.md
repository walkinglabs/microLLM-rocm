# PyTorch零复制RoPE、Embedding与CrossEntropy

三个seed共36条完整PyTorch ROCm对照：RoPE 12、Embedding 12、CrossEntropy 12。

- RoPE覆盖B1/B2、T1/7/17/64、D4–32、offset 0/5/11/17和两种base；
- Embedding覆盖词表7–1024、宽度3–128、一维/二维indices和重复index；
- CrossEntropy覆盖1–64行、7–4096类与ignore-index；
- loss显式提供caller-owned scalar output和`[rows,2]` reduction workspace。

| 算子 | 最大Max | 最大RMS | Max门使用率 |
|---|---:|---:|---:|
| RoPE | 4.77e-7 | 4.37e-8 | 1.59% |
| Embedding | 0 | 0 | 0% |
| CrossEntropy | 9.54e-7 | 9.54e-7 | 3.18% |

108/108指针一致且non-owning，约5.14MiB payload，wrapper复制0字节。

![Sequence/loss matrix](sequence-loss-matrix.svg)

CrossEntropy workspace是公共契约，不在C API内部偷偷分配。CPU路径同样验证workspace shape，虽然
参考循环不使用payload；这样切换到HIP reduction不会突然改变所有权。

当前仍没有混合进程rocprof性能结论。这个节点证明前向数值/布局/所有权，不证明训练梯度零复制。
