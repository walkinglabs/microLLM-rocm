# 2026-08-25 — 路线走不通以后，重新看仪表盘

优化不能一直围着自己刚写的代码转。online Attention模型失败后，我们回到当前默认程序，用
rocprof重新记录T1024一次前向。

加载权重和第一次建plan会让时间看起来很大，所以分别记录“加载+1次”和“加载+6次”，相减后
除以5。留下的是普通一次prefill，不是启动成本。

![Current inference profile](../optimization-log/assets/current-inference-profile.svg)

现在矩阵乘占约60%和67%。softmax是最大的单个非矩阵kernel，但旧实验已经证明换线程只能快约
1%，放到整步几乎看不见。所以下一步不凭“红色条很显眼”就修改它，而是检查T1024矩阵库是否
有更适合QK/PV精确shape的算法。

发布时核心源码沿用上一节点已通过的CPU 341、ASan 339、PyTorch 315、完整HIP 537和RCCL
14/14、multi-GPU 12/12；本节点只新增profile runner/schema，四个配置的新增测试均单独通过。
累计注册为CPU 342、ASan 340、PyTorch 316、完整HIP 538，测试文件104个。
