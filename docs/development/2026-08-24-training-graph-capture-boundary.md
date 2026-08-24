# 2026-08-24 — 训练 HIP Graph 为什么还不能直接打开

## 先用一句话说明

现在的框架会认真告诉你“这次训练还不能安全录成HIP Graph”，而不会录下一段地址会变化、
步数不会变化的错误程序。

## 把HIP Graph想成录像

假设教室里有四张桌子，每张桌子上放一个Tensor。老师把学生从桌子A拿数据、在桌子B计算、
最后写到桌子C的过程录下来。只要桌子一直在原位，录像可以重复播放。

如果第二次播放前桌子被搬走，录像里的“去桌子B”就变成了错误地址。训练forward和backward
目前会在执行途中不断创建临时Tensor，就像边录像边搬进新桌子。因此完整训练不能只在外面
套一层`capture()`。

## 本次增加了什么

### 1. 分配保护

`HipGraphExecutable::capture`在当前线程设置一个很小的内部标记。普通HIP Tensor如果在
capture中申请新Storage，runtime会在调用`hipMalloc`之前抛出清楚的错误：先在capture外面
准备稳定输出和workspace。

这很重要，因为实际MI300X/ROCm环境显示，直接让同步分配进入capture会让Stream变成
Invalidated；后面的错误清理会再次失败，让人误以为问题出在同步。

### 2. 恢复测试

测试故意在capture里创建Tensor，确认它被拒绝。然后在同一条Stream上捕获预分配的
`add_out`、重放并检查四个结果。这样证明保护不仅给出好看的错误文字，也真的保住了Stream。

### 3. 四阶段探针

新benchmark接受：

```bash
microllm_bench_training_graph_capture \
  --precision fp32 \
  --stage forward
```

`--stage`可以是`forward`、`backward`、`optimizer`或`full-step`；`--precision`可以是
`fp32`或`bf16`。输出JSON会写明capture是否成功、节点数、恢复状态、延迟释放规模，以及
optimizer重放是否推进主机step。

### 4. 重复矩阵

Python runner为八个组合各启动三个新进程，并在偶数轮反转顺序，避免“总是先跑某个阶段”
带来的初始化偏差。CPU contract test用假probe检查24行是否齐全、summary是否拒绝错误结论。

## 实测结果

- 24/24进程完成，capture恢复失败为0；
- FP32/BF16 forward、backward和full-step全部在第一次动态Storage处安全停止；
- FP32/BF16 AdamW各捕获21个设备节点；
- capture后的主机step为1，Graph重放后仍为1。

最后一点说明AdamW还不是一个完整可重放的训练状态。bias correction、学习率schedule和step
必须变成设备拥有的稳定状态，或者每次重放前用Graph参数更新机制明确更新。

## 下一版设计条件

完整训练Graph至少需要：

1. 先跑一次liveness规划，知道每个临时Tensor从什么时候活到什么时候；
2. 在capture外申请一个或多个稳定arena；
3. forward/backward尽量使用`*_out`接口写入规划好的地址；
4. Autograd不能每步重建拥有新Storage的节点结果；
5. optimizer step、bias correction和学习率必须在重放时更新；
6. capture、eager fallback和checkpoint恢复必须得到相同loss与参数。

在这些条件满足前，节点数只能证明GPU工作被录到，不能证明训练语义正确。

## 发布验证

| Gate | Result |
|---|---:|
| CPU Debug | 330/330 |
| ASan/UBSan | 328/328 |
| PyTorch-enabled CPU | 304/304 |
| CPU + HIP full configuration | 518/518，含3个条件跳过 |
| MI300X/gfx942 HIP label | 176/176 |
| RCCL label | 14/14 |
| multi-GPU label | 12/12 |
| 注册测试文件 | 93 |
| CPU coverage | 79.8% lines，87.7% functions，60.4% branches |
