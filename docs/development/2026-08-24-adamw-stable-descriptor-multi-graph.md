# 2026-08-24 — 两个 HIP Graph 节点怎样更新 256 个 Tensor

## 用一张点名册理解descriptor

一个AdamW Kernel要知道参数、gradient、两份moment和可选BF16 mirror放在哪里。普通实现每次
调用时直接把这些地址作为Kernel参数。multi-tensor实现则先做一张点名册：每一行记录一个
Tensor的地址和大小，GPU根据block编号查表。

上一版虽然有device step，Graph里仍然录了N个更新Kernel。本版在capture前把点名册上传一次，
之后不再改它。Graph只录：

1. 一个step/correction Kernel；
2. 一个读取点名册的multi-tensor Kernel。

无论点名册有1行还是256行，Graph节点数都是2。

## 为什么点名册必须不可修改

Graph保存真实地址。如果重放期间又上传一张新点名册，旧Kernel可能正在读同一块device buffer，
结果会出现竞态。因此，`adamw_prepare_multi_graph_`只能成功一次；prepare后的workspace也不能
交给普通multi更新入口。要改变地址，必须销毁旧Graph/workspace并重新准备。

## preparation与timed region分开

preparation负责：

- 检查所有shape、dtype、device和contiguous条件；
- 检查moment是统一的FP32或BF16；
- 保存gradient为null还是稳定地址；
- 上传descriptor并等待完成。

正式计时只包含Graph replay。所有90个进程都报告timed H2D/D2H/D2D为0，因此10×–36×收益
不是把descriptor copy藏到计时外后每步继续复制。

## 数值怎样检查

真实HIP单测用两个不同大小Tensor同时检查FP32/BF16 moment，连续重放三次后比较参数、两份
moment和mirror。正式matrix又执行53步，并把per-tensor Graph和multi Graph分别与eager的四组
sample比较。最大误差为`7.45e-8`。

## 性能怎样读

64/256个小Tensor时，一次大grid避免几十到几百次短Kernel提交，因此FP32/BF16都达到约
10×/36×。但一个Tensor没有合并空间；FP32 16个256K Tensor仍略慢，说明大FP32更新已经由
真实内存流量主导。

这不是“AdamW整体快36倍”。它是固定的synthetic optimizer phase，用来证明提交结构。真实
训练还包含forward、backward、数据和同步，必须另外做端到端门。

## 进入真实训练前还缺什么

descriptor记住gradient地址。eager Autograd可能每步产生新的gradient Storage。下一实验必须
记录同一参数跨两次backward的gradient地址、大小和独占状态：

- 地址稳定：可以尝试只捕获optimizer phase；
- 地址变化：先建立stable gradient buffer或Graph参数更新合同；
- 任何一种都要重新比较loss、参数、optimizer时间和端到端tokens/s。

## 发布验证

CPU 332/332、ASan/UBSan 330/330、PyTorch-enabled CPU 306/306、完整CPU+HIP
524/524（3个条件跳过）、HIP标签180/180、RCCL 14/14、multi-GPU 12/12。覆盖清单注册
95个测试文件；CPU覆盖率为78.5% lines、86.8% functions、59.2% branches。新增实现属于HIP
descriptor与Graph路径，CPU分母增加而覆盖率下降；对应真实MI300X数值、节点和transfer门均已运行。
