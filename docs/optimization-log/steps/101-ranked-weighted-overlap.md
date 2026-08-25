# Step 101 — Ranked token-weighted ready overlap

Status: implemented; formal Model-S measurement pending

当前weighted同步路径在backward完成后统一scale；overlap会更早pack，所以两者不能直接组合。下一
节点将每个leaf gradient的local scale放入gradient-ready hook，并在scale Kernel之后记录bucket
Event。这样通信Stream看到的每个gradient已经weighted。

第1步同步建立view plan；后续step hook执行`scale -> mark ready -> Event -> pack/RCCL`。必须证明
每参数只scale一次、3个bucket顺序不变、rank/CPU/loss与同步weighted一致、later allocation0。
Model-S `[B1,B2]`至少三步，端到端不过门则拒绝性能路由但保留正确性原语。

实现门已通过：第1步仍在backward后统一scale并建立view plan；第2步起，每个leaf的
ready hook先发出scale Kernel，再让bucket记录Event。worker逐步输出
`step_weighted_gradient_scales`，因此测试能直接拒绝漏scale或重复scale，而不只看最终参数。

两张MI300X上的Tiny三步smoke得到scale `[12,12,12]`、overlap `[0,1,1]`、rank差0，
CPU Max/RMS为`8.18e-8 / 8.79e-9`。完整RCCL标签50/50。下一提交从干净revision对
Model-S T128的同步`bucket-views`与weighted `overlap-views`交替测量三轮。
