# Step 101 — Ranked token-weighted ready overlap

Status: planned

当前weighted同步路径在backward完成后统一scale；overlap会更早pack，所以两者不能直接组合。下一
节点将每个leaf gradient的local scale放入gradient-ready hook，并在scale Kernel之后记录bucket
Event。这样通信Stream看到的每个gradient已经weighted。

第1步同步建立view plan；后续step hook执行`scale -> mark ready -> Event -> pack/RCCL`。必须证明
每参数只scale一次、3个bucket顺序不变、rank/CPU/loss与同步weighted一致、later allocation0。
Model-S `[B1,B2]`至少三步，端到端不过门则拒绝性能路由但保留正确性原语。
