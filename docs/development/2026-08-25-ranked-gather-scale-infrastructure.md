# 2026-08-25 — ranked persistent gather-scale infrastructure

## 目标

Step 102每步仍向通信Stream提交57次device copy和3次scale。新候选把一个bucket的多段复制与
local token weight融合成一个Kernel：Model-S每步只提交3次gather-scale。

## 为什么不能永久缓存gradient指针

`optimizer.zero_grad()`会释放Value里的旧gradient，下一次backward可以得到新地址。缓存第一次
地址虽然可能在allocator复用时“看起来能跑”，却不是合同。本实现把固定部分和变化部分分开：

- bucket中的`destination_begin/end`固定；
- 每步bucket ready后重新读取每个leaf的source pointer；
- host描述表持久存在；
- device描述buffer持久存在；
- 每bucket向通信Stream复制一次小描述表，再启动一次Kernel。

Kernel按扁平目标index二分描述表，读取对应source、乘local scale并写入bucket。RCCL随后只做
普通sum与`1/world`。Event、bucket顺序和collective数量不变。

## 可观察计数

新统计包括：

- `gather_scale_calls`；
- `gather_descriptor_copy_calls`；
- `gather_descriptor_bytes`；
- `gather_descriptor_capacity_bytes`；
- 每步对应数组。

`gather-weighted-overlap`还强制`pack_copies=0`、leaf scale=0、bucket scale=0。描述表或Kernel
少一次/多一次都会被worker和launcher拒绝。

## 当前证据

- world1同步gather scale 2与下一步overlap scale 0.5均得到手算结果；
- 第二步换新gradient内容，证明source每步刷新；
- gather没有persistent view时被拒绝；
- Tiny `[B1,B2]`三步：每步1次gather、288-byte描述，rank/CPU门通过；
- Model-S T32：每步3次gather、1,368-byte描述，完整策略参数Max/RMS 0/0；
- Model-S current/peak相对对照增加1,368 bytes；later backend allocation仍为0；
- pilot同步对照CV达到77%，因此4.45x被标记为无效，不形成性能声明。
- `DistributedRank.*` 10/10；完整RCCL标签53/53；测试文件审计125/125。

测量器在中途验证失败时现在也会删除已保留的共识参数文件。下一节点先跑完整RCCL门并推送，
再从干净revision做T128三轮；候选必须改善Step 102的1.0661x和敏感性才保留。
