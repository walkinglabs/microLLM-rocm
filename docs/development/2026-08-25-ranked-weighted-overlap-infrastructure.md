# 2026-08-25 — ranked token-weighted overlap infrastructure

## 这次解决什么

两张卡处理的有效token数量不一样时，不能直接平均各卡的“平均梯度”。较短batch必须先乘较
小权重，较长batch必须先乘较大权重。旧同步路径会等整张图反向完成后再缩放全部梯度；旧重叠
路径却会在某个bucket刚准备好时立刻通信，两种顺序原来不能安全组合。

本节点把顺序固定为：

```text
某个叶子梯度完成
→ 在默认Stream上乘本rank权重
→ 标记这个叶子ready
→ bucket最后一个叶子完成后记录Event
→ RCCL Stream等待Event、打包并all-reduce
```

Event像一张“前面的作业已经做完”的小票。先缩放、后开小票，通信Stream读到的就一定是已经
加权的梯度。

## 代码和可观察证据

- `apps/distributed_rank.cpp`不再拒绝`token-weighted + overlap-views`。
- 第1步仍使用同步路径建立固定bucket view；第2步开始才启用ready overlap。
- ready hook捕获对应parameter，先调用`scale_in_place_`，再调用
  `mark_parameter_ready`。
- worker新增总计数`weighted_gradient_scales`和逐步计数
  `step_weighted_gradient_scales`。
- launcher按每个rank的真实token数计算期望值；某个rank的scale恰好是1时，期望Kernel数为0。
- matrix runner保留每个rank的计数，并拒绝漏缩放或重复缩放。
- CTest新增`DistributedRank.WeightedOverlapSmoke`，真实启动两个进程、两张GPU和CPU
  global-batch reference。

## 本节点验证

Tiny `[B1,B2]`、T4、三步、4 KiB bucket：

- 每步叶子scale：`[12,12,12]`；
- overlap启用：`[0,1,1]`；
- overlap bucket：`[0,1,1]`；
- 后两步reducer backend allocation：0；
- rank最大差/RMS：0/0；
- CPU最大差/RMS：`8.18e-8 / 8.79e-9`；
- 加权平均loss最大差：`1.94e-7`；
- `DistributedRank.*`：8/8；
- 完整RCCL标签：50/50；
- 测试文件覆盖审计：125个文件全部登记。

Tiny smoke只证明顺序和数值正确，不能证明Model-S更快。性能结论必须等下一提交在干净revision
上交替运行同步与重叠策略后再写。
