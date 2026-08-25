# 2026-08-25：Cached Attention分段计时工具

## 这次解决什么

上一个节点能分别检查score、probability和context，但还不能公平回答“时间到底花在哪一步”。
这次增加一个HIP单算子基准和一个矩阵运行器，把正确性检查、热身、计时、汇总和画图放进同一条
可重复路径。

可以把融合Attention想成一个不透明的饭盒。诊断接口把饭盒分成Q·K、softmax、P·V三个格子；
这个工具给每个格子各放一只秒表，同时保留整个融合饭盒的秒表。

## 固定合同

`microllm_bench_cached_attention_stages`每次只跑一个shape：

- query为FP32 `[B,12,1,128]`；
- key/value cache为FP32或BF16 `[B,2,T,128]`；
- 独立测score、softmax、context、三段pipeline和当前fused；
- 每个输出与CPU完整Tensor比较，不抽样；
- 至少热身3次，正式测量默认20次；
- 同时报告HIP Event P50/P95和host wall P50/P95；
- 测量区间H2D/D2H必须为0；
- 热身后backend allocation必须为0，并报告每次逻辑allocation/cache reuse；
- 支持forward/reverse两种测量顺序，避免某个方法永远先跑。

`cached_attention_stage_matrix.py`默认运行T512/T2048、B1/B2、FP32/BF16 cache，每格3个新进程，
交替测量顺序。它保存`raw.jsonl`、聚合后的`summary.json`和不依赖第三方绘图库的
`stage-timing.svg`。原始数据始终保留，图不是证据的唯一来源。

## 已验证

- Python合同测试用16条伪设备记录覆盖8格矩阵、顺序轮换、聚合和SVG；
- 缺字段的伪记录会稳定失败；
- MI300X T32/B1/BF16 smoke通过完整五路径精度门；
- smoke测量区间为0 payload transfer、0 warm backend allocation；
- C++目标只在HIP构建中出现，CPU仍能验证矩阵结果合同。

提交前完整回归为CPU 373/373、ASan/UBSan 371/371、PyTorch-enabled CPU 376/376和
MI300X HIP label 192/192。

这只是测量基础设施，不是速度结论。下一提交才运行24个真实新进程，公开原始结果、图、主要
热点和T2048反例，再决定第一个Kernel候选。
