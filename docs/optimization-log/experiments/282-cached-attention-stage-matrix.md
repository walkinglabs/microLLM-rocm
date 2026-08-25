# Experiment 282：把Attention拆开以后，真正能下什么结论

Status: measured diagnostic baseline; admit split-sequence fused candidate

## 方法

DeepSeek形状固定H12/KV2/D128，覆盖T512/T2048、B1/B2和FP32/BF16 cache。每格3个新进程，
交替forward/reverse测量顺序；每个进程先热身3次，再用HIP Event和host wall各测20次。

每次运行都比较完整score、probability、context、透明pipeline和当前fused输出。24条raw全部为
0 payload H2D/D2H、0 warm backend allocation。

![Cached Attention stage timing](../../../benchmarks/results/2026-08-25-cached-attention-stage-matrix/stage-timing.svg)

## 结果

| T | B | cache | score ms | softmax ms | context ms | pipeline ms | fused ms | fused加速 |
|---:|---:|---|---:|---:|---:|---:|---:|---:|
| 512 | 1 | BF16 | 0.0235 | 0.1975 | 0.0676 | 0.2876 | 0.0731 | 3.932x |
| 512 | 1 | FP32 | 0.0369 | 0.1965 | 0.0668 | 0.3001 | 0.0968 | 3.100x |
| 512 | 2 | BF16 | 0.0238 | 0.2095 | 0.0680 | 0.3028 | 0.0732 | 4.135x |
| 512 | 2 | FP32 | 0.0369 | 0.2100 | 0.0670 | 0.3144 | 0.0962 | 3.269x |
| 2048 | 1 | BF16 | 0.0257 | 0.7587 | 0.2581 | 1.0480 | 0.2735 | 3.831x |
| 2048 | 1 | FP32 | 0.0375 | 0.7505 | 0.2549 | 1.0576 | 0.3712 | 2.849x |
| 2048 | 2 | BF16 | 0.0365 | 0.8258 | 0.2602 | 1.1349 | 0.2730 | 4.157x |
| 2048 | 2 | FP32 | 0.0402 | 0.8176 | 0.2559 | 1.1408 | 0.4187 | 2.724x |

透明pipeline的softmax占65.46%–73.56%，T2048为71.96%–73.56%。但这不能写成“fused
Kernel内部softmax也占73%”：fused不写global score/probability，而且整体快2.72x–4.16x，执行
结构不同。矩阵能证明的是generic三段基线很差，以及正式fused仍是下一优化对象。

BF16 fused比FP32快1.313x–1.534x，说明cache流量仍重要。三进程范围很小：pipeline最大1.18%，
fused最大0.77%。完整误差最大值依次为score 0、probability 9.31e-10、context/pipeline
1.83e-9、fused 3.73e-9。

## 下一条可反驳假设

当前fused每个`batch × head`只有一个block：B1/B2只有12/24 blocks，而设备有304个CU。旧实验
已经拒绝缩线程、shared query、BF16 pair Key/Value和额外归一化；不再排列这些标量写法。

下一候选把一个head的sequence切给多个blocks。每个block写局部max、denominator和D128加权值，
第二个Kernel用log-sum-exp合并。更多blocks可能提高T2048占用率；反例是T512或B2上第二次launch和
partial buffer可能更贵。候选必须完整context对齐且至少1.05x，才有资格进模型。

证据：[`cached Attention stage matrix`](../../../benchmarks/results/2026-08-25-cached-attention-stage-matrix/)
