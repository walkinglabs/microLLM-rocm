# 2026-08-25：保序Score物化矩阵工具

## 测什么

`cached_attention_materialized_matrix.py`固定DeepSeek H12/KV2/D128，覆盖T512/T2048、B1/B2、
FP32/BF16 cache。每格3个新进程、3次热身、20次正式测量，总计24条raw。

每个进程在同一个binary中测current fused和materialized-score，forward/reverse交替。必须报告：

- complete context与current逐元素位级相同；
- current/materialized Event和wall P50/P95；
- materialized score bytes；
- 每次2个逻辑allocation和对应cache reuse；
- 热身后0 backend allocation；
- 计时区间0 H2D/D2H。

每格Event至少1.05x才通过算子门。运行器保存`raw.jsonl`、`summary.json`和`comparison.svg`，不会
因为某一格很快而隐藏另一格反例。

## 已跑的pilot

T2048/B2/BF16得到current 0.2730ms、materialized 0.1560ms，即1.750x；196,608-byte score，
两次逻辑allocation，0热backend allocation，完整context位级相同。它只证明值得跑完整矩阵。

伪设备合同覆盖8格×2进程，验证16条raw、位级门、4/3倍聚合、allocation和SVG。下一提交才运行
真实24进程并决定是否进入官方模型。
