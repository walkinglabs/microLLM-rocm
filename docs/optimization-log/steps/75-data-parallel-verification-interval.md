# Step 75 — Separate parameter verification from the hot path

Status: in progress

给DataParallelConfig增加默认1的parameter-check interval；0显式关闭，N每N步检查。Metrics新增
`parameter_check_performed`与`verification_ms`，默认行为和参数等价测试不变。CLI/runner记录
interval。先在tiny双卡测0/1/N，证明收益来自host审计而非通信，再进入bucket readiness。

