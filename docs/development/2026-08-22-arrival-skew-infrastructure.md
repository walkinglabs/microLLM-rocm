# 2026-08-22：延迟到达、偏斜流量与设备门

## 新问题

1/2/4 桶矩阵使用完美均分、同时到达的 8 请求，无法证明固定桶在真实偏斜流量下不会排队。

## 本次只增加的能力

- `--continuous-arrival-steps`：每条请求在指定逻辑 step 调用 `submit`；
- `traffic-skew`：短请求偏斜、长请求偏斜和 step-4 延迟到达；
- focus-request TTFT/completion P50 与 P95；
- `--physical-gpu-index`、VRAM/use pre/post gate；
- raw 保存每进程的 pre/post GPU 状态。

arrival step 是状态机时间，不是固定毫秒或固定 QPS。请求 latency 从真实 submit 开始。

## 验证

- Python runner/shape/idle-gate 合同 14/14；
- Qwen 真实 CLI smoke 使用 arrival `[0,2]`，两请求均正确生成，scheduler 共 4 步；
- ASan/UBSan 214/214；
- 81% VRAM 和 75% use 的模拟拒绝测试。

## 第一次正式尝试为什么没有结果

GPU 3 选择时为 `1% use / 0% VRAM`。首个 Qwen 进程 pre gate 通过，post gate 发现外部作业已
占 `61% VRAM`，runner 立即退出。`raw.jsonl` 为 0 行，没有 summary，也没有换卡挑数据。

状态是 `gate_blocked / no_measurement`。正式性能结论必须等到稳定空闲窗口，不能用接口 smoke
代替。
