# 请求不是同时来的：怎样测试排队和尾延迟

## 为什么“8 条请求一起开始”还不够

前面的矩阵把 8 条请求同时提交。这很适合比较算子和 batch，但真实服务更像食堂排队：有人先来，
有人后到；短请求和长请求的数量也不会总是刚好填满每个桶。

长度分桶第一版没有 slot stealing。于是可能出现：

```text
小桶：4 个 slot 全满，门外还有 2 条短请求
大桶：4 个 slot 只用了 2 个
```

整台 GPU 仍有空位，但短请求不能借大桶。平均吞吐可能看不出问题，排队请求的 TTFT p95 却会
明显变差。

## arrival step 是什么

CLI 可以为每条请求指定逻辑到达步：

```text
--continuous-arrival-steps 0,0,0,0,4,4,4,4
```

含义是前四条在逻辑 step 0 提交，后四条在 step 4 提交。调度循环是：

```text
找出本 step 到达的请求
→ 调用 scheduler.submit
→ 如果有活跃请求，执行一次 scheduler.step
→ 逻辑时钟加一
```

TTFT 和 completion 从真正调用 `submit` 的时刻开始。因此第 4 步才到达的请求不会把“尚未来到
系统”的时间算进延迟，但进入 scheduler 后的排队会计算在内。

## 它不是什么

arrival step 不是固定毫秒定时器。统一 B8 的一步与两个 B4 桶的一步包含不同 GPU 工作量，所以
“step 4”在两种 policy 中不保证发生于同一个 wall time。

这个实验回答的是状态机问题：已有请求运行若干轮后，新请求怎样进入、排队和完成。若要模拟
每秒固定到达率，还需要独立 wall-clock load generator；当前结果不能冒充真实 QPS 服务压测。

## 为什么同时看 P50 和 P95

假设 6 条短请求里 4 条立即进入小桶，2 条排队：

- P50 主要看前 4 条，可能仍很好看；
- P95 会接近排队最久的请求，能暴露没有 slot stealing 的代价。

因此 `traffic-skew` 不只报告所有请求的 P50/P95，还对受影响的 focus indices 单独计算：

- focus TTFT P50/P95；
- focus completion P50/P95；
- bucketed/uniform 比值；
- 每条请求的原始延迟数组。

## 三组固定反例

```text
short_heavy  6 短 + 2 长，小桶可能排队
long_heavy   2 短 + 6 长，大桶可能排队
delayed      4 短先来，4 长在 step 4 到达
```

每组都运行统一 B8 与两个 B4 桶。两边请求 token、输出长度、总 slot、dtype 和权重完全相同。

## 性能环境门

每个 fresh process 前后，runner 读取指定物理 GPU：

```text
--physical-gpu-index 3
--max-idle-vram-percent 5
--max-idle-use-percent 10
```

任一边界超标就立即停止，当前进程不会写进 raw。这个门来自真实失败：一次运行选择 GPU 时为
0% VRAM，但第一个进程结束时外部作业已经占用 61%。没有门时，程序仍会返回 `pass`，却不能
作为性能证据。

门只能检查进程边界。很短、恰好在进程中间开始又结束的外部负载仍可能漏过，因此正式结果还要
检查三进程波动，异常时保留失败而不是挑最快值。

## 使用示例

```bash
HIP_VISIBLE_DEVICES=3 python3 benchmarks/single_gpu/hf_continuous_matrix.py \
  --manifest /path/to/models.json \
  --binary build-hipblaslt/apps/microllm_hf_infer \
  --output-directory /tmp/traffic-skew \
  --suite traffic-skew \
  --warmup 1 --steps 3 --runs 3 \
  --physical-gpu-index 3 \
  --max-idle-vram-percent 5 \
  --max-idle-use-percent 10
```

正式矩阵未通过设备门时，状态只能写 `gate_blocked / no_measurement`，不能从空 raw 推断 policy
好坏。

正式 36 进程最终在空闲 GPU2 完成。short/long-heavy 中，两个固定桶的 focus TTFT P50 看起来
更好，但 P95 约为统一池的 3 倍；delayed 场景则全面小幅退化。因此默认仍是统一池，不能只凭
median 自动启用分桶。完整原始数组见
[Experiment 116](../optimization-log/experiments/116-traffic-skew.md)。
