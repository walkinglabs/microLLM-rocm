# 2026-08-23：偏斜流量下固定桶的尾延迟失败

## 环境恢复

前一天第一次运行在 post gate 发现 61% VRAM，raw 为 0；第二次监控 180 次仍无空闲卡。
本次 physical GPU2 连续三次、间隔10秒均为 `0% use / 0% VRAM`，才启动正式矩阵。

36个fresh process中：pre VRAM最大1%、use最大2%；post VRAM最大2%、use最大5%。

## 结果

- short-heavy：两个B4桶吞吐约为uniform的73%；TTFT P50约0.30×，但P95约3.2×；
- long-heavy：吞吐约0.57×，TTFT P95约3×；
- delayed：吞吐约0.94×，TTFT/completion均退化约7%–9%；
- Qwen/DeepSeek六组token全部exact。

## 解释

中位数只看见先进入桶的请求，P95看见被固定容量挡在门外的请求。错误不在Kernel或KV数值，
而在没有work stealing的静态资源分配。

## 下一合同

只允许实现“短请求在最小桶已满时进入兼容的大桶”。必须满足：

- short-heavy focus TTFT/completion P95下降；
- token exact；
- delayed不越5%回退门；
- long-heavy继续作为反例，因为长请求不能装进小桶；
- 默认uniform策略不改变。

完整证据见[Experiment 116](../optimization-log/experiments/116-traffic-skew.md)。
