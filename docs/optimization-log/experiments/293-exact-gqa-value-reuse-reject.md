# Experiment 293：复用六次Value读取，为什么还是慢一半

Status: performance rejected; local exact-finalize search closed

## 唯一允许改变的事

多个GQA query heads共享一个KV head。候选让每个column线程只读一次value，同时更新6或7个独立
accumulator；每个head仍严格按position 0→T累加，因此可以与materialized current位级相同。

![GQA value reuse](../../../benchmarks/results/2026-08-25-cached-attention-gqa-value-reuse/value-reuse.svg)

两模型、T512/T2048、B1/B2、FP32/BF16、tile 8/16/32/64，每格两个fresh process，共128条。

| 指标 | 结果 |
|---|---:|
| 位级context | 128/128 |
| 性能过门 | 0/16 |
| winner Event | 0.4540x–0.6349x |
| winner wall | 0.4695x–0.6637x |
| 目标DeepSeek Event/wall | 0.4978x / 0.5113x |

## 中间失败也保留

首版用运行时`totals[8]`索引，目标只有约0.099x；编译期实例化repeats=1–8后恢复到约0.5x，证明
私有内存spill是一个真实问题。但即使修掉它，先把exact softmax概率写到196,608-byte全局Tensor，
再启动单独P×V Kernel的成本仍大于BF16 value读取复用收益。

## 决定

不进模型，不改Auto。Step 110完成，并关闭exact-finalize局部搜索：

- 改物理线程数：0/16过门；
- 拆P×V：operator快，但模型logits失败；
- 保序GQA value复用：位级正确，但0/16性能过门。

下一步不再排列finalize Kernel。Step 111测当前保留路径的B1/B2/B4/B8 serving batch扩展，回答
能否通过真实并发填充当前每token只有少量head blocks的硬件空闲。

证据：[`GQA value-reuse matrix`](../../../benchmarks/results/2026-08-25-cached-attention-gqa-value-reuse/)
