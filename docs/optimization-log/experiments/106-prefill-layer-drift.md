# Experiment 106 — embedding exact，差异从block 0开始累积

本节点用相同P5构造B1和`[P5,P5]` B2，完整捕获31个stage。临时大trace不进入性能数据；仓库保存
三对fresh process的完整误差统计和逐层summary。

![Prefill layer drift](../assets/prefill-layer-drift.svg)

## 合同

- B1/B2权重、tokens、dtype和binary相同；
- 每个stage值必须完整，禁止truncated trace；
- B2 row0/row1每层必须exact；
- 三个fresh pair误差统计必须相同；
- 保存max/mean/RMS/relative-L2和max位置；
- host snapshot不用于吞吐。

## 结果

31个stage、3/3 pair通过。embedding的B1/B2和B2两行都逐值一致。首个非零stage是block 0：
max-abs 0.0013504、relative-L2 0.00005166。

误差随后总体累积。block 27为hidden最大点：max 1.900269、mean 0.075833、RMS 0.102217、
relative-L2 0.006261。final norm将绝对值缩小，但relative-L2升到0.008412。完整151936维logits为：

```text
max-abs      0.15301609
mean-abs     0.02892810
RMS          0.03405911
relative-L2  0.01377723
```

B2 duplicate rows在全部31个stage的max-abs均为0，反驳了“batch内两行互相污染”。最终max-abs仍在
仓库官方BF16 0.2门内，但Experiment 104的0.000669 margin说明token exact不能由这个容差保证。

## 决策

保留inference layer trace和完整值比较runner。当前不回退batch prefill，也不修改argmax。下一最小
问题是block 0内部哪一步首次非零，因此下一节点只给第一层Attention/FFN加诊断边界。

数据见[`106-data`](106-data/)。
