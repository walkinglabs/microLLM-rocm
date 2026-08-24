# Selective BF16 FFN Arena

## 实现

- 模型API增加正整数`minimum_rows`；
- cache在`rows<minimum_rows`时返回空，并累计bypassed calls；
- bypass走原`bf16_ffn`，不创建entry或backing；
- CLI增加`--bf16-ffn-arena-minimum-rows`并输出eligible/bypassed/minimum；
- CPU/HIP测试覆盖无效阈值、bypass、eligible、device move与bit-exact；
- 同一60进程官方模型runner支持阈值参数。

## 结果

阈值512时，两模型T512为1.019×/1.022×；八个短case完全bypass，allocation/peak与baseline
逐项相同，速度0.999×–1.005×。60/60完整logits exact，decode token一致。选择策略keep。

完整报告：[Experiment 184](../optimization-log/experiments/184-bf16-ffn-arena-selective.md)。
