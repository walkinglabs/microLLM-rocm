# Complete-model BF16 FFN Arena

## 实现

- 每个模型按`(device, flattened rows, hidden, intermediate)`缓存一个entry；
- 一个owned Storage包含input/gate/up/activated/fallback/output全部slice；
- 所有block按default Stream顺序复用同一entry，不按层复制workspace；
- `set_bf16_ffn_arena_enabled`默认关闭，并公开entry/hit/miss/capacity统计；
- `model.to()`清空旧device cache；详细value trace继续走原diagnostics路径；
- CLI增加`--bf16-ffn-arena`与机器可读统计；
- 60进程官方模型runner覆盖T32/T512、B1/B4和cached decode。

## 结果

所有完整logits bit-exact，生成token完全一致。Arena减少20%–28%左右的测量期分配，
但只有3/10行超过1.01；Qwen/DeepSeek T512为1.022×/1.020×，Qwen B1 decode为
1.031×。其余接近1.0。全局策略拒绝并保持默认关闭；下一步只允许验证`rows>=512`。

完整报告：[Experiment 183](../optimization-log/experiments/183-bf16-ffn-arena-model.md)。
