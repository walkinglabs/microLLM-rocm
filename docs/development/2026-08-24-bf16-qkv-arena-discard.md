# BF16 QKV Arena model experiment

## 实现

- `Bf16QkvWorkspace`保存一次input cast和Q/K/V各自fallback；
- `bf16_qkv_projection_out_`写三个caller-owned FP32 output并检查全部别名；
- model QKV cache按exact rows/device缓存单backing、跨block复用；
- API/CLI有minimum、entry/hit/miss/eligible/bypass/capacity；
- Experiment 184 FFN selective作为baseline，QKV只测增量；
- CPU/HIP算子、完整model logits、CLI和runner均覆盖。

## 结果

Qwen/DeepSeek T512只达到1.004×/1.005×，未过1.01。Profiler证明malloc/free继续下降，
Kernel/launch不变。60/60 logits exact，但一个完全bypass的Qwen decode聚合为0.976×。
模型策略拒绝、默认关闭；caller-owned算子和显式诊断开关保留。

完整报告：[Experiment 185](../optimization-log/experiments/185-bf16-qkv-arena-discard.md)。
