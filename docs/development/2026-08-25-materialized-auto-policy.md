# 2026-08-25：Materialized-score有界自动策略

## 为什么不是全局默认

Experiment 287只证明MI300X/gfx942、BF16 KV、Qwen/DeepSeek两种head签名与uniform cached decode。
Radeon、FP32 KV、其他模型结构和divergent-row serving没有证据，因此不能简单把Model默认改成true。

## auto条件

`microllm_hf_infer`未显式传开关时，只有同时满足以下条件才配置materialized策略：

- HIP设备架构以`gfx942`开头；
- KV cache为BF16；
- 使用KV cache；
- 没有显式partial-split策略；
- 模型为已测head签名之一：H14/KV2/D64或H12/KV2/D128。

即使auto启用，Model也只在当前uniform prefix达到2048后真正走新算子。T512继续使用旧fused路径。
positions-aware路径本身没有接入该算子。

## 用户控制和可观察性

```bash
# 强制开，并允许从T512开始
--cached-attention-materialized true \
--cached-attention-minimum-sequence 512

# 强制关，用于回归或其他硬件
--cached-attention-materialized false
```

JSON输出：

- `cached_attention_materialized_scores`：Model最终是否配置新策略；
- `cached_attention_materialized_minimum_sequence`：运行时阈值；
- `cached_attention_materialized_policy`：`auto-enabled`、`auto-bypass`、`explicit-on`或`explicit-off`；
- `cached_attention_materialized_auto_eligible`：auto条件是否全满足；
- `cached_attention_materialized_measured_head_signature`：head签名是否在证据范围。

CPU fixture验证auto-bypass；CLI合同验证显式开关、互斥和输出身份。下一提交使用官方gfx942/BF16模型
在不传开关时验证auto-enabled与T2048性能，并用显式false作为current对照。

实现smoke已用官方Qwen T2048/B1/BF16、不传策略开关运行：JSON报告`auto-enabled`、eligible=true、
measured-head=true、minimum=2048并生成有效token。CPU fixture报告`auto-bypass`。正式默认性能复测仍在
下一节点，不能用这个N1 smoke代替三对结果。

成对runner的`candidate_policy=auto`会让current显式传false，而candidate完全不传开关；合同测试
检查candidate报告`auto-enabled`。这避免用“显式true的结果”替代“默认行为的结果”。
