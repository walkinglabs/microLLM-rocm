# Experiment 205 — BTHD Attention 直接读取 grouped BF16 Q/K

Status: keep explicit policy; default remains off

## 旧路径与假设

Experiment 204 已经把 Attention 的布局复制归零，但 grouped QKV 仍把 Q、K、V 全部从 BF16
转成 FP32。紧接着，融合 bias+RoPE Kernel 又读取 Q/K。假设是：只让这个融合 Kernel直接读取
BF16 Q/K，V与Attention输出继续保持FP32，就能每层删除两次转换，同时保持完整logits不变。

```text
旧：GroupedGemm BF16 Q/K → FP32 cast → fused bias+RoPE
新：GroupedGemm BF16 Q/K ───────────→ fused bias+RoPE
```

## 安全边界

- 新路径必须同时满足HIP、T≥256、BTHD Attention、BF16 Attention、QKV Arena和精确
  grouped-QKV注册；
- grouped计划未命中时，`bf16_qkv_projection_out_`返回`false`并完整生成原FP32输出；
- V始终转成FP32；cache、训练、trace和普通BHTD路径不进入候选；
- 策略由`--inference-bthd-bf16-qk true`显式打开，默认关闭。

## 正确性与正式测量

![BTHD BF16 Q/K](../assets/inference-bthd-bf16-qk.svg)

MI300X/gfx942、T512、B1、2次热身、5次计时。五个独立进程对按奇偶轮换顺序：

| Model | FP32边界 | BF16 Q/K | 加速 | 完整logits | Peak |
|---|---:|---:|---:|---:|---:|
| Qwen2.5-0.5B | 110,961 tok/s | 113,441 tok/s | 1.0224× | bit-exact | 不变 |
| DeepSeek Distill 1.5B | 56,979 tok/s | 58,333 tok/s | 1.0238× | bit-exact | 不变 |

第一次三进程窗口中，Qwen为1.0227×，DeepSeek只有1.0068×，未通过1.01门。该失败没有被
覆盖；扩大到五进程后两模型均通过。这个差异说明2%左右的小收益必须增加独立进程数，不能
依据单次或三次恰好有利的窗口改变默认策略。

## Kernel trace

每个策略分别profile 1次与6次forward，再用差分除以5，排除加载与plan初始化：

| Model | cast calls | cast时间 | 总Kernel | Kernel加速 |
|---|---:|---:|---:|---:|
| Qwen | 144→96 | 0.639→0.334 ms | 5.128→4.754 ms | 1.0787× |
| DeepSeek | 168→112 | 0.927→0.514 ms | 9.463→8.928 ms | 1.0600× |

调用数精确减少`2 × block_count`，证明优化确实删除了目标工作，不是只改变计时。BF16输入
RoPE本身也没有更慢：Qwen 0.216→0.184 ms，DeepSeek 0.323→0.317 ms。

## 决策

保留显式策略和完整回退机制，但不改默认值。原因不是结果失败，而是当前只验证了一个MI300X
环境、T512/B1与两套精确solution。下一步优先扩展sequence/batch矩阵；若跨shape仍稳定，
再讨论把它并入已显式启用的BTHD组合策略。

原始证据：

- [三进程初始门](../../../benchmarks/results/2026-08-24-inference-bthd-bf16-qk/)
- [五进程正式门](../../../benchmarks/results/2026-08-24-inference-bthd-bf16-qk-formal5/)
