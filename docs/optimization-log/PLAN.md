# Optimization roadmap: 0.1917× → selected-matrix parity

## 总路线

```text
M0 可信基线       0.1917×  已完成
M1 删除串行热点   CE / RMSNorm / transpose（完成）
M2 删除搬运抖动   KV Cache / sampling / allocator（完成）
M3 算子体系成熟   batched GEMM / plan / fusion / FMHA（进行中）
M4 低精度         BF16 → FP8
M5 系统调度       HIP Graph / overlap / persistent cache
M6 固定矩阵验收   正确性 + 吞吐 + 显存 + 失败图集
```

## 里程碑定义

| 里程碑 | 状态 | 主要产物 | 通过条件 |
|---|---|---|---|
| M0 baseline | complete | 多步 microLLM/PyTorch raw JSONL | 4/4 workload 可比 |
| M1 serial kernels | complete | parallel CE、transpose GEMM、parallel RMSNorm | 三组旧热点均从 trace 主导位置消失 |
| M2 data movement | complete | preallocated KV、device greedy、steady-state exact-size pool | measured decode 只回传 token scalar |
| M3 optimized ops | in progress | batched GEMM、hipBLASLt plan、FMHA/fusion | Model-S/M 与 HF 均改善 |
| M4 low precision | in progress | streaming load；batched backward；saved-prob T512 tradeoff | flash-style row/forward；multi-shard preflight |
| M5 scheduling | in progress | serving reference；static/admission batch；HIP Graph；slot refill | 请求语义不变且跨group活跃slot扩展 |
| M6 report | in progress | 博客、曲线、trace、失败图集、局部饱和审计 | 新 track 仍待完成 |

## 步骤索引

| Step | 状态 | 主题 | 首要指标 |
|---:|---|---|---|
| [00](steps/00-baseline.md) | complete | 固定基线与 profiler | score 0.191660 |
| [01](steps/01-parallel-cross-entropy.md) | complete | 并行 CE forward/backward | Qwen train 3.29× |
| [02](steps/02-transpose-aware-gemm.md) | complete | 无复制 transpose GEMM | score 0.479227 |
| [03](steps/03-parallel-rmsnorm.md) | complete | block-parallel RMSNorm | score 0.885816 |
| [04](steps/04-device-kv-cache.md) | complete | 预分配 device KV/GQA | score 1.167931 |
| [05](steps/05-device-sampling.md) | complete | device greedy argmax；随机 top-k 保留 reference | score 1.219170 |
| [06](steps/06-memory-pool.md) | complete | exact-size pool + phase-independent default-Stream reuse | score 2.470863 + Experiment 087 inference evidence |
| [07](steps/07-autograd-buffers.md) | complete | generic COW失败；source-aware独占累加也因整机中性而默认关闭 | 下一步必须做graph-wide liveness |
| [08](steps/08-batched-fmha.md) | complete | cached/full GQA；Q/K RoPE；BTHD P×V/dP/dV | diagnosed layout copies = 0 |
| [09](steps/09-fusion-autotune.md) | complete | plan/alpha/repeat/zero-stride families measured | local Attention search saturated |
| [10](steps/10-bf16.md) | in progress | DeepSeek first-four robust-strict；Qwen long-constant要求全FP32 fallback | broader checkpoints and prompt families |
| [11](steps/11-fp8.md) | planned | cached FP8 weight/dynamic scale | FP8 track |
| [12](steps/12-hip-graph-final.md) | complete | exact FP32 QK/PV registry与整模反驳 | 24进程bit-exact；最高1.009×，默认关闭 |
| [13](steps/13-bf16-grouped-qkv.md) | complete | pointer-stable BF16 GroupedGemm QKV | operator 1.881×/1.225×；model 1.032×/1.001× |
| [14](steps/14-bf16-grouped-qkv-expanded.md) | complete | 64候选 + device user arguments | steady 1.046×/1.030×；setup约204ms |
| [15](steps/15-bf16-grouped-qkv-prewarm.md) | complete | serving前显式prewarm | 首请求比lazy快892/947ms；总启动不变 |
| [16](steps/16-hipblaslt-preload.md) | complete | 全kernel预载反例 | 首forward变慢3.417×/3.447×，策略拒绝 |
| [17](steps/17-bf16-exact-startup.md) | complete | 精确gate/up冷启动门 | 算子快但cold 0.990×/0.996×，策略拒绝 |
| [18](steps/18-bf16-grouped-gate-up.md) | complete | 双gate/up GroupedGemm能力 | device arguments 1.188×/1.155× |
| [19](steps/19-bf16-grouped-gate-up-model.md) | complete | FFN Arena稳定plan接入 | 整模1.0176×/1.0117×；少24/28提交 |
| [20](steps/20-bf16-grouped-composition.md) | complete | QKV + gate/up组合 | both/base 1.0655×/1.0474× |

## 为什么按这个顺序

baseline trace 已经证明：

- 训练 CE forward/backward 占约 75.7% Kernel 时间；
- 推理 tied transpose + RMSNorm 占约 81.1%；
- Qwen 推理有 7407 次 allocation、7403 次 free；
- AdamW 只占 microLLM Qwen 训练 Kernel 时间约 1.5%；
- GEMM 不是当前最大热点。

因此不能先做 FP8、HIP Graph 或复杂手写 GEMM。前置复制和串行 reduction 不消失，
后续优化的结论会被噪声掩盖。

## 阶段目标不是承诺

图中的 0.35、0.50、0.75、1.00 是研究检查点，不是预先声称可以达到的结果。
每个点必须来自实际 `results.tsv`。失败实验同样保留。
