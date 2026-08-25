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
| [21](steps/21-bf16-grouped-shape-matrix.md) | complete | rows256/1024能力矩阵 | 8 case user args 1.124×–1.695× |
| [22](steps/22-bf16-grouped-shape-models.md) | complete | sequence/batch完整模型 | 六case 1.0212×–1.1075× + CLI batch修复 |
| [23](steps/23-bf16-grouped-composed-profile.md) | complete | 组合后phase profile | GEMM calls 217→145、253→169 |
| [24](steps/24-hf-strided-copy-sources.md) | complete | copy source归因 | 100% Attention；100.7/205.5MB |
| [25](steps/25-inference-bthd-attention.md) | complete | 推理BTHD island | copy归零；整模1.1146×/1.0936× |
| [26](steps/26-inference-bthd-shape-models.md) | complete | BTHD sequence/batch矩阵 | 六case 1.0852×–1.1421× |
| [27](steps/27-inference-bthd-profile.md) | complete | BTHD后phase profile | strided=0；下一步BF16 Q/K RoPE |
| [28](steps/28-inference-bthd-bf16-qk.md) | complete | grouped BF16 Q/K直入融合RoPE | 少48/56次cast；T512 1.0224×/1.0238× |
| [29](steps/29-inference-bthd-bf16-qk-shapes.md) | complete | BF16 Q/K sequence/batch矩阵 | 六case 1.0128×–1.0244×；pilot反例保留 |
| [30](steps/30-causal-softmax-128-discard.md) | complete | causal softmax 128线程反例 | 4/6过门；Deep T512 1.0071×，模型策略停止 |
| [31](steps/31-bf16-repeat-fusion-discard.md) | complete | BF16 V cast+repeat融合 | 3/8过1.05；B2 1.004×/0.995×，不接模型 |
| [32](steps/32-post-bf16-qk-saturation.md) | complete | 推理微融合饱和审计 | 两次连续反例；下一步必须online/tiled Attention |
| [33](steps/33-training-add-rms-norm-fusion.md) | complete | residual add + RMSNorm训练融合 | 两模型模型门失败，保留Autograd原语 |
| [34](steps/34-multi-tensor-adamw.md) | complete | descriptor驱动AdamW研究原语 | Qwen有效、DeepSeek未过门，模型路由删除 |
| [35](steps/35-training-bf16-shared-activation.md) | complete | QKV/gate-up共享BF16 cast | 三种模型策略均被反例拒绝 |
| [36](steps/36-post-training-micro-saturation.md) | complete | 训练微融合饱和审计 | GEMM+AdamW占72.71%/83.77% |
| [37](steps/37-bf16-adamw-moments.md) | complete | BF16 optimizer状态带宽/显存 | 端到端与显存过门；Qwen optimizer stretch未过 |
| [38](steps/38-hybrid-bf16-adamw.md) | complete | BF16小Tensor分层合并 | 1M双模型全过；16M DeepSeek反例 |
| [39](steps/39-post-hybrid-training-profile.md) | complete | Hybrid后训练profile | GEMM占59.33%/63.81%，转入GEMM架构 |
| [40](steps/40-grouped-weight-gradient-discard.md) | complete | FP32 grouped weight gradient | 8/8 case无supported candidate，路由未建 |
| [41](steps/41-packed-weight-gradient-discard.md) | complete | packed weight gradient | 0/4过1.05，组合搜索关闭 |
| [42](steps/42-fp32-weight-gradient-solutions-discard.md) | complete | rank-2 exact solution | 算子过门、模型0.993×/0.996×，默认拒绝 |
| [43](steps/43-training-graph-capture-boundary.md) | complete | 完整训练HIP Graph边界 | 24进程；动态Storage安全拒绝，AdamW主机step不可重放 |
| [44](steps/44-adamw-device-step-graph.md) | complete | device-owned AdamW step | FP32 many-small 1.427×/1.436×；BF16和大Tensor拒绝 |
| [45](steps/45-adamw-stable-descriptor-multi-graph.md) | complete | immutable descriptor + 两节点multi Graph | BF16 small 10.813×/36.929×；FP32 large 0.908×反例 |
| [46](steps/46-gradient-address-stability.md) | complete | real backward gradient地址审计 | Qwen T8/T512稳定；DeepSeek T512变化198项/7.108GB |
| [47](steps/47-optimizer-graph-model-preflight.md) | complete | graph-ready Stream/allocator preflight | 四case 12进程snapshot失配，0次launch |
| [48](steps/48-quiescent-allocator-handoff.md) | complete | device-wide静止后的allocator阶段交接 | 救回Qwen T8/T512与DeepSeek T8；Deep T512仍拒绝 |
| [49](steps/49-optimizer-graph-model-gate.md) | complete | 两节点optimizer Graph模型门 | optimizer 0.656×–0.807×，模型路由拒绝并关闭track |
| [50](steps/50-rocwmma-qk-tile.md) | complete | rocWMMA QK矩阵单元能力 | 48进程全对齐；T512胜出、T2048 D128为0.688×反例；只进入原型 |
| [51](steps/51-rocwmma-online-attention.md) | complete | online rocWMMA causal GQA原型 | 42进程；14/14胜当前1.260×–4.041×；只准入fallback operator |
| [52](steps/52-rocwmma-online-operator.md) | complete | 公共online causal GQA算子 | 10 native全胜；4 fallback精确反例；准入模型A/B |
| [53](steps/53-rocwmma-online-model-discard.md) | complete | online Attention完整模型门 | 六格0.761×–0.884×且Qwen logits失败；拒绝模型路由 |
| [54](steps/54-rocwmma-direct-bf16-model-discard.md) | complete | 去三cast反驳实验 | direct BF16仍0.777×–0.906×；关闭online模型track |
| [55](steps/55-current-inference-profile.md) | complete | 当前B1T1024重新profile | GEMM 59.7%/66.8%；下一步筛exact Attention solution |
| [56](steps/56-fp32-attention-t1024-discard.md) | complete | T1024 exact QK/PV | PV descriptor失配；Qwen错/Deep慢，默认全拒绝 |
| [57](steps/57-bf16-swiglu-vector-discard.md) | complete | BF16 SwiGLU vector | operator 1.249×/1.190×；整模1.007×/1.001×，Auto拒绝 |
| [58](steps/58-bf16-grouped-swish-discard.md) | complete | grouped Swish epilogue | operator 1.097×/1.069×；整模1.000×/0.991×且logits变化 |
| [59](steps/59-bf16-rms-norm-output.md) | complete | RMSNorm直写BF16 | Event 1.866×/2.070×；位级相同，准入模型门 |
| [60](steps/60-bf16-ffn-norm-model.md) | complete | FFN Norm直入Arena | 整模1.0122×/1.0092×；位级相同，默认启用 |
| [61](steps/61-post-bf16-ffn-norm-profile.md) | complete | 融合后重新profile | cast 96→72/112→84；下一边界Attention Norm |
| [62](steps/62-bf16-attention-norm-model.md) | complete | Attention Norm直入QKV Arena | 整模1.0131×/1.0130×；位级相同且降峰值 |
| [63](steps/63-post-bf16-attention-norm-profile.md) | complete | 两Norm后重新profile | Kernel 8.069/14.489ms；每层cast剩一进一出 |

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
