# Experiment 219 — 单个 GEMM 更快，整步仍然没变快

Status: `discard default; keep diagnostic seam`

## 先修一条真实基础设施边界

已有 FP32 solution registry 只查询 `batches > 1`，因为它最初服务 Attention。rank-2 weight
gradient 注册后出现1 entry却0 hit。现在显式注册表支持rank-2，并有真实Qwen shape测试；没有注册
时默认路径完全不变。

## 三进程完整输出筛选

| Model | Weight-gradient shape | Common index | Median speedup | Minimum speedup |
|---|---|---:|---:|---:|
| Qwen | 896×512×4864 | 289155 | 1.077× | 1.070× |
| DeepSeek | 1536×512×8960 | 284846 | 1.133× | 1.114× |

6个进程共384次候选评估全部通过完整输出。CLI显式注册后，一次热身+两步测量命中Qwen
`24 layers × 2 × 3 = 144`次，DeepSeek命中168次。

## 模型反驳

| Model | Baseline tok/s | Candidate tok/s | Ratio | Final-loss relative diff |
|---|---:|---:|---:|---:|
| Qwen | 15,493.55 | 15,377.81 | 0.9925× | 0.0252% |
| DeepSeek | 6,488.55 | 6,461.88 | 0.9959× | 0% |

![FP32 weight-gradient solutions discarded](../assets/fp32-weight-gradient-solutions-discard.svg)

operator局部收益真实、registry命中真实、数值也正确，但端到端两模型都回退。可能原因包括其他
GEMM/调度波动，以及单个family只占完整step的一小部分。不能因为局部快7%–13%就设默认。

## 决定

- 不安装默认solution，不持久化index；
- 保留通用rank-2 explicit registry、完整输出tuner和CLI研究flag；
- 关闭当前backend版本的training GEMM solution-index搜索；
- 下一方向回到图级liveness/capture，必须先解决地址稳定和workspace生命周期。

原始证据在
[`benchmarks/results/2026-08-24-fp32-weight-gradient-solutions/`](../../../benchmarks/results/2026-08-24-fp32-weight-gradient-solutions/)。
