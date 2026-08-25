# Experiment 224 — Graph还没launch，为什么Qwen的稳定地址已经失效

Status: `reject optimizer-only model Graph; keep safety gate`

## 被推翻的解释

Experiment 223在纯default-Stream训练中发现：Qwen T8/T512与DeepSeek T8的gradient地址稳定。
我们原本准备对这些case运行两节点AdamW Graph。

但HIP Graph runtime要求一条显式非默认Stream。当前exact-size allocator的安全合同是：只要出现
非默认Stream，就永久关闭default-Stream地址复用，因为它无法证明退役地址是否仍被另一条Stream
使用。

完整因果链是：

```text
创建optimizer Graph Stream
→ notify_non_default_stream
→ exact-size pool永久关闭
→ 下一次backward改用backend allocation/free行为
→ gradient地址不再匹配准备descriptor时的snapshot
→ safety gate拒绝launch
```

## 12进程正式preflight

| Model | Context | Pool enabled | Snapshot matches | Graph launches |
|---|---:|---:|---:|---:|
| Qwen 0.5B config | 8 | false | false | 0/3 |
| Qwen 0.5B config | 512 | false | false | 0/3 |
| DeepSeek-Distill 1.5B config | 8 | false | false | 0/3 |
| DeepSeek-Distill 1.5B config | 512 | false | false | 0/3 |

![Optimizer Graph model preflight](../assets/optimizer-graph-model-preflight.svg)

每个进程都先创建Graph Stream，再完成warmup、snapshot backward和下一次backward。四个case三次
全部拒绝，Graph launch总数为0。preparation中位数为2.46–5.20ms，但没有性能数字，因为在地址
合同失败后计时launch属于未定义实验。

## 为什么不能“重新打开pool”

简单把一个布尔值改回true不够。非默认Stream可能仍在使用某块地址；default Stream若立即把它
分给新Tensor，会产生跨Stream use-after-free。恢复复用必须有明确的quiescent handoff：

1. 等待相关非默认Stream完成；
2. 证明所有退役块不再被它引用；
3. 才允许default Stream重新使用；
4. 下一次非默认工作前再次关闭或用Event保护。

现有permanent disable很保守，但正确。

## 决定

- 保留`graph_workspace_matches_current_gradients`并绑定workspace owner；
- 保留模型preflight runner，任何地址不匹配必须在launch前停止；
- 不发布Qwen/DeepSeek optimizer Graph速度，不添加CLI默认或实验开关；
- 关闭“只改optimizer Graph就能接模型”的方向；
- 下一最小系统实验是显式quiescent Stream handoff或Event-aware retirement，必须先在小Tensor链证明
  生命周期，再回到模型；
- stable gradient buffer仍是备选，但DeepSeek T512需要覆盖7.108GB，不能当作小补丁。

原始证据位于
[`benchmarks/results/2026-08-24-optimizer-graph-model-preflight/`](../../../benchmarks/results/2026-08-24-optimizer-graph-model-preflight/)。
