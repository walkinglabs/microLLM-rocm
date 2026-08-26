# Experiment 318 — Max过门，RMS只改善3.3%

## 结论

B1/B2/B8 gate+up使用296100、B4保持default后，四个prefill为
`0.993/0.981/1.005/1.002×`。全局Max从0.001354降到0.001191，改善12.0%；全局RMS从
0.0002294降到0.0002218，只改善3.3%。候选拒绝。

## 为什么还要测一次B4 exact

candidate的B2/B8都明显改善，B4却成为新的Max/RMS上限。B4没使用exact方案，不是因为数值失败，
而是operator M8192只有0.941×。算子回退不一定等于整模回退超过5%，所以最后一次让B4也exact，并仍用
完整模型0.95门决定。这个问题由本次失败直接提出，不是事后随意换策略。

## 证据

- 16个precision、16个反向performance进程；
- baseline真实upstream；
- 每个candidate进程严格检查0或1个scope entry和56次dispatch；
- peak、allocation和token不变；
- performance/Max通过，RMS失败。

原始结果见
[`benchmarks/results/2026-08-26-fp32-prefill-ffn-model-gate`](../../../benchmarks/results/2026-08-26-fp32-prefill-ffn-model-gate/README.md)。
