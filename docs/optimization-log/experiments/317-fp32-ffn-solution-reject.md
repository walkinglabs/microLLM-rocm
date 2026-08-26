# Experiment 317 — 唯一exact方案在M8192回退了5.9%

## 结论

M2048/4096/8192/16384、K1536、N8960有33个共同hipBLASLt候选。只有296100让完整重复block
位级一致，但它的四个speedup是`1.040/0.951/0.941/0.995×`。M8192低于0.95门，因此没有共同
策略被准入。

## 为什么不能看平均数

296100的几何平均是0.981×。如果只报告平均数，会看见“只慢约2%”；但B4请求对应的M8192明确慢
5.9%。课程和框架合同要求每个batch守门，所以推荐index仍是-1。

## 证据顺序

```text
四shape inventory交集
→ CPU sentinel
→ 完整2048-row重复block bitwise
→ 同进程default Event
→ 每个M性能门
```

所有33个候选通过sentinel；只有1个block exact；0个通过exact+每M性能双门。原始结果见
[`benchmarks/results/2026-08-26-fp32-ffn-row-invariance`](../../../benchmarks/results/2026-08-26-fp32-ffn-row-invariance/README.md)。

下一步不是降低门槛，而是保持B4 default，只在另外三个batch给gate+up使用296100，做一次完整模型反驳。
