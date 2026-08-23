# Experiment 142 data

这份数据在同一worker组中直接比较FP32、原生`full`和`both-roundtrip`，不再从两个独立summary
间接推断。

- `raw.jsonl`：12个worker及其量化计数、dtype和GPU门；
- `pairs.jsonl`：四组完整向量的三方直接比较；
- `per-pair.tsv`：绘图所需的比例和误差；
- `verification.json`：50步构建、合同、计数、规则和结论；
- `gates.json`：区分“向量变化很大”和“最终总误差变大”；
- 构建、命令、3次GPU预检、退出码和空stderr均原样保留。

固定`warmup=0, steps=1`，runner明确写出吞吐不是性能证据。这个实验只回答数值因果问题。
