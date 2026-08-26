# Experiment 323 — 旧128线程其实仍在模拟256条lane

## 审计结论

当前finalize用256-lane max/sum tree。到了P×V，DeepSeek width只有128，所以后128线程不工作。旧
128-thread mapping为了位级保持，仍循环模拟256条logical lane和旧tree，因此没有测试真正的128-lane
归约。

选择的新假设是native128：score stride、reduction tree、shared reduction footprint和P×V物理lane都为
128。它与旧mapping有结构差异，也不同于split-PV、GQA reuse、online和materialized-score选择本身。

## 风险和反驳门

归约顺序变化，不能假设位级相同。先比较完整输出Max/RMS和finite，再计时。DeepSeek T2048 B1/B2、
FP32/BF16 cache的每个case都必须Event≥1.05×、wall≥1.02×。失败就删除candidate并关闭finalize局部线。

![Finalize architecture gap](../../../benchmarks/results/2026-08-26-finalize-architecture-gap-audit/finalize-gap.svg)
