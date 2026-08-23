# Cooperative bias gradient

## 初中生版本

我们要把很多行数字按列相加。旧办法给每一列一个人，这个人从第一行一直加到第512行。虽然
列很多，但每个人自己的长队伍仍然是串行的。

新办法让一个小组同时负责32列。每列有8个人，分别加第0、8、16…行，第1、9、17…行，最后
再把8个小答案相加。横着的32个人仍读取连续内存，所以既增加了并行，也没有把读取变成乱跳。

## 边界

- `rows < 32`：保留Scalar，16-row实测只有1.005×；
- `rows >= 32`：使用`CooperativeRows`；
- 每个候选先比较完整输出，不只抽样；
- 求和顺序变化允许小浮点误差，正式上限Max 3e-5、RMS 1e-5；
- CPU仍是顺序参考，显式Cooperative只接受HIP Tensor。

## 实测

78条算子记录全部通过。T512真实宽度加速3.21×–3.27×。Qwen训练提升1.222×，DeepSeek提升
1.111×，峰值不变；rocprofv3中216次bias-gradient从26.00 ms降到4.01 ms。

原始结果见
[`benchmarks/results/2026-08-23-cooperative-bias-gradient/`](../../benchmarks/results/2026-08-23-cooperative-bias-gradient/)，
完整解释见[Experiment 158](../optimization-log/experiments/158-cooperative-bias-gradient.md)。

最终回归：CPU 255/255、ASan/UBSan 253/253、PyTorch-enabled CPU 229/229、完整CPU/HIP
380/380（2个条件跳过，HIP标签121/121）。干净覆盖率为80.0% lines、89.5% functions、
61.3% branches；安装package外部consumer通过。
