# AdamW correctness-before-timing tuner

## 初中生版本

AdamW每一步会同时改四份“作业本”：参数、第一份历史平均、第二份历史平均，以及可选的BF16
副本。只检查参数的第一个和最后一个数字，就像只看一本作业的封面和最后一页，不能证明中间
没有写错。

现在程序先复制完整作业本，用最容易理解的Scalar版本写出标准答案。另一个候选必须把每一页
都对上，程序才允许给它计时。未对齐的内存不适合float4读取，所以Vectorized会在计时前被
拒绝，而不是先崩溃再猜原因。

## 公共接口

- `make_adamw_tuning_key()`：生成不会跨shape、对齐、GPU或运行时串用的钥匙；
- `autotune_adamw()`：只筛选和测量，不修改调用者状态，也不改变`Auto`；
- `register_adamw_autotune_winner()`：调用者完成整机回归后显式接受；
- `save/load_adamw_tuning_cache()`：事务式保存，并过滤旧环境数据；
- `microllm_tune_adamw`：输出可解析的完整状态误差和Event/墙钟P50/P95；
- `adamw_autotune_matrix.py`：fresh process真实参数量矩阵。

## MI300结论

15个fresh process全部完成。四个对齐case的Vectorized加速分别是1.000×、0.860×、0.959×、
1.010×，没有一个达到1.05保留门。因此机制合入，但`Auto`仍回退Scalar。这个结果比“代码看起来
一次处理四个数，所以一定快”更可靠。

原始记录、cache与三次端到端回归在
[`benchmarks/results/2026-08-23-adamw-correctness-before-timing/`](../../benchmarks/results/2026-08-23-adamw-correctness-before-timing/)，
完整实验解释见[Experiment 157](../optimization-log/experiments/157-adamw-correctness-before-timing.md)。

最终回归：CPU 255/255、ASan/UBSan 253/253、PyTorch-enabled CPU 229/229、完整
CPU/HIP 378/378（2个条件跳过，HIP标签119/119）。安装package的外部consumer会调用新AdamW
tuner符号。干净覆盖率为80.0% lines、89.5% functions、61.3% branches。
