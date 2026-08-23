# Experiment 141 data

这份证据把同一个官方模型的FP8 Linear拆成两种慢速诊断：`weight-only`只保留权重舍入，
`activation-only`只保留激活舍入。每套包含Qwen/DeepSeek、context 8/512，以及相同进程内的
FP32/BF16参考。

- `verification.json`：24个worker的合并合同和四组误差归因；
- 两个模式目录：命令、3次GPU空闲预检、12条原始行、汇总、退出码和空stderr；
- `fresh-configure.log`、`fresh-build.log`：从空目录开始的50步构建；
- `diagnostic-strings.log`与CLI合同：证明运行的不是旧二进制；
- `gates.json`：哪些解释被支持、哪些解释仍缺反事实。

每个诊断FP8 worker比较151,936个完整logits。两个模式都使用FP32 GEMM且native FP8 dispatch为
0，因此本数据中的TPS只表示诊断成本，不进入任何FP8性能曲线。
