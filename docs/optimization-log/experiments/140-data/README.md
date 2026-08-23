# Experiment 140 data

这是选择性FP32 block反事实的完整证据包。Qwen只把block 21的7个Linear保留为FP32；
DeepSeek只保留block 27。其余Linear继续使用相同的dynamic FP8路径。

- `verification.json`：两个模型的合并合同、基线差值和结论所需字段；
- `qwen/`、`deepseek/`：每套18个worker的原始行、汇总、命令和GPU空闲检查；
- `fresh-configure.log`、`fresh-build.log`：从空目录开始的50步构建；
- `hf-cli-binary-contract.log`：实际运行文件包含新CLI，不接受旧二进制；
- `gates.json`：哪些事实通过，以及为什么拒绝默认策略。

完整logits每个worker检查151,936项，36个worker全部正常结束，top token全部一致；但四个
精度门仍失败。T512只与Exp135同一共享量化实现比较，避免把后续性能优化算到本实验头上。
