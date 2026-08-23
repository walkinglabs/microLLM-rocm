# Experiment 143 data

这是`output-channel-amax`模型策略的正式Qwen/DeepSeek短长上下文证据。

- `raw.jsonl`：36个交错FP32/BF16/FP8 worker；
- `summary.json`：四个shape的三进程中位数、完整logits和显存；
- `verification.json`：scale字节、调用次数、基线差值和keep门；
- `per-case.tsv`：图表输入；
- fresh build、两个合同、命令、3次GPU预检、退出码和空stderr均保留。

每个worker含1次warm-up和3次测量，因此动态激活与post-scale raw计数对应4次forward。
T8性能跨越了共享激活优化，只保留数值，不作性能归因；T512与Exp135同口径。
