# FP8 activation row-range evidence

新增filtered full-value T8 trace runner，按最后一维把每个Linear输入拆成8个token rows。正式得到
Qwen96、Deep112个Tensor。FFN的median row spread约3.8–4.8倍，极端activated达到
1106/2076倍；DeepSeek Attention约1.1–1.6倍且没有row落在tensor范围四分之一以下。

因此下一API只设计FFN row scale及其输出恢复规则，不把per-row强加给所有Attention。约95MB完整
值trace不进入Git历史；逐row amax raw、命令、worker、summary和trace字节/记录manifest均保留。

详见[Experiment 130](../optimization-log/experiments/130-fp8-activation-row-range.md)。
