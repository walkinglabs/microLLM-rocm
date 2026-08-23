# Experiment 155 data

Qwen layer9 candidate与同revision retained FP8 control各18个worker，共36个worker、12个FP8行。

- T8/T512各3个独立进程，每个进程包含FP32/BF16/FP8；
- 每个FP8行比较151,936个完整logits；
- candidate/control分别有独立3次GPU2预检、命令、raw、summary、退出码和空stderr；
- candidate精确记录layer identity `9`、161 linears、368 dynamic与92 post；
- control为168/384/96；两边native=4、fallback=0；
- T512速度退化0.877%过门，但Max/RMS恶化5.26%/36.40%；
- resident与peak都增加44,724,712B；完整precision 0/2。

Fresh build与合同复用Exp154相同revision，日志复制到本目录，保证归档可独立审计。
