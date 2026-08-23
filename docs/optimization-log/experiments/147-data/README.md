# Experiment 147 data

Attention-only candidate与同revision device-Tensor control各36个worker，共72个。

- `candidate/`、`control/`分别保留命令、独立GPU预检、raw、summary和空stderr；
- `candidate/per-case.tsv`保存四组同revision误差、速度和显存差值；
- `candidate/verification.json`检查24个FP8目标行和scope计数；
- fresh build与两个合同均保留。

本实验没有使用任何历史host baseline。keep失败只来自Qwen T512 RMS，不由top token或平均分覆盖。
