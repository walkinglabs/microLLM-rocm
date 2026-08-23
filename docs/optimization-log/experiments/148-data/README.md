# Experiment 148 data

O-projection-only candidate与同revision device-Tensor control各36个worker，共72个。

- 两个目录分别保留命令、独立GPU预检、raw、summary、退出码和空stderr；
- `candidate/per-case.tsv`保存真正决定targeted keep的四组比较；
- `candidate/verification.json`检查24个FP8目标行、scale、post和状态；
- fresh 50步构建与两个合同均保留。

Targeted keep只表示候选相对control无回归且有改善；完整FP8精度门仍单独报告为0/4。
