# Experiment 146 data

这份证据包含候选和同revision device-Tensor control两套正式矩阵。

- `candidate/`：output-head-only的36个worker、命令、summary和verification；
- `control/`：相同binary/shape/顺序参数的device-tensor-amax 36个worker；
- `candidate/per-case.tsv`：真正决定keep的同revision对比；
- `candidate/historical-context.tsv`：最初误用的Exp129/135背景，只保留审查过程；
- fresh build、两个合同、两套独立GPU预检、退出码和空stderr全部保留。

历史host Tensor-amax不是候选未选Linear的device Tensor-amax，不能决定keep。追加control后结论
从“可能改善Deep”改为“数值完全不变”。
