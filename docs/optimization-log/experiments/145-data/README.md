# Experiment 145 data

这是外部PyTorch ROCm逐权重重建审计，不是microLLM模型精度实验。

- `raw.jsonl`保存365个Linear的scalar/column误差和scale spread；
- `summary.json`按模型和Attention/FFN/output head合并SSE；
- `verification.json`检查shape、元素数、有限值、分组计数和分布；
- `group-summary.tsv`用于画图；
- runner合同、GPU预检、命令、退出码与空stderr全部保留。

HF权重布局为`[output,input]`；逐row scale对应microLLM转置后`[input,output]`的逐column scale。
