# Tied embedding sparse accumulation

## 初中生版本

Qwen把“读token的词表”和“输出下一个token的词表”绑成同一份参数。反向时，输出头先写好一张
1.36亿格的完整梯度表；Embedding随后只有512行需要补充。旧程序却先再造一张544MB、几乎
全是0的大表，然后把两张大表逐格相加。

新程序确认原大表只有一个主人后，直接把512行加进去。若还没有原表、内存有别名、shape不对
或不连续，就回到旧办法，不能冒险原地修改。

## 证据

- 诊断确认先到的是`matmul_right`，后到的是`embedding_backward`；
- CPU/HIP重复token和完整图梯度对齐；
- Qwen峰值13.025→11.969GB，减少1.056GB；
- Qwen吞吐1.018×，DeepSeek untied为1.006×且零命中；
- loss差0.0207%，固定参数guard相等；
- profile少3次dense add和3次dense fill。

完整实验见[Experiment 161](../optimization-log/experiments/161-tied-embedding-sparse-add.md)，原始记录见
[`benchmarks/results/2026-08-23-tied-embedding-sparse-add/`](../../benchmarks/results/2026-08-23-tied-embedding-sparse-add/)。

最终回归：CPU 259/259、ASan/UBSan 257/257、PyTorch-enabled CPU 233/233、完整CPU/HIP
387/387（2个条件跳过，HIP标签124/124）。干净覆盖率80.3% lines、89.7% functions、
61.4% branches；安装package consumer实际链接新诊断和稀疏累加符号。
