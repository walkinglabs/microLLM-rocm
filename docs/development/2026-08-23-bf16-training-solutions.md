# BF16 training solution tuner

## 初中生版本

同一道矩阵乘法，库里有很多“做题路线”，每条路线有一个本机版本相关的编号。单独做一道题时，
某个编号可能快10%–19%；但模型每步还有反向、优化器、同步和很多别的矩阵乘法，所以不能直接
把单题冠军写成全局默认。

新工具先用默认路线写完整答案。每个候选编号必须把整张输出表对上，才允许计时。8个真实
shape、24个进程、1536次候选全部通过；但是多数shape在三个进程里选出的单次冠军不同。

把中位冠军显式接进真实模型后，全部shape策略是Qwen 0.995×、Deep 1.005×；删掉收益很小的
gate/up后也只有1.020×/1.007×。都没有达到1.05，所以不设默认、不写持久cache。

保留内容：完整输出先于Event/墙钟的tuner、fresh-process矩阵和显式研究CLI。原始记录见
[`benchmarks/results/2026-08-23-bf16-training-solutions/`](../../benchmarks/results/2026-08-23-bf16-training-solutions/)。

最终回归：CPU 255/255、ASan/UBSan 253/253、PyTorch-enabled CPU 229/229、完整CPU/HIP
381/381（2个条件跳过，HIP标签122/122）。默认registry仍为0，安装package consumer通过。
