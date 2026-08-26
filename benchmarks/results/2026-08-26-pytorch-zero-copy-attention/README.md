# PyTorch零复制MHA/GQA Attention

三个随机seed、5组shape，共15条完整context输出：MHA 3条，GQA 12条。每个case直接包装PyTorch
Q/K/V、output和三个workspace，共105/105指针一致、105/105 non-owning，wrapper复制0字节。

| Shape | 头配置 | 三次最大Max | 三次最大RMS |
|---|---|---:|---:|
| B1 T1 D4 | H2/KV2 | 0 | 0 |
| B1 T7 D8 | H4/KV2 | 2.38e-7 | 6.20e-8 |
| B2 T17 D16 | H4/KV1 | 0 | 0 |
| B1 T64 D32 | H4/KV2 | 8.34e-7 | 6.79e-8 |
| B1 T256 D64 | H4/KV2 | 8.94e-8 | 4.57e-9 |

T256 hipBLASLt路径还逐项检查scaled-Q、最终expanded-V和causal probabilities，workspace Max最大
`2.98e-8`。15/15 Event在记录时pending。

![Attention matrix](attention-matrix.svg)

外部workspace不是可选的“临时实现细节”：shape、dtype、device、contiguity和四个可写Tensor互不
alias都在launch前检查。短序列融合Kernel不使用workspace payload，但仍验证契约，以便同一个调用面
安全切换到T256库路径。

rocprof/PyTorch注入冲突仍存在，因此这是完整输出和零复制证据，不是Attention速度报告。
