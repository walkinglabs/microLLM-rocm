# BTHD Attention 的 BF16 Q/K 边界

## 问题

BTHD Attention已经删除布局复制，但精确grouped QKV仍把Q/K从BF16转成FP32，融合
bias+RoPE紧接着读取这些值。Experiment 205只跨过这一条边界：融合Kernel新增BF16输入，
V和Attention输出不变。

## 实现合同

- `bf16_qkv_projection_out_`只有在精确grouped计划命中时才能返回“Q/K保留在BF16”；
- miss、CPU、非精确shape和普通调用继续写完整FP32 Q/K/V；
- plan key只绑定GroupedGemm真正读写的地址，不再错误绑定未由它写入的FP32输出；
- `--inference-bthd-bf16-qk`要求BTHD、QKV Arena与精确grouped算法同时存在；
- 新计数器区分普通grouped dispatch与真正保留Q/K的dispatch。

## 证据

| Gate | Result |
|---|---:|
| CPU | 311/311 |
| ASan/UBSan | 309/309 |
| PyTorch-enabled build | 285/285 |
| CPU+HIP完整配置 | 484/484，3项按环境跳过 |
| gfx942 HIP label | 163/163 |
| RCCL multi-GPU | 12/12 |
| RCCL完整label | 14/14 |
| 注册测试文件 | 83 |

正式T512五进程门：Qwen/DeepSeek为1.0224x/1.0238x，完整logits位级一致，peak不变。
Phase profile删除48/56次cast，总Kernel改善1.0787x/1.0600x。

第一次三进程DeepSeek只有1.0068x，因此保留为反例。一次错误的`ctest -j4`让多个RCCL测试
同时申请全部GPU，导致6/14失败；不改代码、按合同串行重跑后14/14与multi-GPU 12/12通过。

完整实验、原始数据与图见
[Experiment 205](../optimization-log/experiments/205-inference-bthd-bf16-qk.md)。
