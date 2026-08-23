# Experiment 155：Qwen layer9短文本改善，长文本RMS反而恶化36%

Exp154在T8单进程搜索中选出Qwen layer9。本轮用同revision candidate/control，各自独立GPU
预检，T8/T512分别三个进程、每进程1次warm-up与3次测量。唯一变量是candidate把layer9的7个
Linear保留FP32。

| Context | Max变化 | RMS变化 | TPS变化 | Resident/Peak |
|---:|---:|---:|---:|---:|
| T8 | -28.74% | -33.42% | +18.93% | +44,724,712B |
| T512 | **+5.26%** | **+36.40%** | -0.88% | +44,724,712B |

![Qwen layer9 formal discard](../assets/fp8-qwen-layer9-formal-discard.svg)

T512速度门通过，但Max和RMS同时回归；完整precision仍0/2。candidate每worker四次forward共
161个FP8 Linear、368次dynamic与92次post，control为168/384/96；差值正好对应一个block的
7个Linear、4个共享dynamic输入与一个O-projection post，native shapes均为4且fallback为0。

因此`keep=false`。这不是测量噪声解释：完整logits在三进程内确定一致，且T512两种误差方向
相同。Exp154的T8搜索只发现局部候选，不能预测长上下文传播。

DeepSeek已在Exp154关闭全部单层，Qwen又在正式长上下文门失败，所以“恢复一个完整block”的
方向到此关闭。保留`fp8_fp32_layers`诊断API，不设置模型策略；下一步改变量化误差进入方式，
而不是继续组合更多未经证明的FP32岛。
