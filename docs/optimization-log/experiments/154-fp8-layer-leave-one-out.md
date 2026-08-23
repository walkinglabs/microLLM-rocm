# Experiment 154：Qwen layer 9改善三成，DeepSeek没有一个安全单层

Exp140只测试误差放大最明显的Qwen21/Deep27，无法说明其他层是否更适合作为FP32岛。本轮固定
retained E4/O-only/dynamic policy，穷举每个block的单层FP32反事实。每个模型先跑FP32 oracle
和FP8 baseline，再为Qwen 24层、DeepSeek 28层各启动一次fresh process。

| 模型 | 最佳层 | Max / baseline | RMS / baseline | 两项都不差的层 | 完整门 |
|---|---:|---:|---:|---:|---:|
| Qwen2.5-0.5B | 9 | 0.713× | 0.666× | 20/24 | 0 |
| DeepSeek-Distill-Qwen-1.5B | 9 | **1.022×** | 0.994× | **0/28** | 0 |

![FP8 layer leave-one-out](../assets/fp8-layer-leave-one-out.svg)

Qwen layer9让Max/RMS改善28.74%/33.42%，明显优于凭layer drift选择的layer21；但绝对RMS仍为
0.18497，离0.05门很远。DeepSeek按RMS排序的最佳也是layer9，却让Max恶化2.22%；28个候选
没有一个同时守住Max和RMS。这直接否定跨模型“只恢复一个共同block”的策略。

56/56行成功，52个layer候选各比较151,936 logits，所有top token一致。路由计数证明每个候选
恰好少7个FP8 Linear：Qwen linears/dynamic/post为161/92/23，Deep为190/109/27。搜索吞吐没有
轮换与重复，因此不参与选择。

决定：关闭DeepSeek单层方向；Qwen layer9只进入下一轮T8/T512三进程正式反驳，不在本轮keep。
如果长上下文回归或速度超过5%门，就同样删除候选；即使通过相对门，也必须单独报告完整precision。
