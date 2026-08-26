# Step 117 — BF16 decode algorithm consistency

Status: complete; solution 75892 rejected

Experiment 299拒绝Block0-only FP32，并发现B>1相同行在FP32控制中也有小漂移。下一节点先隔离BF16
gate/up GEMM的batch-shape algorithm选择。

第一阶段调用`microllm_bench_bf16_algorithms`，固定DeepSeek gate/up的`K=1536,N=8960`、BF16
输出、32 MiB workspace，枚举M=1/2/4/8各自候选与交集。记录solution index、workspace和waves；
index只对当前ROCm/hipBLASLt/gfx942环境有效。

若交集非空，第二阶段加入显式decode-only注册：同一个index同时注册M=1/2/4/8。重复Experiment 299
的全BF16完整logits、吞吐和peak，默认dispatch作为对照。若没有共同solution，记录硬边界并转向
down projection/Attention context，不伪造固定算法。

任何候选必须先通过每个shape的support与单算子CPU对照，再跑完整模型。版本局部index永不写成默认。

结果：M1/2/4/8的64个候选全部相交，但75892让全局Max变成1.1104x，B4/B8 RMS变成
1.6063x/1.4678x，吞吐最低0.9853x，peak增加4,587,520 bytes。共同index不保证跨M保序。
Step 118在算子层搜索64个候选的row invariance。
