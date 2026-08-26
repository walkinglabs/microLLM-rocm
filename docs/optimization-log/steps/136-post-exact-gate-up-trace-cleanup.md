# Step 136 — Post-exact gate/up trace and rejected-route cleanup

Status: planned

Experiment 319已经关闭模型优化线，但296100仍可作最后一次诊断控制。固定exact Attention stack和exact
gate/up，运行B1/2/4/8、两个fresh Release进程的七阶段FFN完整值trace：

```text
FFN norm → gate → up → SwiGLU activation → down → outputs
```

若gate/up/activation全部cross/within exact且down首差，就把问题交给down descriptor；否则修正解释。
无论结果如何，随后删除`PrefillFfnGateUpProjection`、CLI参数和两个模型候选runner，保留通用operator
矩阵、raw model evidence与可读实验文档。
