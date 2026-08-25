# 2026-08-25 — 把怀疑真的删掉，再看解释还站不站得住

上一轮我们怀疑：模型慢，是因为每层都把Q、K、V从FP32转成BF16。只看代码无法证明，所以这次
让投影、V bias和RoPE都直接产出BF16，online Attention前不再有三次转换。

结果六组都变快了一点，说明转换确实花时间；但最好也只有旧路径的91%，最差约78%。Qwen长
上下文的完整logits最大差仍约0.485。也就是说，删除转换没有救回模型，原解释不成立。

![Direct BF16 model discard](../optimization-log/assets/rocwmma-direct-bf16-model-discard.svg)

这正是反驳实验的价值：不是为了证明自己一开始猜对，而是把一种解释真正拿掉。现在我们知道，
继续优化小cast不会解决问题。公共BF16 bias/RoPE算子仍有用，但online模型路线停止。

提交前PyTorch拒绝门还发现一个测试夹具问题：原本想构造错误bias，长度却碰巧等于输入最后一维，
所以函数正确地没有拒绝。把bias改成真正不匹配后，完整315项PyTorch-enabled测试通过。这里修的
是反例输入，不是放宽算子合同。

## 发布回归

CPU 341/341、ASan/UBSan 339/339、PyTorch-enabled CPU 315/315、完整CPU/HIP 537/537
（3个条件跳过）、HIP标签184/184、RCCL标签14/14、multi-GPU 12/12；覆盖清单仍注册103个
测试文件。直接BF16新算子已进入PyTorch oracle，模型开关默认仍为false。
