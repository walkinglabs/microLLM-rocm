# FP8 shared activation quantization

动态FP8旧路径让每个Linear独立执行amax、finalize和quantize。Q/K/V实际共享同一个attention norm
输入，gate/up共享同一个FFN norm输入，却分别重复3次和2次。

`Linear`现在可以接收caller-owned `ScaledTensor`。Attention先量化一次再依次执行Q/K/V GEMM；
FFN同样为gate/up共享一次。O、down和非tied output仍各自量化。GEMM求值顺序与bias处理不变。

新增thread-local机器计数：Tensor/row调用数和处理元素数。tiny一层untied模型的Tensor动态调用从
8降为5；FFN-only row调用从3降为2。lazy/prepared输出逐值相同，0 payload transfer与fallback
门不变。

完整回归首次为347/348：唯一失败是coverage audit尚未识别新统计返回类型；扩展审计器后该门和
全部受影响测试通过。不能把首次结果写成全绿。正式Qwen/DeepSeek T512需验证96/113调用、完整
logits逐值相同和Release吞吐。
