# Qwen3 decode-only up-FP32双表示路径

日期：2026-08-27
状态：机制与smoke通过，完整测量待做

全局up-FP32精度8/8，但T512/B2 prefill只有0.8875x。这个节点只实现它提出的反驳实验：

```text
prefill      gate BF16 + up BF16 mirror + down BF16 -> fused FFN
cached decode gate BF16 + up FP32 parameter + down BF16 -> readable mixed path
```

阶段来自`forward_prefill_cached`和`forward_cached*`调用点，不能用`T==1`猜。mirror是派生运行状态，
不进入named parameters和checkpoint；`model.to()`会一起移动。CLI为
`--bf16-ffn-decode-up-fp32 true`，默认关闭，且与非all scope、逐层排除和token-prefill互斥。

Qwen3 smoke准确报告28份FP32 decode权重、28份BF16 prefill mirror、常驻1,855,717,376字节。
相对全BF16增加352,321,536字节，相对全局up-FP32增加176,160,768字节。CPU 433/433、
ASan/UBSan 430/430、HIP 215/215和runner 82/82通过。

这只让候选具备可测条件，不证明它更快。下一节点必须原样重跑八个oracle、64-worker shape和
五场景3进程性能门；如果prefill恢复但内存或decode门失败，仍然拒绝。
