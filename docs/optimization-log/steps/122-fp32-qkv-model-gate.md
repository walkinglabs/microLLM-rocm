# Step 122 — FP32 QKV solution complete-model gate

Status: complete; default rejected, research path retained

Experiment 304为DeepSeek T2048 full-prefill选出Q solution 296100与K/V solution 292135。它们只对
当前gfx942、ROCm runtime/driver和hipBLASLt版本有效。

新增显式full-prefill projection registry，固定B1/B2/B4/B8、FP32 Linear、BF16 KV、两个fresh
process，对比默认与candidate：

- Q/K/V exact descriptor registry hit/miss和dispatch；
- Block0完整BF16 K/V prefix Max/RMS/bitwise；
- cached step0完整151,936 logits、batch内部相同行和top1；
- prefill时间、decode tokens/s、peak和workspace。

candidate必须实际注册Q一项、K/V共享一项，且不能影响decode projection shape。完整精度改善必须在
B2/B4/B8一致，不能只看全局最大值。性能只在无诊断的fresh process中声明。通过后才扩展Qwen与其他
context；失败则保留研究API并拒绝默认。

结果：Block0 K/V四个batch全部exact，但完整logits Max仅改善7.41%，RMS恶化1.2677x，B4/B8 Max
反向；B1 prefill只有0.9014x。默认拒绝。Step 123在cache exact的candidate路径寻找第一处新漂移。
