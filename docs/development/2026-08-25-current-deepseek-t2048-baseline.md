# 2026-08-25 — current DeepSeek T2048/B2/N64 baseline

Step 104从干净`4ac2393`重跑了当前跨框架长上下文失败。两边都是一token一次forward、BF16
cache、精确2112-token容量、B2、输出64、warm-up 2、measured 5、fresh process三轮交替。

结果：

- 64个生成token完全一致；
- microLLM：133.50 tok/s，CV 0.064%；
- PyTorch ROCm：163.64 tok/s，CV 2.96%；
- 当前比值：0.8158x；
- microLLM/PyTorch峰值：5.23/6.38GB；
- 两边KV：121,110,528 bytes，利用率100%；
- microLLM measured区：1120次/587.2MB D2D，5次H2D，5次D2H。

旧0.868x没有被继续引用为当前结果。microLLM相对旧行快0.8%，但当前PyTorch快7.2%，因此比值
降到0.816。下一结论必须来自当前rocprof，不能把差距简单解释为microLLM回退。

第一次命令漏了显式AMDSMI fallback：三条micro通过、三条PyTorch共同拒绝。runner随后覆盖并
重跑全部六个进程。失败原因和处理保存在`attempts.json`。

证据：[`current DeepSeek T2048`](../../benchmarks/results/2026-08-25-current-deepseek-t2048/)
