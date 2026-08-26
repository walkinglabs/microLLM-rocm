# Step 131 — Batch-selective exact-linear stack gate

Status: completed by Experiment 315; rejected, track closed

Experiment 314比较的是两个都不适合作为默认的诊断策略。现在回到真实上游策略，只组合已经测过的
局部因果，并通过新进程重新测量，不能只拼接旧表格：

- B1：只用Q=296100、K/V=292135，保持upstream；
- B2/B4：再用QK=304681、P×V=295716、O=296100；
- B8：使用QK=304681、P×V=295716，但不使用会让B8 Max恶化的O；
- 所有batch：完整151,936 logits、BF16 cache、两个fresh precision进程；
- 性能使用反向排序的两个fresh process，检查prefill、peak和allocation。

准入仍要求全局Max和RMS相对真实upstream都至少改善10%，且每个batch prefill不低于0.95×。
失败就关闭这组exact Linear solution组合，不再继续按旧结果挑batch；下一步细分FFN gate、up、SwiGLU
和down的完整值trace。

结果：Release的四个prefill比值为0.997/0.987/1.020/1.008×，性能门通过且资源不变；但是全局Max
从0.001253恶化到0.001340，RMS只改善2.5%。双数值门失败，candidate拒绝，exact Linear solution
组合线关闭。详见[`Experiment 315`](../experiments/315-prefill-exact-stack-reject.md)。
