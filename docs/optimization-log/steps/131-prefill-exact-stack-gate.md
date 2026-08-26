# Step 131 — Batch-selective exact-linear stack gate

Status: planned

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
