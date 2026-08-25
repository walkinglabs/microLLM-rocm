# Step 86 — Gradient-ready Event and asynchronous all-reduce prototype

Status: planned

Experiment 262证明25MiB自然bucket 2/1分别在backward 1/57和35/57完成。下一步仅在显式开关下：

1. compute/default Stream在bucket最终参数ready后记录Event；
2. communication Stream等待Event并pack/all-reduce/average；
3. host继续执行剩余backward；
4. optimizer前等待所有bucket work；
5. 同步control保持可选。

先做tiny故障/顺序门，再做Model-S同二进制3进程A/B。必须逐step loss一致、末步rank参数差0、
峰值不增加、total≥1.01×；只看到communication阶段变短不能保留。
