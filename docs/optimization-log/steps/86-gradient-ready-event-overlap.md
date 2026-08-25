# Step 86 — Gradient-ready Event and asynchronous all-reduce prototype

Status: implemented, Model-S A/B pending

Experiment 262证明25MiB自然bucket 2/1分别在backward 1/57和35/57完成。下一步仅在显式开关下：

1. compute/default Stream在bucket最终参数ready后记录Event；
2. communication Stream等待Event并pack/all-reduce/average；
3. host继续执行剩余backward；
4. optimizer前等待所有bucket work；
5. 同步control保持可选。

先做tiny故障/顺序门，再做Model-S同二进制3进程A/B。必须逐step loss一致、末步rank参数差0、
峰值不增加、total≥1.01×；只看到communication阶段变短不能保留。

实现边界：必须同时启用persistent buckets与gradient views；step 1同步建plan，后续step在两个
rank同bucket都ready后，各自default Stream记录Event，communication Stream wait后pack并enqueue
RCCL sum+in-place scale。全部backward返回后统一wait，替换parameter grad views，再进入optimizer。

单进程仍按rank0→rank1顺序backward，所以只能与rank1剩余计算重叠；不能外推到标准DDP。
Model-S pilot的steady total约14.01ms、finish wait 1.48ms，peak与sync views相同；正式三策略
轮换矩阵前保持默认关闭。
