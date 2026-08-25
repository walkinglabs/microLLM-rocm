# 2026-08-25 — persistent gradient bucket结果

![Persistent data-parallel buckets](../optimization-log/assets/data-parallel-persistent-buckets.svg)

Model-S同二进制A/B中，step 2–5的communication从7.070降到4.205ms，total从21.025降到
16.360ms；后续通信backend allocation从120降到0。30个loss和6次末步参数门全部通过。

这版不能默认启用：live增加124,689,408B，peak增加157,958,408B。机制保留为显式实验路径，
下一节点用bucket view删除unpacked Storage和copy，再重新检查速度与显存。
