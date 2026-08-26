# 2026-08-26 — gate/up只有一个exact候选，但M8192太慢

真实FFN gate/up descriptor在四个M上有33个共同候选。所有候选先通过CPU sentinel，再比较完整重复
2048-row block，最后才计时。只有296100让跨M与同M重复block位级一致。

296100相对同进程default的四个speedup是1.040×、0.951×、0.941×、0.995×。M8192回退5.9%，超过
允许的5%，所以通用策略拒绝，推荐index保持-1。不能用0.981×几何平均掩盖这个失败。

下一步只做一个预先固定的batch-selective模型反驳：B1/B2/B8 gate+up用296100，B4保持default。
如果完整logits没有稳健改善，或任一batch prefill低于0.95×，就删除模型路由并关闭vendor solution线。
