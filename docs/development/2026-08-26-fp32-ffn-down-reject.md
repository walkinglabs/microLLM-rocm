# 2026-08-26 — down唯一exact候选最差只有0.506×

K8960/N1536的四个M有15个共同候选，所有候选通过CPU sentinel，只有296100完整重复block exact。
它的逐M speedup为0.506×、0.758×、0.686×、0.863×，因此不进入模型门，也不新增down scope。

gate/up失败route仍保持删除状态。至此，Q/K/V、QK/P×V、O、gate/up、down的vendor row-order搜索都已
用真实descriptor和完整模型门验收或拒绝。下一步回到clean upstream重跑当前端到端与profile，选择真正
占时间的新目标。
