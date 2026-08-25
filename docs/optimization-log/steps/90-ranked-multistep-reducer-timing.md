# Step 90 — Ranked multi-step reducer timing

Status: planned

Experiment 266把collective从57降到3，但one-step bucket Reducer的CV达到89.3%。下一节点不先写
persistent实现，而是在每个fresh rank进程内运行多步并逐step记录forward/backward、Reducer、
optimizer和total。

第一步单独标记为cold；后续步骤才形成steady分布。per-parameter与bucket继续交错进程顺序，
完整loss/参数/故障门不变。只有steady bucket Reducer稳定且确实占用可优化，才准入persistent
rank bucket；否则先调查RCCL初始化、Stream等待或pack/unpack。
