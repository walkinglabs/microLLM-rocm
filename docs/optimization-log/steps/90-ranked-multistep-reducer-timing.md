# Step 90 — Ranked multi-step reducer timing

Status: complete, transient bucket steady performance rejected

Experiment 266把collective从57降到3，但one-step bucket Reducer的CV达到89.3%。下一节点不先写
persistent实现，而是在每个fresh rank进程内运行多步并逐step记录forward/backward、Reducer、
optimizer和total。

第一步单独标记为cold；后续步骤才形成steady分布。per-parameter与bucket继续交错进程顺序，
完整loss/参数/故障门不变。只有steady bucket Reducer稳定且确实占用可优化，才准入persistent
rank bucket；否则先调查RCCL初始化、Stream等待或pack/unpack。

实现已逐step输出时间与Reducer分配/copy计数。单次pilot显示bucket steady约4.56ms，而
per-parameter约2.66–2.79ms；bucket每步仍有60次backend allocation、124,689,408 bytes和
57+57次pack/unpack。三步完整数值门通过。该反例必须由三次交错正式矩阵确认后才形成结论。

正式6个steady样本确认：bucket Reducer `4.205ms`对逐参数`2.837ms`，speedup `0.6747×`；
完整step `10.396ms`对`8.864ms`，speedup `0.8527×`。bucket CV仅2.72%，并且每个steady
step仍有60次backend allocation、124,689,408 bytes与57+57次copy。拒绝transient性能路线，
准入只改变Storage生命周期的persistent rank plan。
