# 2026-08-26 — all-batch exact FFN最终反驳runner

这个小wrapper复用Experiment 318已经验证的32进程流程，只把预先固定的策略改成B1/B2/B4/B8都对
gate+up使用296100。baseline仍是完全真实upstream，route计数仍是每candidate进程1个entry和56次
dispatch；precision、反向performance、cache/logits、peak/allocation门不变。

单独的runner避免修改Experiment 318的历史合同。输出使用新的record type和SVG文件名，便于状态测试
确认“选择性拒绝”和“all-exact最终反驳”没有混在一起。
