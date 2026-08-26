# 2026-08-26 — exact stack组合路线关闭

我们没有从旧表格拼结果，而是用Release二进制重新跑了16个precision和16个反向排序performance进程。
固定候选为：B1保持upstream，B2/B4启用exact core+O，B8启用exact core但关闭O。

四个prefill比值为0.997×、0.987×、1.020×、1.008×，性能门通过，峰值显存和分配不变。但是完整
logit Max从0.001253升到0.001340，恶化6.9%；RMS只改善2.5%，远低于10%门。因此候选拒绝。

预检时曾用Debug构建跑过一次。Debug绝对时延约为Release三倍，而且编译浮点路径让完整logit误差分布
也不同。Debug目录已移到`/tmp`，仓库只保留Release证据。这避免把不同构建类型的绝对时间混在一起。

Q/K/V、QK、P×V和O的solution组合线到此关闭。保留所有显式scope作为诊断工具；下一节点把FFN
拆成gate、up、SwiGLU activation和down output，继续寻找第一处差异。
