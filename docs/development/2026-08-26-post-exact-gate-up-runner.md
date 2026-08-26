# 2026-08-26 — exact gate/up后的FFN因果runner

wrapper复用7阶段filtered-binary runner，并在exact Q/K/V/QK/P×V/O诊断stack上再给gate/up注册296100。
每个进程应有6个entry、224次dispatch、6次cache miss和218次cache hit。完整中间值仍只保留在临时目录，
比较后删除。

这个节点只为回答gate/up/activation能否全部exact以及down是否首差。它不重新打开已被Experiment 319
关闭的模型优化线；结果记录后立即执行candidate CLI/model scope清理。
