# 2026-08-25 — ranked checkpoint ownership result

![Ranked checkpoint resume](../optimization-log/assets/ranked-checkpoint-resume.svg)

tiny两rank的2+3步恢复checkpoint与不中断5步checkpoint均10,796 bytes且逐字节相等；所有参数
差0，rank0写3次、其他rank写0次。写失败为rank0=1、peer=−15，无文件残留。

rank0 ownership/resume合同保留。下一节点扩展Model-S one-step checkpoint，记录约187MB完整状态的
实际大小、写/读时间和恢复后参数/optimizer step。
