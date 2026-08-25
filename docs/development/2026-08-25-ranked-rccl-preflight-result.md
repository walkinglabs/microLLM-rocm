# 2026-08-25 — ranked RCCL preflight result

![Ranked RCCL preflight](../optimization-log/assets/ranked-rccl-preflight.svg)

官方per-rank debug日志把world4失败定位为共享内存容量耗尽：4/4日志均无法创建21,823,872-byte
segment；prelaunch shm total/free为67,108,864/43,724,800 bytes。required total保持unknown。

world2同一preflight非阻塞通过；507,069-byte原始日志已删除。当前world4仍不可用。下一节点转向
独立rank的uneven local-batch weighting合同，无需等待四卡资源。
