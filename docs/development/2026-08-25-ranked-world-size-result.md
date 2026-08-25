# 2026-08-25 — ranked world-size result

![Ranked world-size boundary](../optimization-log/assets/ranked-world-size-boundary.svg)

world1/2 tiny一步完整rank/CPU门通过。world4四个rank在2.756秒内全部返回1，均为
`ncclCommInitRank system error`；`/dev/shm=67,108,864` bytes，无挂死。

保留一般world-size接口，但当前环境不声明4卡执行。下一节点增加RCCL debug与共享内存preflight，
把不透明错误变成可操作资源报告。
