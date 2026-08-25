# Step 97 — Ranked world-size boundary

Status: implemented, formal clean-revision boundary evidence pending

当前`RankCommunicator`本身接受一般world size，但worker/launcher与global-batch reference仍写死2。
下一节点把身份、local batch、CPU global batch、进程组监控与rank参数比较泛化到`--world-size N`。

必须保持world-size 2全部门；world-size 4先跑tiny一step。当前容器历史上因`/dev/shm`只有64MB而
稳定失败，不能把接口推演写成四卡实测成功。若仍失败，保存四rank return code、stderr、有限时
peer终止与共享内存证据；在资源满足前不声称4卡可运行。

worker/launcher/matrix已泛化world size。world1/2通过；当前world4四rank均在约2.7秒返回
`ncclCommInitRank system error`，`/dev/shm=67,108,864` bytes。结构化group-init模式已实现，正式
干净revision结果前不更新四卡能力状态。
