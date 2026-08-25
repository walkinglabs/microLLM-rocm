# Step 87 — One process per GPU bootstrap

Status: complete, bootstrap kept

单进程Event overlap只比同步views快1.0159×，主要限制是rank0/rank1顺序backward。下一节点先不
做性能优化，建立one-process-per-GPU最小合同：

- 从`RANK`、`LOCAL_RANK`、`WORLD_SIZE`或显式CLI读取身份；
- rank0生成并通过文件/标准输入交换RCCL unique ID；
- 每进程只创建一份模型、optimizer和一张GPU上下文；
- 一个固定Tensor all-reduce和一个tiny global-batch step与单进程参考对齐；
- timeout、坏rank、缺ID和一个进程失败必须让其他rank退出而不是永久等待；
- checkpoint只由rank0写，所有rank先完成参数一致性门。

完成bootstrap后再移植ready bucket overlap。四卡仍受当前容器共享内存限制，不提前宣称。

实现新增rank-local RCCL communicator和opaque ID byte API。worker支持CPU global-batch reference或
单rank GPU训练；rank0原子发布ID文件，其他rank有限时等待。launcher先启动rank1，组级deadline
监控所有进程，任一非零peer会终止仍在NCCL等待的进程，并逐项比较完整参数。

pilot：两个独立rank各3step后728个参数值完全相同，和CPU reference最大差1.19e-7；坏rank
注入返回1，等待peer被SIGTERM，组不挂死。正式三次fresh launch前不迁移overlap。

正式结果：3次fresh launch、6个rank进程、728值跨rankexact、CPU最大差1.19e-7；故障返回
[1,-15]且peer被终止。准入rank-local同步bucket reducer，尚无跨进程overlap或性能声明。
