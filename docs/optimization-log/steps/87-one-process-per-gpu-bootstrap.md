# Step 87 — One process per GPU bootstrap

Status: planned

单进程Event overlap只比同步views快1.0159×，主要限制是rank0/rank1顺序backward。下一节点先不
做性能优化，建立one-process-per-GPU最小合同：

- 从`RANK`、`LOCAL_RANK`、`WORLD_SIZE`或显式CLI读取身份；
- rank0生成并通过文件/标准输入交换RCCL unique ID；
- 每进程只创建一份模型、optimizer和一张GPU上下文；
- 一个固定Tensor all-reduce和一个tiny global-batch step与单进程参考对齐；
- timeout、坏rank、缺ID和一个进程失败必须让其他rank退出而不是永久等待；
- checkpoint只由rank0写，所有rank先完成参数一致性门。

完成bootstrap后再移植ready bucket overlap。四卡仍受当前容器共享内存限制，不提前宣称。
