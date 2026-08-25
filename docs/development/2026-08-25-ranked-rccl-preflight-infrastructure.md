# 2026-08-25 — ranked RCCL debug and resource preflight

world4的`unhandled system error`不够可操作。本节点按照AMD官方RCCL环境变量与troubleshooting
说明收集每进程日志：

- [RCCL 2.28.3 environment variables](https://rocm.docs.amd.com/projects/rccl/en/7.13.0-preview/api-reference/env-variables.html)
- [AMD RCCL troubleshooting](https://rocm.docs.amd.com/projects/rccl/en/develop/how-to/troubleshooting-rccl.html)

第一次只设`NCCL_DEBUG=INFO`没有stderr日志。按官方建议加入每进程`NCCL_DEBUG_FILE`，并兼容
当前包的`RCCL_LOG_LEVEL=5`后得到完整日志。

## Launcher diagnostics

新增`--rccl-debug`与非阻塞preflight：

- 记录KFD可见GPU数、world size、`/dev/shm` total/free；
- `visible_gpu_count_sufficient`只报告，不替代RCCL初始化；
- `required_shared_memory_bytes=null`和`required_shared_memory_unknown=true`，不猜阈值；
- rank进程使用INFO与INIT/SHM/NET/ALLOC日志并写独立文件；
- 提取RCCL版本、日志数/bytes、No-space日志数和失败segment bytes；
- 原始verbose日志提取后删除，只保存小型diagnostic JSON；
- world1/2不会因preflight提示被拒绝。

## Pilot evidence

world4：visible GPU 4，prelaunch `/dev/shm total=67,108,864`、`free=43,724,800` bytes。四份日志
都明确：

```text
shared memory segment ... size 21823872 ... No space left on device (28)
```

RCCL版本为`2.28.3-HEAD:3309c61`；4份原始日志合计507,069 bytes并被删除。诊断为
`shared-memory-capacity-exhausted`。21,823,872只是某个失败segment，不能推导总minimum。

相同preflight下world2继续完整训练/CPU门通过。正式干净revision结果前不改变world4能力状态。
