# 2026-08-25 — ranked Model-S checkpoint result

![Ranked Model-S checkpoint](../optimization-log/assets/ranked-model-s-checkpoint.svg)

Model-S T32两rank的1+1恢复与不中断2步checkpoint均187,042,096 bytes且逐字节相等；三组
57 Tensor/15,586,176值rank exact。write约1.02–1.07s、wait 1.07s、verify 532ms、restore
740ms，所有大文件清理。

Model-S checkpoint smoke完成，无I/O性能推广。下一节点泛化rank worker/launcher到world-size参数，
在当前四卡共享内存边界下保存成功或稳定失败证据。
