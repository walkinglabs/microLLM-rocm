# 2026-08-25 — Model-S gradient-ready顺序结果

![Gradient-ready bucket order](../optimization-log/assets/data-parallel-gradient-ready-order.svg)

三个进程、9个step、两rank的57参数order完全相同，并严格反转parameter order。25MiB bucket 2
在1/57完成，bucket 1在35/57，bucket 0在57/57；两bucket具备backward内理论通信窗口。

当前实现仍同步。这一结果只准入Event+async collective原型，下一节点继续守loss、最终参数、
peak与端到端total门。

发布门：CPU `364/364`、ASan/UBSan `362/362`、RCCL `32/32`、42个graph API与120个测试源。
