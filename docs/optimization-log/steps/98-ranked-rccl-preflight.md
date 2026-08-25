# Step 98 — Ranked RCCL shared-memory preflight

Status: complete

Experiment 274证明world4失败有界，但`unhandled system error`对用户仍不够可操作。下一节点在不
改变RCCL初始化语义的前提下：

- 用RCCL/NCCL debug环境复现四rank，提取共享内存/IPC相关根因；
- launcher在启动前记录visible GPU数、world size、`/dev/shm` total/free和拓扑摘要；
- preflight只在证据充分时给出minimum/observed判断，不能猜一个阈值；
- world1/2不得被共享内存提示误拒绝；
- world4资源不足时仍启动可选probe以验证诊断，不把preflight当作成功实测。

产物是诊断与修复说明，不是绕过RCCL或降低world size。

实现已记录visible GPU和shm total/free，并从每进程官方debug日志提取根因。pilot四rank均出现
21,823,872-byte segment `No space left on device`，诊断`shared-memory-capacity-exhausted`；
required total保持unknown，原始507KB日志删除。world2非回归通过。

正式结果复现：4/4日志No-space、segment21,823,872、shm free43,724,800、RCCL2.28.3；world2
完整门通过。诊断保留，world4能力状态仍未通过。
