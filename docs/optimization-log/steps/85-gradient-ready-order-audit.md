# Step 85 — Gradient-ready order audit

Status: complete, overlap prototype admitted

Persistent bucket views仍能让Model-S total相对transient达到1.367×，但peak高33.3MB；leaf target和
producer微优化已分别被完整反例关闭。下一步不再减少单个copy，而是在Autograd记录57个参数
gradient第一次ready的序号/时间，比较当前parameter order、反向ready order与3个25MiB bucket。

先只做诊断，不改变同步：要求CPU拓扑稳定、HIP每个参数恰好ready一次、两rank order一致、
zero/unused/shared参数显式。只有至少两个bucket能在完整backward前ready，才进入Event+async
all-reduce overlap；否则调整bucket order或关闭当前重叠假设。

实现新增leaf final-contribution hook，默认无hook时不建计数表。DataParallel显式audit将57个参数
映射为index，要求每rank完整permutation且两rank一致；CLI输出name/elements/order。runner用与
reducer相同算法重建25MiB bucket并计算完成位置。

Model-S smoke：ready order恰为parameter order逆序。bucket 2在1/57完成，bucket 1在35/57，
bucket 0在57/57；两个bucket理论上能在backward结束前通信。正式3进程×3step审计仍待运行。

正式结果：3进程×3step×2rank全部同一57参数逆序；bucket完成位置稳定为57/57、35/57、
1/57，最终参数差0。两个自然bucket在backward结束前ready，准入Event-based overlap原型；
当前同步路径不变。
