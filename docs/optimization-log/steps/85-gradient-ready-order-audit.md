# Step 85 — Gradient-ready order audit

Status: planned

Persistent bucket views仍能让Model-S total相对transient达到1.367×，但peak高33.3MB；leaf target和
producer微优化已分别被完整反例关闭。下一步不再减少单个copy，而是在Autograd记录57个参数
gradient第一次ready的序号/时间，比较当前parameter order、反向ready order与3个25MiB bucket。

先只做诊断，不改变同步：要求CPU拓扑稳定、HIP每个参数恰好ready一次、两rank order一致、
zero/unused/shared参数显式。只有至少两个bucket能在完整backward前ready，才进入Event+async
all-reduce overlap；否则调整bucket order或关闭当前重叠假设。
