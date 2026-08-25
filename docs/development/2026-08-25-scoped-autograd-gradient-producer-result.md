# 2026-08-25 — scoped Autograd producer拒绝

![Scoped Autograd producer discard](../optimization-log/assets/scoped-autograd-gradient-producer-discard.svg)

五shape全部gradient exact且地址稳定，每次少一个logical allocation；但Event只有
0.976×–1.035×，Wall 0.991×–1.018×，0/5过1.05门。

普通first leaf assignment已经接管producer Tensor，因此scoped状态机没有可测收益。Autograd
route和target状态API撤回，caller-owned weight-gradient operator保留；下一步进入gradient-ready
顺序与overlap审计。
