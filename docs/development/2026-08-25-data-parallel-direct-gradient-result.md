# 2026-08-25 — direct bucket-gradient模型路径拒绝

![Direct bucket-gradient discard](../optimization-log/assets/data-parallel-direct-bucket-gradient-discard.svg)

Direct accumulation把pack/unpack copy全部降到0，communication相对bucket views为2.173×，
peak少13,205,768B；但额外leaf add让forward/backward只有0.830×，total只有0.991×。

45个loss和9次参数门全部通过，说明拒绝原因纯粹是性能。模型C++/CLI route已经撤回，
保留leaf accumulation target作为真正producer out-kernel的独立基础。
