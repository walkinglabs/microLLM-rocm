# 2026-08-25 — 当前双卡先把审计成本单列

双卡20-step loss从2.75降到0.55，参数最大差为0，RCCL配置14/14通过。

![Current data parallel audit](../optimization-log/assets/current-data-parallel-audit.svg)

steady total 2.290ms中，forward/backward 1.565ms、communication 0.350ms、optimizer 0.070ms。
剩余0.305ms来自optimizer后全参数host一致性检查等未单列工作，占13.32%。

生产路径不应每步搬回全部参数，但正确性默认也不能静默删除。所以下一节点增加独立计时和
显式interval，默认仍每步检查。tiny只有一个bucket，真实overlap留到测量边界干净之后。

