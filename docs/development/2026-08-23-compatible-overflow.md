# 2026-08-23：短请求兼容溢出

## 合同

最小桶负载达到 slot 数后，短请求可进入第一个有即时容量的更大兼容桶。默认关闭、提交后不迁移、
长请求不能进入小桶。

## 第一次失败

第一次 official runner 在第一条 overflow route 阻断。环境干净、uniform/fixed token正常；错误是
`active_request_count` 已包含 pending，代码又加一次 `pending_request_count`。max-slots=1测试无法
发现，新增 max-slots=4 精确阈值后修复。

## 正式证据

- Release 319/319；sanitizer 215/215；Python 15/15；
- 54/54 fresh process，pre/post GPU门通过；
- short-heavy route `[0,0,0,0,1,1,1,1]`、overflow count 2；
- 六组对 fixed/uniform token exact；
- short-heavy相对fixed：TPS +12.9%–13.3%，TTFT P95 -60.8%–62.2%，completion P95约-40%；
- long/delayed count 0，candidate/fixed基本中性。

## 决定

保留显式candidate，不设默认。它只修复可装入大桶的短请求；相对uniform仍少约17%吞吐，尾延迟
高23%–35%。下一步先测不同slot比例，不直接引入paged Cache。

详见[Experiment 117](../optimization-log/experiments/117-compatible-overflow.md)。
