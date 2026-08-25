# 2026-08-25 — ranked persistent bucket result

![Ranked persistent buckets](../optimization-log/assets/ranked-persistent-buckets.svg)

三策略正式矩阵确认persistent warmup后backend allocation为0。相对transient，Reducer/完整step
为1.539×/1.250×；相对逐参数，完整step为1.056×，Reducer仍为0.933×。

plan容量124.69MB/rank，常驻+62.34MB，峰值相对逐参数+124.69MB；57+57次copy仍在。
正确性与故障门通过，因此显式保留但不默认。下一节点用rank gradient views删除unpack Storage/
copy，再决定是否值得迁移ready overlap。
