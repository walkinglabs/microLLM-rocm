# 2026-08-25 — ranked gradient-as-bucket view result

![Ranked gradient views](../optimization-log/assets/ranked-gradient-bucket-views.svg)

四策略正式矩阵确认views把unpack 57→0、plan容量减半到62.34MB，current回到逐参数水平；
peak仍+62.34MB。相对persistent-copy Reducer/total为1.120×/1.006×，相对逐参数为
0.984×/1.055×。

完整参数、CPU、loss与故障门通过。views显式保留、不默认；下一节点迁移rank-local
gradient-ready Event overlap，并要求不增加views的current/peak。
