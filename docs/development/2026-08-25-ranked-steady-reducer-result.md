# 2026-08-25 — ranked Model-S steady reducer result

![Ranked steady reducer](../optimization-log/assets/ranked-steady-reducer-discard.svg)

三次三步正式矩阵把cold与steady分开。cold bucket看似快1.321×；steady却从逐参数2.837ms退化
到bucket 4.205ms，Reducer用时多48.2%，完整step用时多17.3%。bucket steady CV仅2.72%。

每个steady bucket step精确产生60次backend allocation、124,689,408 bytes和57+57次
pack/unpack；per-parameter均为0。完整参数、CPU、loss与peer-failure门通过。因此拒绝transient
bucket的steady性能解释，下一节点只测试persistent rank Storage，不提前加入overlap。
