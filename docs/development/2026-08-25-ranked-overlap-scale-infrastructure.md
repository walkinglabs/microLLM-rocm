# 2026-08-25 — ranked overlap context-scale infrastructure

Experiment 270关闭Model-S T32 reducer局部搜索。新节点改变workload尺度，不再改Event/Stream/
bucket实现。

## Context contract

worker、launcher和通用matrix新增显式`--context`：tiny只允许4；Model-S允许1–512。两个rank的
确定性local batch和CPU global batch使用同一context，所有JSON都回报并校验该值。

新增专用`ranked_overlap_context_matrix.py`，只比较同步`bucket-views`和`overlap-views`，默认
T32/T128。每个run反转context/policy顺序；第1步cold，后续step形成steady样本。每个context
分别报告finish、forward/backward、total CV、current/peak和精度，不把不同尺度求一个平均分。

## Rejected assumption

第一次pilot在T128同步views后主动停止。参数/CPU/loss都通过；真正失败的是runner错误要求T128
peak等于T32的324,929,288 bytes。T128激活使peak合理上升到364,546,568 bytes。合同改为每个
run只要求`peak >= current`，最终在同一context内比较同步/overlap exact memory；精度门没有放宽。

## Pilot

一次T32/T128完整pilot：

| Context | Finish speedup | Total speedup | F/B added | Current added | Peak added |
|---:|---:|---:|---:|---:|---:|
| 32 | 2.139× | 1.039× | 0.922 ms | 0 | 0 |
| 128 | 1.955× | 0.959× | 1.521 ms | 0 | 0 |

T128 CPU Max/RMS为`0.003842/2.595e-6`，loss差`1.812e-5`；所有rank exact、peer failure通过。
单次结果只验证尺度runner，不能关闭track。

完整RCCL标签48/48；context静态合同加入CTest；测试文件审计增至124。下一提交从干净revision
运行T32/T128、两策略各三次。
