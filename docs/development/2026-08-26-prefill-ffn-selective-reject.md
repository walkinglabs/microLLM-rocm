# 2026-08-26 — 选择性gate/up通过速度门，但RMS改善不够

B1/B2/B8的gate+up使用296100，B4保持default。16个precision和16个反向排序performance进程全部
完成。四个prefill比值为0.993×、0.981×、1.005×、1.002×，peak和allocation不变。

全局完整logit Max改善12.0%，通过10%门；RMS只改善3.3%，未通过。候选因此拒绝。B2/B8数值明显
改善，但保持default的B4成为candidate的全局Max/RMS上限。

下一步是最后一个直接反驳：B4也使用296100。虽然operator M8192是0.941×，仍需实测整模是否低于
0.95；如果端到端性能或数值双门任一失败，就删除FFN模型路由并关闭vendor solution线。
