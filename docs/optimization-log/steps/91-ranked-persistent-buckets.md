# Step 91 — Ranked persistent buckets

Status: implemented, formal clean-revision measurement pending

Experiment 267证明transient bucket steady Reducer比逐参数慢48.2%，并把差额定位到每步60次
backend allocation、124,689,408 bytes以及57+57次pack/unpack。下一节点只改变Reducer Storage
生命周期：第一次建立3个bucket和对应unpack Tensor，后续step复用地址与容量。

必须证明warmup后backend allocation为0、地址稳定、collective/pack/unpack语义不变、三步完整
参数/CPU/loss/故障门不退化。正式矩阵同时报告Reducer、完整step、live/peak显存；如果persistent
仍不能超过per-parameter，就拒绝该性能路线，再考虑gradient views或直接ready overlap。

实现新增move-only `RankGradientBucketPlan`和显式`persistent-bucket`策略。pilot的plan reuse
`[0,1,1]`、backend allocation `[60,0,0]`，steady Reducer约2.78ms；代价是常驻+62.34MB、
峰值相对逐参数+124.69MB。完整参数/CPU/loss/故障门通过。三策略正式矩阵前不作保留决定。
