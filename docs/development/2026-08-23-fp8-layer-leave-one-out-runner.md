# FP8逐层leave-one-out搜索器

## 为什么需要

Exp139定位到残差抵消层，Exp140只把这些“看起来最危险”的block恢复FP32，但T512误差反而
更差。这说明误差放大的位置不等于误差来源。继续凭layer drift图片手选层会重复同一个错误。

## 搜索边界

`hf_fp8_layer_leave_one_out.py`固定当前retained policy：

```text
E4M3 activation/weight
device Tensor activation amax
attention O-projection output-channel weight scale
weight minimum 0.005
```

每个模型/context先运行一个FP32 oracle和一个完整FP8 baseline，然后为每个block启动一个fresh
process，只把该block恢复FP32。每行比较完整last-token词表logits，记录Max、RMS、top token、
resident/peak、native/fallback、dynamic与post计数。

## 选择规则

排序依次使用完整precision gate、top token、RMS、Max和layer编号。搜索吞吐只作诊断，因为每个
候选只有一个进程且没有轮换顺序。搜索结果不能直接成为模型默认；最佳候选必须再进入独立的
T8/T512、三进程同revision正式矩阵。

## 离线合同

`Benchmark.HfFp8MatrixContract`检查：

- runner固定O-only/dynamic/E4基线；
- layer编号确实进入`--fp8-fp32-layers`；
- Qwen风格config能读出正整数层数，布尔值与缺失字段被拒绝；
- 排名优先完整RMS而不是layer编号；
- 已拒绝的E5模型参数不会重新进入命令。

这个节点只交付可复现搜索工具，不提前宣称任何层会改善精度。
