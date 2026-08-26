# 2026-08-26 — cached-prefill FFN gate/up独立scope与整模runner

新增`PrefillFfnGateUpProjection`，只在full cached-prefill时传给每层gate和up。down、Attention、decode、
training、BF16 FFN和FP8 Linear保持独立。CLI只接受HIP、full cache、FP32 FFN权重，并按当前batch真实
descriptor注册一个共享key；每层gate/up共56次dispatch。

模型runner在测量前固定B1/B2/B8使用296100，B4保持default。baseline是完全真实upstream，不叠加之前
被拒绝的Q/K/V/QK/P×V/O策略。precision与performance各16个fresh process，后者第二轮反向排序；准入
仍要求全局Max/RMS都改善至少10%，每个batch prefill不低于0.95×，并检查peak/allocation/token。
