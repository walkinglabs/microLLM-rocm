# Step 89 — Ranked Model-S synchronous bucket smoke

Status: planned

tiny无法测bucket性能。下一步扩展rank worker到Model-S B1T32 one-step：两rank各本地batch，和CPU
global B2T32 reference比较完整15,586,176参数值；25MiB应得到3个自然bucket，per-parameter为57次
collective。

由于CPU reference与JSON全参数输出成本很高，先做一次smoke并记录rank组时间、collective、peak和
最大参数差。正确后再设计不输出全部参数的多run摘要；ready overlap仍不接入。
