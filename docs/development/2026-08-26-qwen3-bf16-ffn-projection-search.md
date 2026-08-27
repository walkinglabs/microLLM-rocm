# Qwen3 BF16 FFN投影搜索

日期：2026-08-26
状态：三投影共同作用定位完成

框架新增`Bf16FfnWeightScope`与CLI `--bf16-ffn-weight-scope`。准备仍然事务式和one-way；
partial projection block使用可读Linear fallback，all-BF16融合不变。

两个最小层集合的12个partial scope全部选320，两个all scope都选25。20行raw、summary、
CPU模型/CLI正反例和Python evidence通过。CPU 432/432、ASan/UBSan 429/429、PyTorch-enabled
CPU 434/434；HIP默认融合路径仍为214/214，partial scope由真实Qwen3 12/12验证。默认策略没有改变。
