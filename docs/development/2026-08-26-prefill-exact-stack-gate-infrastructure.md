# 2026-08-26 — 选择性exact stack完整模型门

这个runner回答一个很窄的问题：把以前分别测过的exact core和exact O按batch组合后，相对真实upstream
是否同时更准确、也不明显变慢。

策略是固定的，不在测量后挑结果：B1保持upstream；B2/B4使用QK、P×V和O；B8只使用QK和P×V。
每个进程都检查实际注册的index、缓存entry、hit、miss和dispatch次数。precision保存完整BF16 cache和
151,936个logits；performance使用相反顺序，避免永远让同一策略先跑。

准入门与前面相同：全局Max和RMS都至少改善10%，每个batch至少0.95×，峰值显存和分配不得暗中增加。
runner只生成原始JSONL、汇总JSON和SVG，不改变引擎默认行为。
