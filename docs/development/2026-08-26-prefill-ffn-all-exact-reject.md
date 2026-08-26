# 2026-08-26 — all-exact gate/up仍被RMS否决

B1/B2/B4/B8的gate+up全部使用296100。16+16个Release进程显示，四个端到端prefill为
1.000×、0.983×、0.964×、0.991×；因此M8192的局部0.941×没有让整模跌破0.95。

数值仍失败。全局Max改善35.5%，但RMS从0.0002294升到0.0002427，恶化5.8%。候选拒绝，FFN
vendor-solution模型线关闭。不能用更漂亮的最大误差掩盖整体误差能量变大。

在删除scope前只做一次因果trace：exact Attention+exact gate/up应让SwiGLU之前全部exact。如果down
成为第一处差异，就记录后删除候选CLI/runner；不会再次把已拒绝策略包装成默认优化。
