# 2026-08-26 — exact gate/up以后，down成为第一处差异

8个Release进程显示：FFN norm、gate、up和SwiGLU activation在跨batch与同batch重复行上全部位级一致。
down统一成为第一处差异，B2/B4/B8跨batch Max为1.72e-5、1.05e-5、1.43e-5。

这支持“gate/up GEMM是上一阶段首因”的解释，但不改变Experiment 319的模型拒绝。临时完整值比较后
全部删除。下一提交先移除失败的CLI/model scope和两个candidate runner，再用通用operator工具筛
M2048–16384、K8960、N1536的down descriptor。
