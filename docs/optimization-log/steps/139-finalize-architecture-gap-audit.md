# Step 139 — Cached Attention finalize architecture gap audit

Status: completed by Experiment 323

当前finalize为42.27%最大项，但相邻路线已经有证据：

- 64/128线程保序mapping：0/16过性能门；
- split-P×V：operator快，完整logits失败；
- exact GQA value reuse：位级正确但仅0.454–0.635×；
- rocWMMA online：operator快，完整模型0.777–0.906×；
- materialized scores：已成为当前保留默认。

下一节点先做gap audit，不立即写Kernel。候选必须说明它与上述路线的结构差异、减少哪类全局内存或
同步、预期资源占用，以及哪个最小operator实验能推翻它。没有新架构差异就停止cached finalize局部线，
转向33.25%的GEMM或更高层serving架构。

审计选择一个未覆盖假设：DeepSeek width128使用原生128-lane max/sum reduction与128-column P×V。旧
128-thread mapping仍模拟256 logical lanes；新候选改变tree、stride和shared reduction footprint，数值顺序
也改变。下一步先做完整输出operator矩阵。
