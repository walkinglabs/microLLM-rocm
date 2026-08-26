# Step 141 — Remove native128 and hand off to GEMM gap audit

Status: completed by Experiment 325; decode rows2 selected

删除native128 public research API、HIP Kernel、benchmark flag、formal runner和candidate-specific tests。保留当前256
默认、Experiment 323/324文档、raw矩阵与SVG。源码缺失测试阻止candidate复活。

随后审计当前33.25% GEMM：区分output head、QKV/O、gate/up/down与其他shape，核对已有BF16 solution、
grouped和precision路线覆盖范围。只有未被旧实验覆盖且有当前时间占比的shape才进入新节点。

clean decode rows2未被旧rows256/512/1024矩阵覆盖。DeepSeek 65193稳定且exact，Event/wall
1.814×/1.519×；只进入固定DeepSeek模型门。
