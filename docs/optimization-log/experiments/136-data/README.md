# Experiment 136 data

Exp135 shared dynamic activation T512的rocprofv3复测。保存两模型命令、独立3×0/0预检、parsed
summary和kernel/API stats。大型逐调用JSON/PFTrace可由命令重建，不进入Git。

沿用Exp134分类边界；known-forward只含dynamic三段、Tensile GEMM和fallback。other仍是
whole-process mixed。共享前后GEMM/other calls不变，launch减少量与三段调用减少量一致。
