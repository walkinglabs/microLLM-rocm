# Step 52 — public online causal-GQA operator

Status: complete, admit model gate

## Decision

公共BF16→FP32 BTHD operator支持gfx942 native B1/B2并显式fallback。42进程中10/10 native为
1.534×–2.456×当前路径，4个T31/T33/D32 fallback精确但只有0.607×–0.696×。PyTorch、CPU、
HIP batch/tail和CMake Config门通过；下一步只准入模型A/B，默认路由关闭。
