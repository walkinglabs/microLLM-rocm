# Step 114 — Cached step-0 block drift

Status: planned

Experiment 296选择BF16 FFN作为主要放大源。当前trace只支持prefill；本节点先让`--trace-output`在
cached decode的选定`--cache-logits-step`上激活，而且只允许warmup 0、steps 1的诊断运行。

第一阶段固定B1/B2、T2048、step0、两次fresh process：

- FP32 Linear；
- BF16 FFN-only；
- 记录embedding、28个block output、final norm与logits；
- B2两行必须相同，B1/B2每层报告Max/RMS/relative-L2；
- 找到首次非零和首次超过FP32同层10倍的block。

第二阶段只对选中的第一个放大block开启detail，比较attention residual、FFN norm、input BF16、gate、
up、activated、down和block output。trace运行不做性能声明。
