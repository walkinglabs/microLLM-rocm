# Step 103 — Ranked persistent gather-scale

Status: planned

Step 102已经把weighted overlap从0.9594x修复到1.0661x，但每步仍向通信Stream提交57次
device-to-device copy，再提交3次bucket scale。下一节点只融合这两件事：

```text
每个bucket持久保存 source pointer + destination offset + element count
→ 一个gather-scale Kernel复制多段gradient并乘local scale
→ RCCL all-reduce average
```

目标计数是pack copy 57→0、bucket scale 3→0、gather-scale 0→3；bucket、Event与collective数量
保持3。必须验证描述地址在每步仍有效、输入范围不越界、同步/候选15,586,176个参数逐项一致、
CPU门与显存门不退化。正式T128三轮整步和leave-one敏感性都改善才保留；否则拒绝融合并停止
当前ranked reducer局部优化线。
