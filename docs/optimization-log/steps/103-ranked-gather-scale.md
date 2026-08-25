# Step 103 — Ranked persistent gather-scale

Status: implemented; formal Model-S measurement pending

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

实现使用每bucket持久device descriptor buffer。每步host只刷新`source pointer + begin + end`，
向通信Stream复制很小的描述表；一个Kernel用目标offset二分找到source，完成gather与local scale，
随后直接all-reduce。gradient地址没有被假设为跨step稳定。

Tiny三步得到pack/leaf/bucket scale全0、gather-scale `[1,1,1]`，每步descriptor 288 bytes，
rank/CPU门通过。Model-S T32 pilot得到每步3次gather、1,368-byte descriptor、current/peak只增
1,368 bytes，完整参数精确一致；同步对照出现77% CV，所以pilot的4.45x无效，只保留候选约
7.95ms这一诊断。完整RCCL标签53/53、`DistributedRank.*` 10/10；正式T128三轮仍待干净
提交。
