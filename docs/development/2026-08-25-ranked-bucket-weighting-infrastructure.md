# 2026-08-25 — ranked ready-bucket weighting infrastructure

## 从失败里只改一件事

Step 101已经证明通信能提前，但每步57次leaf scale使整步只有0.9594x。本节点没有换模型、
bucket大小或通信公式，只把同一个local token weight移动到完整bucket上。

```text
旧候选：57个leaf各scale一次 → 3个bucket通信
新候选：57个leaf只报ready → 3个bucket各scale一次 → 通信
```

## 接口与顺序

- `RankCommunicator::enqueue_all_reduce_average_in_place(tensor, local_scale)`在同一通信
  Stream上执行`local scale → RCCL sum → 1/world scale`。
- 非有限或非正scale会在任何collective前被拒绝。
- `RankGradientBucketPlan::begin_overlap_step`接受当步local scale。
- Event只表示默认Stream上的bucket梯度已完成；通信Stream等待后先pack，再scale完整bucket。
- `RankBucketStats.weight_scale_calls`让测试直接核对每bucket恰好一次。
- 新路由名为`bucket-weighted-overlap`；原`overlap-views`不被覆盖，旧实验仍可复现。

## 证据门

- world1 communicator的scale 2得到精确`[2,4,6]`；scale 0被拒绝；
- 同步persistent bucket和ready overlap都验证一次bucket scale；
- Tiny `[B1,B2]`三步：leaf scale `[0,0,0]`，bucket scale `[1,1,1]`；
- rank Max/RMS 0/0；CPU Max/RMS `8.18e-8 / 8.79e-9`；
- 策略的后两步overlap为`[0,1,1]`，later backend allocation为0；
- `DistributedRank.*` 9/9；完整RCCL标签51/51；测试文件审计125/125。

## pilot，不是结论

Model-S T32单轮pilot把steady step从8.669ms降到8.029ms，约1.080x；同步/候选最终
15,586,176个参数逐项完全相同，显存增量0。一次dirty pilot不能准入性能路由。下一节点从本
提交的干净revision运行T128、三轮交替、每策略6个steady sample，并保留完整CPU与参数门。
