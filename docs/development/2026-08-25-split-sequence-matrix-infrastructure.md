# 2026-08-25：Split-sequence搜索与画图工具

## 为什么单点3.64倍还不能保留

T2048/B2/BF16、S8 pilot得到0.0751ms，而当前fused是0.2732ms，Event比为3.64x；完整context
Max/RMS为3.67e-9/1.08e-9，热分配与payload传输为0。

但一个S8单点可能刚好适合这组shape。T512可能被第二个Kernel拖慢，FP32和BF16可能需要不同S，
B1与B2也可能因为block数量不同而选择不同。单点只能让我们继续测，不能让候选进入模型。

## 搜索合同

`cached_attention_split_matrix.py`固定搜索：

```text
T = 512 / 2048
B = 1 / 2
cache = FP32 / BF16
S = 1 / 2 / 4 / 8 / 16
每格3个新进程
每项3次热身 + 20次正式测量
```

总计120条raw。每个candidate都重新测当前fused，避免拿旧进程时间作分母；进程交替
forward/reverse顺序。每条记录必须满足：

- 完整split context Max不超过8e-4、RMS不超过8e-5；
- current/split Event和wall P50/P95均为正数；
- 计时区间H2D/D2H为0；
- 热身后backend allocation为0；
- partial blocks、combine blocks和partial bytes与shape公式完全一致。

运行器为每个shape选择Event speedup最大的S，1.05x才通过算子门。它保存raw JSONL、聚合JSON和
`split-search.svg`，图中绿点表示过1.05，红点表示没有过门。

## 合同测试

CPU测试用2个sequence、3个S、2次运行构造12条伪记录。它验证顺序轮换、6个candidate聚合、
2个winner、S4选择、门限和SVG内容。真实MI300X pilot另外验证benchmark输出可以通过同一合同。

下一节点才运行120个真实进程，并依据T512/B2反例决定保留、限制shape或完全拒绝。
