# 给空座位换一位新同学：单个 KV Cache row 的 prefill

## 1. 问题是什么

把一个 batch 想成一排座位，每个座位都有一本 KV Cache 草稿本。开始时两位同学都写了 3 页：

```text
row 0: [A A A]  position=3
row 1: [B B B]  position=3
```

row 0 的请求结束后，我们清空它，但 row 1 还要继续：

```text
row 0: [     ]  position=0
row 1: [B B B]  position=3
```

现在新请求 C 带着两个 prompt token 到来。我们需要只把 C 写入 row 0，不能重算或破坏 B：

```text
row 0: [C C  ]  position=2
row 1: [B B B]  position=3
```

这一步叫“单槽位 prefill”。没有它，continuous batching 只能看见空座位，却不能让新请求真正入座。

## 2. 公共接口

```cpp
Tensor logits = model.forward_prefill_cached_row(prompt, cache, row);
```

接口合同很窄：

- `prompt` 必须是非空的 `[1,T]` int32 Tensor；
- `row` 必须在 batch 范围内；
- 目标 row 的 position 必须为 0；
- `T` 不能超过 Cache 和模型容量；
- Cache 的层数、dtype、设备和 Storage 布局必须与模型匹配；
- 成功后只推进目标 row，其他 row 的 K/V 和 position 不变。

如果目标 row 不是空的，接口会明确报错。这样可以防止调用者不小心覆盖仍在生成的请求。

## 3. 第一版为什么先用临时 B1 Cache

第一版选择最容易检查的做法：

```text
新 prompt [1,T]
  → 用已有完整 prefill 写入临时 B1 Cache
  → 每层、每个 KV head 在同一设备上复制到共享 Cache 的目标 row
  → 返回临时 B1 的最后一个 token logits
```

旧 row 不参加模型计算。HIP 上的 K/V 搬运是 D2D，执行区间没有把 payload 拿回 CPU。

它的优点是复用了已经通过数值门的 B1 prefill；缺点也很明确：会临时分配一份 B1 Cache，并做
逐层、逐 head copy。所以它是正确性 reference，不是最终吞吐方案。

## 4. 一次完整状态变化

```text
共同 prefill        [0,0] → [3,3]
row 0 请求结束       [3,3] → [0,3]
新 prompt 填 row 0   [0,3] → [2,3]
两行各 decode 一次   [2,3] → [3,4]
```

最后一步调用 `forward_cached_rows()`。它证明新来的请求能从自己的第 2 页继续，旧请求也能从第
3 页继续，两者不会读错草稿本。

## 5. 测试怎样抓错

CPU 测试对 FP32 和 BF16 分别检查：

- row prefill logits 等于一个独立 B1 模型；
- 未替换 row 的每层 K/V 逐项不变；
- 共享 Storage 地址不变；
- 两行继续 decode 后分别等于各自独立 B1；
- 非空 row、越界 row 和不兼容 Cache 明确失败；
- 全空 B2 Cache 可以直接把 prompt 填入 row 1。

HIP 测试再检查 CPU/HIP logits 对齐、未替换 row 不变，并确认执行期间 D2H payload copy 为 0。

测试文件：

```text
tests/model/model_test.cpp
tests/inference/hip_shape_matrix_test.cpp
```

## 6. 还缺什么

现在“空座位能放入新请求”已经有模型层 oracle，但 scheduler 还没有自动管理这些座位。下一步是
把 submit、完成、reset、row prefill 和 divergent decode 接成一个可测试的 slot scheduler。
再下一步才是把串行 divergent-row 路径改成 positions-aware 并行 HIP Kernel。

实验原始证据见 [Experiment 094](../optimization-log/experiments/094-slot-row-prefill.md)。
