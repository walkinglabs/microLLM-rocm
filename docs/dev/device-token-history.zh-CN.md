# GPU token history：先在显卡上记完，再一次拿回来

生成文字时，每一步都要选出下一个token。旧greedy路径这样做：

```text
GPU算argmax → 把token给CPU → CPU记下来 → GPU算下一步
```

生成8个token就有8次很小的D2H。每次只有几个字节，但CPU必须等GPU，像每写一个字就把整本
草稿本交给老师看一次。

没有stop token、也不做随机采样时，CPU中途不需要知道token是什么。新路径预留一个
`history[new_tokens, batch]`：

```text
GPU argmax直接写history第0行
→ 这个device view直接喂给下一次forward
→ GPU继续写第1、2、3行
→ 全部完成后一次D2H
```

`argmax_out_`和`argmax_last_dim_out_`不会偷偷分配输出，而是检查调用者给出的Tensor：shape必须
匹配、dtype必须是Int32、设备必须相同、布局必须连续。

## 为什么stop和sampling不走快路径

- stop token：CPU每一步要决定请求是否结束；
- 随机sampling：当前随机数与top-k reference在host；
- 不同row提前结束：调度器要立即释放对应Cache。

这些路径继续逐步读取，不能为了少同步改变语义。

## 怎样证明没有少算

- `measured_forward_steps == batch × decode_tokens × steps`；
- D2H bytes不变，只是calls减少；
- GPU history逐项等于旧baseline和CPU；
- KV active/capacity不变；
- peak最多增加`batch × new_tokens × 4` bytes；
- sampling与stop的旧测试继续通过。

正式数据和失败前史见[Experiment 090](../optimization-log/experiments/090-device-token-history.md)。
