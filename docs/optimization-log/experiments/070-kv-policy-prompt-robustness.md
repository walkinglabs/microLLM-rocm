# Experiment 070 — 换一句话，layer 1 strict还安全吗

Experiment 067/069只使用同一组官方token循环填满context。一个真正的strict策略不能只对
一道题有效，所以runner加入四种确定性prompt：

```text
repeat    原token序列循环
rotated   从下一个token开始循环
constant  全部使用第一个token
ramp      按固定步长遍历词表
```

## layer 1被反例推翻

DeepSeek、T32/512/2048 B1，并补ramp T512/2048 B8：

| pattern | 通过 | 稳定失败 |
|---|---:|---|
| repeat | 3/3 | — |
| rotated | 3/3 | — |
| constant | 0/3 | T512 max_abs 15.829、RMSE 2.995、token分叉 |
| ramp | 3/5 | T512 B1/B8 RMSE 0.097/0.088 |

layer 1总计只过9/14，不能继续叫robust strict。

## 最小扩展搜索

constant T512上，前4/8/14层FP32和全FP32都通过。选择最小的前4层，再重跑全部14 case：

```text
14/14 pass
worst max_abs = 0.1821
worst RMSE    = 0.0328
finite/top/suffix全部一致
```

![KV policy prompt robustness](../assets/kv-policy-prompt-robustness.svg)

前4层FP32、其余24层BF16时，DeepSeek Cache仍比全FP32小`1.75×`。同binary、策略顺序
交替的六shape性能矩阵中，最差decode比`0.9695×`、最差E2E比`0.9719×`、最大peak比
`1.0075×`，在3%边界附近但没有越过。

## 决定

- per-layer API继续保留；
- layer 1降级为“原repeat prompt的最小补偿”，不再推荐为strict；
- 当前固定DeepSeek的**robust-strict**配方改为layers `0,1,2,3` FP32，其余BF16；
- 全局默认仍是全FP32，uniform BF16仍是速度/显存选项；
- robust只覆盖当前checkpoint和四类确定性prompt，新模型仍必须重跑。

这是用更多显存换更强证据，不是免费优化。
