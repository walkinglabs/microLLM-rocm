# Experiment 067 — 一层用FP32，能不能修好整本BF16草稿

## 从哪个失败开始

Experiment 065的uniform BF16 Cache在Release矩阵中只有一条完整logits失败：DeepSeek
T512 B1的RMSE为`0.058645`，超过固定门`0.05`。全部Cache退回FP32会丢掉一半显存收益；
问题变成：能否只让极少数敏感层用FP32？

## 接口

`KVCache`可以接收每层dtype列表。模型在第`i`层读取`layer_dtype(i)`，Storage、prefix、step
store和cached Attention继续走已有FP32/BF16合同。CLI用：

```text
--kv-cache-dtype bf16 --kv-cache-fp32-layers 1
```

编号从0开始。这是显式策略，不根据模型名字偷偷硬编码。

## 搜索与反例

先在DeepSeek T512 B1搜索：

| policy | RMSE | 状态 | 相对全FP32 Cache缩减 |
|---|---:|---|---:|
| uniform BF16 | 0.058645 | fail | 2.000× |
| last 1 / 2 / 4 | 0.0572 / 0.0553 / 0.0526 | fail | 1.93–1.75× |
| last 8 | 0.0482 | pass | 1.556× |
| layer 0 | 0.0268 | pass | 1.931× |
| layer 1 | 0.0395 | pass | 1.931× |
| layer 2 | 0.0383 | pass | 1.931× |

只看T512会错误地选择layer 0。反驳实验换到T32 B1：layer 0最大误差升到`0.313`并失败；
layer 1为`0.203/0.0391`并通过，layer 2最大误差`0.303`也失败。因此选择**仅layer 1
FP32**，不是“前面随便一层都可以”。

## 完整精度矩阵

Qwen/DeepSeek、T32/512/2048、B1/B8共12条全部通过：

```text
maximum absolute error max = 0.2077
RMSE max                   = 0.0455
finite                     = 12/12
top token / suffix         = 12/12
```

DeepSeek T512 B1从`max_abs/RMSE 0.2245/0.0586`变成`0.1851/0.0395`。Qwen也全部
过门，但它的uniform BF16本来就通过，所以不推荐给Qwen增加这一层FP32。

## Release性能

相对Experiment 065 uniform BF16，三进程中位数：

| 模型 | T | B | decode比 | prepare比 | E2E比 | peak比 |
|---|---:|---:|---:|---:|---:|---:|
| Qwen | 32 | 1 | 0.999× | 1.004× | 1.000× | 1.0000 |
| Qwen | 32 | 8 | 1.022× | 1.106× | 1.032× | 1.0002 |
| Qwen | 512 | 1 | 0.988× | 0.944× | 0.984× | 1.0002 |
| Qwen | 512 | 8 | 1.008× | 1.006× | 1.008× | 1.0013 |
| Qwen | 2048 | 1 | 0.992× | 1.005× | 0.993× | 1.0006 |
| Qwen | 2048 | 8 | 0.991× | 0.925× | 0.941× | 1.0015 |
| DeepSeek | 32 | 1 | 1.002× | 0.988× | 1.002× | 1.0000 |
| DeepSeek | 32 | 8 | 0.997× | 0.996× | 0.996× | 1.0001 |
| DeepSeek | 512 | 1 | 1.000× | 1.007× | 1.000× | 1.0001 |
| DeepSeek | 512 | 8 | 0.976× | 0.991× | 0.980× | 1.0009 |
| DeepSeek | 2048 | 1 | 0.983× | 0.983× | 0.983× | 1.0004 |
| DeepSeek | 2048 | 8 | 0.977× | 0.782× | 0.866× | 1.0019 |

![Mixed-layer KV policy](../assets/mixed-layer-kv-policy.svg)

这张跨时段表最初显示DeepSeek T2048 B8 prepare慢27.9%、端到端慢13.4%。Experiment 069
用同binary、交替顺序重测后得到prepare`0.994×`、E2E`1.011×`，因此旧差异保留为漂移
反例，不再解释为strict策略的因果代价。

Qwen 24层时，1 FP32 + 23 BF16的Cache仍比全FP32小`1.920×`；DeepSeek 28层时为
`1.931×`。T2048 B8分别是`201.6MiB/467.6MiB`，uniform BF16为`193.5/451.5MiB`。

profile的144次cached Attention可分解为138次BF16和6次FP32，精确对应23个BF16层、
1个FP32层、warm-up与measured各3个decode forward。全Kernel只比uniform BF16多0.66%。

## 决定

**keep explicit**。保留per-layer dtype API、CLI、混合字节证据和完整测试；不改变默认策略：

```text
默认安全：全FP32 Cache
显式速度/显存：uniform BF16 Cache
显式strict logits：layer 1 FP32，其余BF16（当前固定DeepSeek实验）
```

layer 1不是对所有模型的普遍定律。Experiment 070随后用constant/ramp推翻其prompt鲁棒性，
当前robust-strict配方升级为前4层FP32；本实验保留为最小原prompt策略证据。小于10%的策略
性能结论必须使用Experiment 069的同binary配对协议。
