# Experiment 222 — 256个更新节点缩成一个，BF16反例被救回了吗

Status: `keep explicit two-node candidate; no model/default route`

## 上一轮留下的唯一问题

Experiment 221已经让step住进GPU，但每个参数Tensor仍是一个Graph节点。BF16更新Kernel很短，
节点调度成本反而使256个小Tensor只有`0.805×`。

本次只改变descriptor生命周期：

```text
capture前一次性验证并上传：
  parameter / gradient / moment / mirror地址
  每个Tensor元素数和首block

capture内固定两节点：
  device step + correction
  一个multi-tensor AdamW Kernel
```

descriptor上传后变为immutable。相同workspace不能再次prepare，也不能回到会重写descriptor的
普通multi-tensor入口。所有引用的Storage必须活到最后一次Graph launch完成。

## 正确性与传输

- FP32/BF16 moment各用17和1025元素两个不同Tensor；
- 两节点Graph连续重放三次；
- 参数、两个moment和BF16 mirror与eager完整对齐；
- 正式90进程每个执行53步，per-tensor Graph与multi Graph都对齐eager；
- 最大状态sample误差仍为`7.45e-8`；
- descriptor只在preparation上传，timed region三类payload transfer全部为0；
- 每个`graph-multi`进程的captured nodes严格等于2，与Tensor数量无关。

## 90进程结果

| Moment | Tensor形状 | per-Tensor Graph/eager | multi Graph/eager | multi/per-Tensor |
|---|---:|---:|---:|---:|
| FP32 | 1 × 1K | 0.211× | 0.181× | 0.859× |
| FP32 | 16 × 1K | 1.013× | 2.547× | 2.514× |
| FP32 | 64 × 1K | 1.443× | 10.952× | 7.588× |
| FP32 | 256 × 1K | 1.486× | 36.162× | 24.338× |
| FP32 | 16 × 256K | 0.870× | 0.908× | 1.044× |
| BF16 | 1 × 1K | 0.192× | 0.197× | 1.025× |
| BF16 | 16 × 1K | 0.674× | 2.757× | 4.092× |
| BF16 | 64 × 1K | 0.767× | 10.813× | 14.102× |
| BF16 | 256 × 1K | 0.806× | 36.929× | 45.841× |
| BF16 | 16 × 256K | 0.819× | 1.630× | 1.989× |

![Stable-descriptor AdamW multi Graph](../assets/adamw-graph-multi.svg)

巨大的小Tensor倍数不是计算变少：总元素和AdamW公式相同。旧eager/per-Tensor路径为每个Tensor
提交一次短Kernel；multi路径把全部block放在一个grid，Graph又把整个区域一次提交。因此，
Tensor越多、每个越短，提交差异越明显。

## 仍然失败的地方

- 单Tensor没有可合并对象，两节点Graph固定成本使它更慢；
- FP32 16×256K仍是`0.908×`，说明大连续FP32读写已接近带宽主导；
- preparation约0.17–0.21ms，必须在多次重放中摊销；
- 真实训练的Autograd可能每步替换gradient Storage。descriptor保存的是地址，不是参数名字；地址
  一变，旧Graph不能继续使用。

## 决定

- 保留immutable descriptor、两节点FP32/BF16 multi Graph和训练层显式workspace；
- 不设默认、不接官方模型，因为gradient地址跨backward尚未稳定；
- optimizer-only Graph的kernel/descriptor问题已解决，不再扫描block size；
- 下一节点先测真实训练两步gradient地址是否稳定。如果不稳定，必须先做stable gradient buffer；
  如果稳定，再做Qwen/DeepSeek optimizer阶段和端到端门。

原始证据在
[`benchmarks/results/2026-08-24-adamw-graph-multi/`](../../../benchmarks/results/2026-08-24-adamw-graph-multi/)。
