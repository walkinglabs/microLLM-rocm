# 2026-08-24 — 为什么创建一条 Stream 会改变 gradient 地址

## 两条队伍与一批可重复使用的桌子

default Stream像第一支按顺序工作的队伍。exact-size pool知道这支队伍里的任务严格排队，所以一
张桌子退休后，后面的任务可以安全复用。

HIP Graph需要第二支队伍，也就是非默认Stream。如果第一支队伍把桌子重新分配时，第二支队伍
还在使用旧桌子，就会读写错误内存。当前runtime无法跟踪每张桌子跨队伍的最后使用者，因此采用
最保守规则：第二支队伍一出现，地址复用永久关闭。

## 为什么上一实验的“稳定”不够

上一实验没有创建Graph Stream，Qwen T8/T512和DeepSeek T8地址稳定。这只证明纯default-Stream
环境。真正接Graph时，Stream必须在allocator warmup前存在；这个新条件改变了allocator行为，
于是同样的backward不再返回同一地址。

系统实验必须把准备运行优化所需的全部条件放进去。少一个条件得到的结论不能推广。

## safety gate怎样工作

`AdamWGraphWorkspace`现在同时绑定：

- 创建它的具体AdamW对象；
- 每个参数在准备时有没有gradient；
- 每个gradient的Storage地址。

每次准备launch前，`graph_workspace_matches_current_gradients()`重新逐项比较。任何一项变化都返回
false。正式preflight的12个进程全部在这里停止，Graph从未读过过期地址。

## 下一版不能只是改一个开关

安全的Stream-aware方案需要明确阶段：

```text
default backward完成
→ gradient地址交给optimizer Stream
→ optimizer Stream完成
→ 全部相关地址归还default阶段
```

可以用全设备同步先建立易验证reference，再用Event把等待缩小。只有生命周期测试、跨Stream
反例和性能矩阵都通过，才允许替换permanent disable。

## 发布边界

本节点只有preflight结果，没有Graph吞吐。拒绝运行不安全候选本身就是正确结果。Qwen和DeepSeek
的模型级速度仍使用现有eager/Hybrid AdamW证据，不能用micro multi-Graph的36×替代。

## 发布验证

CPU 334/334、ASan/UBSan 332/332、PyTorch-enabled CPU 308/308、完整CPU+HIP
527/527（3个条件跳过）、HIP标签181/181、RCCL 14/14、multi-GPU 12/12。覆盖清单注册
97个测试文件；CPU覆盖率为78.4% lines、86.7% functions、59.1% branches。
