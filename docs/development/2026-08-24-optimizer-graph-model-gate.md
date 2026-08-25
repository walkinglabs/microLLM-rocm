# 2026-08-24 — 为什么一个更少节点的 optimizer 反而更慢

## micro-benchmark回答的只是局部问题

256个很小Tensor各启动一次Kernel时，启动成本很大。multi Graph把它们合成一个grid，当然可能快
几十倍。真实LLM参数并不是256块同样的小Tensor：embedding和projection非常大，norm/bias很小，
shape分布极不均匀。

当前Hybrid AdamW已经做了一次分层：小Tensor合并，大Tensor保留专用更新。新的Graph候选把所有
Tensor重新塞进一个通用descriptor Kernel，减少了提交，却让大Tensor路径变差。

## 公平比较做了什么

eager和Graph都创建同样的非默认Stream，都在每次backward前执行同样的quiescent handoff。两边
使用相同synthetic finite weights、token、BF16 mirrors、BF16 moments和1M Hybrid阈值。

不同的只有optimizer提交：

- eager：当前Hybrid路径；
- Graph：device step + immutable multi update两节点。

Graph preparation单独计时，不放进steady step；device-wide handoff放进step，因为真实训练不能
省略它。

## 证据优先级

1. loss和参数必须精确一致；
2. 每步gradient snapshot必须匹配；
3. Graph必须真的是2节点且无metadata copy；
4. optimizer phase必须更快；
5. 完整step也必须在两个模型上更快。

候选通过前三项，失败后两项。因此“实现正确”与“应该启用”得到不同结论。

## Qwen T8为什么不能单独保留

Qwen T8完整step快约5%，但optimizer慢约20%。这说明差异来自短backward的排队、同步或测量噪声，
不是目标组件收益。相同Qwen T512完整step慢7%，DeepSeek T8慢13%。预先约定的双模型/目标phase门
不能在看到结果后改成单case。

## 留下什么

底层原语仍有价值：它们说明如何安全表达device step、稳定pointer table和Stream阶段交接。未来
完整训练图可以复用。但当前框架不会给用户一个看似高级、实际更慢的optimizer Graph开关。

## 发布验证

CPU 336/336、ASan/UBSan 334/334、PyTorch-enabled CPU 310/310、完整CPU+HIP
529/529（3个条件跳过）、HIP标签181/181。runtime代码与上一节点相同，RCCL 14/14、multi-GPU
12/12和CPU覆盖率78.4% lines、86.6% functions、59.1% branches继续适用；覆盖清单新增schema
测试后注册99个测试文件。
