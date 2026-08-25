# 2026-08-24 — 安全地在 default Stream 和 Graph Stream 之间交接

## 不是把开关重新打开

如果非默认Stream还在读一块内存，default Stream就不能把这块内存重新分配。新的API先执行
device-wide synchronize，证明所有Stream都完成，再开启下一段default-Stream复用。

它相当于交通路口的清场：确认横向车辆全部离开，才给纵向放行。不能只把红灯强行变绿。

## 状态机

```text
DefaultOnly
  ├─ default工作：保持DefaultOnly
  └─ 任意non-default提交：进入Disabled

Disabled
  ├─ 普通enable：拒绝
  └─ quiesce + device synchronize：回到DefaultOnly
```

Stream对象存在本身不是下一阶段的持续危险；危险来自提交的工作。因此旧Stream在handoff后可以保留，
但它下一次Graph launch、Event、copy或Kernel提交会立即重新关闭pool。

## 为什么使用全设备同步

这是第一版reference合同。全设备同步容易解释和测试，但可能太慢。后续可以把它替换成：

- 指定Stream Event完成；
- 每个退役block记录最后使用Event；
- 只等待真正冲突的size class。

在reference正确前直接写Event allocator，很难区分生命周期bug和性能bug。

## 模型结果怎样解释

三次handoff分别发生在warmup、snapshot backward和比较backward之前。Qwen两个context和
DeepSeek T8重新获得相同gradient地址；DeepSeek T512仍有allocator顺序反例。这说明API恢复了
原default阶段语义，却没有保证所有shape稳定。

## 使用限制

- API会等待整张GPU，不能放进希望通信/计算重叠的热路径而不测量；
- 调用者仍要在Graph launch前逐项检查workspace snapshot；
- handoff后任何非默认提交都会关闭pool；
- 多GPU时每个device/rank独立交接，不代表RCCL全局同步；
- 当前默认训练路径完全不调用这个API。

## 发布验证

CPU 335/335、ASan/UBSan 333/333、PyTorch-enabled CPU 309/309、完整CPU+HIP
528/528（3个条件跳过）、HIP标签181/181、RCCL 14/14、multi-GPU 12/12。覆盖清单注册
98个测试文件；CPU覆盖率为78.4% lines、86.6% functions、59.1% branches。
