# 2026-08-24 — 让 AdamW 的步数跟着 HIP Graph 一起走

## 给第一次接触优化器的读者

AdamW不只保存模型参数，还保存两本“小账本”：一阶moment记录最近梯度的大致方向，二阶moment
记录梯度大小。每走一步，它还要知道当前是第几步，因为早期账本里的数字偏小，需要做
`bias correction`。

普通代码把step放在CPU变量里。HIP Graph像一段只包含GPU工作的录像；重放录像时，CPU变量
不会重新执行`step++`。因此上一版虽然录下了AdamW Kernel，却不能正确训练第二步。

## 新的状态由谁保管

`AdamWGraphStepState`在GPU上保管：

- 一个Int32 step；
- 两个FP32 correction；
- 它们都在capture前申请，地址不会随重放改变。

Graph中的第一个Kernel把step加一并计算correction，后面的所有参数更新读取这两个数。因为工作
位于同一Stream，更新Kernel一定看见这一次advance的结果。

## 为什么还要显式同步

checkpoint格式仍把step保存成普通整数。Graph运行期间，CPU不知道它被重放了多少次。调用
`synchronize_graph_step()`会等待设备工作、读取一个Int32并更新optimizer的主机状态。

如果忘记同步，框架会拒绝：

- 普通`AdamW::step()`；
- `state()`，也就是checkpoint读取；
- `load_state()`。

这比静默保存错误step更安全。同步以后，可以继续普通step，也可以正常保存checkpoint。

## API使用顺序

```cpp
auto graph_state = optimizer.make_graph_step_state();
auto graph = HipGraphExecutable::capture(stream, [&] {
    optimizer.step_graph_replayable(graph_state, context);
});

graph.launch(stream);
graph.launch(stream);
stream.synchronize();
optimizer.synchronize_graph_step(graph_state);
```

捕获时哪些参数有gradient、这些gradient的Storage地址是什么，会成为Graph合同的一部分。重放
期间不能替换参数、gradient、moment、mirror或graph state的Storage。

## 测试怎样证明它不是“只会运行”

真实MI300X测试同时覆盖FP32和BF16 moment：

1. 同样的参数和gradient建立eager与Graph optimizer；
2. Graph连续重放三次；
3. 比较参数、两个moment和BF16 mirror；
4. 同步device step，确认值为3；
5. 两边再走一个普通step，确认step 4仍一致；
6. 从step 4重新建立Graph state，确认恢复起点的step 5仍一致；
7. 检查Graph timed region没有payload transfer。

PyTorch oracle也从两步扩到三步。Graph路径通过“Graph等于普通AdamW、普通AdamW等于PyTorch”
两道独立门，而不是用自身输出证明自身正确。

## 性能边界

Graph主要减少CPU提交工作，不能减少AdamW必须读写的参数和moment字节。FP32 64/256个1K
Tensor快约1.43×；单Tensor、大Tensor以及全部BF16 case更慢。它现在是研究原语，不是默认
优化器策略。

下一版会测试一个稳定descriptor描述全部Tensor的multi-tensor Kernel，把“advance + N个更新”
缩成“advance + 一个更新”。只有完整状态和大/小Tensor矩阵都通过，才考虑整模接入。

## 发布验证

CPU 331/331、ASan/UBSan 329/329、PyTorch-enabled CPU 305/305、完整CPU+HIP
521/521（3个条件跳过）、HIP标签178/178、RCCL 14/14、multi-GPU 12/12。覆盖清单注册
94个测试文件；CPU覆盖率为79.0% lines、87.3% functions、59.6% branches。新增Graph实现主要
位于HIP专用分支，因此CPU覆盖百分比诚实下降，真实MI300X测试负责该路径的数值与重放门。
