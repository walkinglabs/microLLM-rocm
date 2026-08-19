# N1 — CPU 与 AMD GPU 怎样得到相同答案

## 运行前预测

给定两个 `64x64` Tensor，先写下：

1. CPU add 的前几个结果；
2. H2D、Kernel、Event、D2H 的顺序；
3. 如果在完成 Event 之前读取输出，当前证据允许保证什么。

## 运行

```bash
./scripts/configure.sh -DMICROLLM_ENABLE_HIP=ON
./scripts/build.sh
./build/examples/microllm_n1_cpu_hip
ctest --test-dir build -L hip --output-on-failure
```

示例不使用一次全局设备同步修复异步问题。Kernel 和完成 Event 进入同一条
显式 Stream；CPU 只等待这个 Event，然后逐元素比较 CPU/HIP 输出。

## 可见边界

- CPU-only 构建仍然成功，HIP 用例不注册；
- Stream 与 Tensor device 不匹配会在 launch 前失败；
- 非连续 HIP view 目前会被拒绝；
- Event 的单次时间不是 benchmark，不能用于性能结论。

## 下一步

forward 已有 CPU/HIP 共同路径，但训练还需要沿计算图累加梯度。N2 将先在
CPU reference 上实现 eager reverse-mode Autograd 和有限差分检查。
