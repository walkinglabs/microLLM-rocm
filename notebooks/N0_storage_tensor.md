# N0 — 从数组到 Storage 与 Tensor

## 先预测

给定行主序矩阵：

```text
0 1 2
3 4 5
```

在运行代码前写出它的 shape、stride，以及 transpose 后的逻辑读取顺序。

## 最简单路径

构建并运行：

```bash
./scripts/configure.sh -DMICROLLM_ENABLE_HIP=OFF
./scripts/build.sh
./build/examples/microllm_n0_ppm /tmp/n0.ppm
```

示例创建一个 `8x8` CPU Tensor，返回共享 Storage 的 transpose view，再按逻辑
顺序输出 PPM。当前金标准为：

```text
checksum_fnv1a=17940541173909021477
```

## 可见失败

如果 transpose 只交换 shape 而不交换 stride，输出像素顺序和 checksum 会变化。
如果 view 复制 Storage，修改 view 后原 Tensor 不会变化。对应测试位于
`tests/core/tensor_test.cpp`。

## 当前契约

- Storage 管理共享内存生命周期；
- Tensor 解释 shape、stride、dtype 和 offset；
- transpose 和 slice 不复制；
- contiguous 按逻辑顺序复制；
- 越界 view 构造必须失败；
- CPU float32 是 N0 唯一可操作数据路径。

## 下一步问题

普通 CPU 指针不能由 GPU kernel 直接使用。N1 将加入 HIP allocation、copy、
Stream、Event，并让 CPU/HIP 执行同一组算子用例。
