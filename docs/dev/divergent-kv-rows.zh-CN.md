# 不同页数的KV row：两位同学可以写到不同位置吗？

把batch想成一排同学，每个row有一本KV草稿本。静态batch要求大家总在同一页：

```text
row 0: 已写3页
row 1: 已写3页
```

如果row 0提前结束并换成新请求，它应该从第0页开始；row 1还要接着第3页：

```text
row 0: position 0
row 1: position 3
```

这时不能拿一个共同`position`糊弄。RoPE角度、K/V写入位置、Attention能看到的旧页数都必须逐row
决定。

## `forward_cached_rows()`现在怎样做

第一版先追求容易证明：

1. 大Cache仍是同一块`[B,H,capacity,D]` Storage；
2. 为row 0建立一个共享Storage的B1 view，只暴露它自己的prefix；
3. 调用已经验证过的B1 cached forward；
4. 为row 1重复；
5. 把每行logits合并回`[B,1,V]`；
6. 每个row只把自己的position加1。

因此`[0,3]`一次decode后变成`[1,4]`，下一次变成`[2,5]`。row 0的Attention不会读取第1–3页
残留，row 1也不会错误回到第0页。

## 为什么这还不是最终快路径

分叉时会串行做B次B1 forward，logits还需要同设备copy合并。它是正确性oracle，不是continuous
batching加速声明。全部position相同时，API直接转回原来的并行`forward_cached()`。

下一版HIP Kernel要同时接收`positions[B]`：

- RoPE按row读取position；
- K/V store写`positions[row]`；
- cached Attention只看`positions[row] + 1`；
- 一次并行输出全部row。

## 测试怎样抓错

B2先prefill到`[3,3]`，清空row 0得到`[0,3]`。实际B2结果分别与两个独立B1比较：一个从空Cache
开始，另一个保留3-token prefix。测试覆盖FP32/BF16、连续两步、CPU/HIP、Storage地址不变、执行
期间0次D2H，以及reset最大row后logical prefix从5缩到2。

实验记录见[Experiment 093](../optimization-log/experiments/093-divergent-row-cache-reference.md)。
