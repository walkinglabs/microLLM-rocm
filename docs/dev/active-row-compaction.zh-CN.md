# Active-row compaction：空座位不再假装成请求计算

## 1. 上一版浪费在哪里

Continuous scheduler固定有`max_slots`个座位。上一版为了让模型接口总看到固定batch，会给空座位
塞入dummy token：

```text
真实row: B(position=5) → 正常计算
空row:   dummy(0)      → 完整计算 → position变1 → 再清空整行
```

结果虽然正确，但一次无用请求会经过Embedding、每层Attention、FFN和output head，最后还把整条
KV capacity清零。

## 2. 新接口只带真实row

```cpp
model.forward_cached_active_rows(
    active_tokens, shared_cache, active_rows);
```

例如固定4槽只有row 1和row 3活跃：

```text
active_rows   = [1, 3]
active_tokens = [token_B, token_D]
```

模型只建立row 1、row 3的B1 Storage view，只推进这两行，返回`[2,1,V]`。scheduler再把两行logits
放回固定slot位置。row 0和row 2的完整capacity逐项不变，position也保持0。

## 3. 什么时候仍走原batch快路径

如果所有slot都有真实请求，而且position完全相同，仍调用原来的并行batch路径：

```text
full slots + uniform positions → existing forward_cached fast path
其他情况                       → compacted active-row oracle
```

所以这个改动不会为了消除dummy而破坏理想uniform路径。

## 4. 指标怎样变化

- `dummy_decode_rows`：真正送进模型的假row，候选变为0；
- `inactive_rows_skipped`：这次跳过多少空row；
- `compacted_batch_decode_calls`：多少scheduler batch call使用active列表；
- `divergent_batch_decode_calls`：active rows之间position确实不同时才增加；
- logical rows、slot utilization和Cache active/allocated不应改变。

## 5. MI300X Release结果

五个divergent shape相对上一版加速1.134×–1.348×，continuous/reference从0.748×–0.858×提高到
0.935×–0.985×。严格交替A/B的两个重点shape仍有1.292×和1.226×，而reference漂移不到0.8%。

Cache容量没有减少：compaction减少的是本步计算，不是固定KV预留。下一阶段仍需要positions-aware
并行Kernel，才能让多个真实但不同position的row一起算，而不是逐row B1。

实验与图见 [Experiment 097](../optimization-log/experiments/097-active-row-compaction.md)。

下一步已经在[positions-aware decode](positions-aware-decode.zh-CN.md)完成：多个真实但position不同的
active row不再逐个B1，而是让RoPE、KV store和Attention直接读取逐row位置表。
