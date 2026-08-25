# Split-P×V怎样接进模型而不偷偷改默认

## 三条路线必须互斥

模型现在有三条uniform cached decode研究路线：

```text
full split       score/softmax/P×V都分段，快但已知完整logits失败
materialized     score并行，softmax/P×V保持原顺序，当前长上下文Auto
split P×V        score/softmax保持原顺序，只分段最后的value累加
```

同时打开任意两条都会在运行前失败。`split P×V`默认splits=0，只有显式设置才会运行：

```cpp
model.set_cached_attention_split_pv(16, 2048);
```

```bash
--cached-attention-pv-splits 16 \
--cached-attention-minimum-sequence 2048 \
--cached-attention-materialized false
```

JSON必须报告实际splits和minimum。未达到minimum的早期prefix走旧路径；positions-aware/divergent
serving不使用该实验路线。

## 为什么current要显式materialized=true

本次问题不是“split-P×V比最老Kernel快多少”，而是“它能否替代当前已保留的长上下文路径”。所以
模型runner的current显式打开materialized，candidate显式关闭materialized并打开P×V S16。这样
比较的是running best，而不是一个更弱的旧基线。

接线测试覆盖默认值、非法split、三个策略两两互斥、CPU tiny cached logits，以及CLI二进制参数和
JSON字段。真正是否保留仍由官方DeepSeek完整logits与三对性能决定。
