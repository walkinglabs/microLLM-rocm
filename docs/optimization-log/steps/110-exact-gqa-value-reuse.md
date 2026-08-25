# Step 110 — Exact-order GQA value-load reuse

Status: planned

Step 109证明只改变P×V加法树也会使完整logits失败。下一候选不分sequence：同一KV head对应的
6或7个query heads仍各自按position 0→T累加，但一个线程每个position只读取一次value，然后更新
多个独立head accumulator。

```text
parallel score
exact 256-lane softmax → probabilities[B,H,T]
grouped GQA P×V:
  for position 0→T:
    value = V[kv_head, position, column]    # 只读一次
    head0 += p0[position] * value
    ...
    headR += pR[position] * value
```

合同：

1. 每个head的乘法与累加顺序必须与materialized current位级相同；
2. 覆盖H14/KV2/D64与H12/KV2/D128、T512/T2048、B1/B2、FP32/BF16；
3. 保存probability bytes、allocation、backend和完整context；
4. Event至少1.05x、wall至少1.02x才进入模型；
5. 若operator不能补回exact-softmax全局buffer成本，就关闭exact-finalize局部线，转向serving batch
   或新的端到端profile，而不是改变加法顺序。
