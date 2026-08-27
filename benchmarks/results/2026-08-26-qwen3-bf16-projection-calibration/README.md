# Qwen3 symmetric projection calibration

在同一五case FP32-oracle preflight上比较三条全模型规则：分别让gate、up或down保持FP32，
其他FFN投影与Attention为BF16，Cache为BF16。

```bash
# up保持FP32
python3 benchmarks/single_gpu/qwen3_bf16_gate_fp32_oracle.py ... \
  --candidate-policy micro-mixed-gate-down-bf16

# down保持FP32
python3 benchmarks/single_gpu/qwen3_bf16_gate_fp32_oracle.py ... \
  --candidate-policy micro-mixed-gate-up-bf16
```

| rule | oracle通过 | 最小top-2 margin | 结论 |
|---|---:|---:|---|
| gate FP32 | 4/5 | — | 拒绝 |
| up FP32 | 5/5 | 0.001289 | 进入候选 |
| down FP32 | 5/5 | 0.009707 | 选择进入shape门 |

up/down两个候选都保留56个FFN和112个Attention BF16 tensor，常驻均为1,679,556,608字节，
比当前增加176,160,768字节。因此按最小margin选择down-FP32，不存在内存口径偏置。

[`up-fp32-summary.json`](up-fp32-summary.json)与
[`down-fp32-summary.json`](down-fp32-summary.json)保存五格；[`raw.jsonl`](raw.jsonl)保存40个
worker。这个节点只选择下一候选，尚未通过完整shape或性能门。
