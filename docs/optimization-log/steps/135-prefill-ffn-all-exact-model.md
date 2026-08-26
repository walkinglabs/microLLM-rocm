# Step 135 — All-batch exact prefill FFN gate/up rebuttal

Status: completed by Experiment 319; rejected, track closed

Experiment 318中B1/B2/B8数值改善，但保持default的B4成为全局Max/RMS上限。最后一个直接反驳让
B1/B2/B4/B8的gate和up都使用296100：

- baseline仍是完全真实upstream；
- candidate每个batch注册1个scope key，每forward命中56次；
- 16个fresh precision与16个反向排序performance进程；
- 完整cache/logits、prefill、peak、allocation、token；
- Max/RMS同时改善≥10%，每个batch端到端prefill≥0.95×。

这里允许operator M8192的0.941×进入整模反驳，只因为模型端到端门仍独立存在，不代表降低了算子门。
任一模型门失败就删除CLI/模型scope并关闭vendor FFN solution线。

结果：端到端prefill为1.000/0.983/0.964/0.991×，性能门通过。Max改善35.5%，但RMS恶化5.8%。
候选拒绝，vendor FFN model line关闭。删除前只保留一次exact gate/up后的down-first因果trace。
详见[`Experiment 319`](../experiments/319-prefill-ffn-all-exact-reject.md)。
