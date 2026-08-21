# Experiment 104 — 0.000669的margin把两条生成路径分开

Experiment 103发现DeepSeek short在S1/S2与S4/S8间稳定分叉。本节点先增加默认关闭的top-2
诊断，再做只关闭batched prefill的反驳实验，不改模型权重、Cache dtype或decode batch。

![DeepSeek slot divergence](../assets/continuous-divergence.svg)

## 首个分叉

18/18 fresh processes通过且逐字段稳定。request 5、generated index 4、cache position 36处：

| case | source/batch | top-1/logit | top-2/logit | margin |
|---|---|---|---|---:|
| S1 | uniform B1 | 23606 / 18.917286 | 1196 / 18.901663 | 0.015623 |
| S2 | positions B2 | 23606 / 18.915491 | 1196 / 18.904139 | 0.011353 |
| S4 default | positions B4 | 1196 / 18.890232 | 23606 / 18.889563 | 0.000669 |
| S8 default | positions B8 | 1196 / 18.890232 | 23606 / 18.889563 | 0.000669 |

GPU argmax与host top-1 18/18×全部决策一致。S4/S8虽然decode batch不同，generated index0–4的
top-2 logits逐值相同；B1/B2在prefill输出第一个token时已经出现logit差异。

## 反驳decode假设

只把`batch_equal_length_prefill`设为false：

| case | prefill batches/batched calls/rows | decode保持 | 完整输出 | 分叉点margin |
|---|---:|---|---|---:|
| S4 default | 4 / 4 / 8 | B4 | S4组 | 0.000669 |
| S4 serial | 8 / 0 / 0 | B4 | S1组 | 0.011353 |
| S8 default | 4 / 4 / 8 | B8 | S8组 | 0.000669 |
| S8 serial | 8 / 0 / 0 | B8 | S1组 | 0.011353 |

serial反驳组的分叉点logits逐值等于S2。相同decode batch下只改变prefill就翻转完整输出，因此
“decode active batch导致分叉”被推翻；batched prefill是因果变量。

## PyTorch外部门决定是否回退

默认S4与PyTorch只有request 7、token 14不同。serial S4除这个差异外，又让request 5、token 4
从PyTorch的1196变成23606。因此为了跨slot复制S1而关闭batch prefill，会降低外部token对齐并
丢掉已测性能收益；该方案被拒绝为默认值。

保留内容：默认关闭的诊断API、CLI/runner、明确的诊断计时边界，以及serial prefill实验控制。
未改变内容：生产默认仍批量等长prompt。

下一节点交换B2 local rows并使用相同prompt两行，区分正常M-shape数值差异与row/copy错误。

数据见[`104-data`](104-data/)。
