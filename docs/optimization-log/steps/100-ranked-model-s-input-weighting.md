# Step 100 — Ranked Model-S uneven-input smoke

Status: complete

tiny已证明公式。下一节点固定Model-S T32、rank rows `[1,2]`、一步：equal-only继续有界拒绝；
token-weighted使用同步25MiB bucket views或per-parameter，比较57个Tensor/15,586,176值与CPU B3。

必须记录local/average tokens、scale、rank/CPU Max+RMS、loss差、显存和故障。weighted overlap继续
禁止，直到scale能在每个ready bucket enqueue前完成。

正式结果：equal-only 2/2拒绝；weighted rank exact，CPU Max/RMS `0.007760/3.639e-6`、loss
`3.20e-7`、peak275,790,348 bytes。同步模式保留，scale-before-ready移交Step101。
