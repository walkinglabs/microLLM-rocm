# Step 95 — Ranked checkpoint ownership and resume

Status: planned

one-process-per-GPU训练已经有rank identity、同步bucket、persistent views、context-selective overlap
和peer-failure传播，但还没有生产checkpoint ownership。下一节点不继续调Reducer，而是规定：

- 只有rank0在optimizer完成且所有rank参数一致后写checkpoint；
- checkpoint包含模型、AdamW、step、数据游标、seed和配置；
- 其他rank不写同一路径，并等待rank0发布完成标记；
- resume后每rank恢复相同状态，后续三步与不中断双rank轨迹一致；
- rank0写失败必须让其他rank有限时退出，不得永久等待。

先做tiny多步二进制合同，再扩展Model-S one-step。该节点是可靠性/恢复证据，不作为性能实验。
