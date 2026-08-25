# Step 96 — Ranked Model-S checkpoint smoke

Status: complete

tiny证明ownership和状态语义，但不能代表15,586,176参数与两组AdamW moments的实际I/O。下一节点
固定Model-S、T32、两rank、一步checkpoint：

- rank0写完整model+AdamW+ExperimentState，rank1只验证；
- 记录文件bytes、rank0写时间、rank1等待时间和加载/恢复时间；
- 新两rank恢复后再跑一步，与不中断两步最终checkpoint逐字节比较；
- checkpoint/ready/tmp验证后删除，不提交大文件；
- 写失败传播合同继续执行。

该节点先证明资源可用与恢复等价，不把磁盘缓存下单次最快时间写成通用吞吐。

实现已支持Model-S临时rank safetensors全量比较与checkpoint write/wait/verify/resume计时。pilot的
checkpoint为187,042,096 bytes，final逐字节相等，rank exact；write约1.02–1.05s、restore最大
745ms、verify最大544ms。正式干净revision结果前只算资源pilot。

正式结果：write 1.022–1.068s、wait1.069s、verify532ms、restore740ms；final checkpoint逐字节
相等、rank exact、无大文件残留。Model-S smoke完成，不作I/O性能推广。
