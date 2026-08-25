# 2026-08-25 — ranked Model-S checkpoint infrastructure

tiny已证明ownership与状态语义；本节点扩展相同runner到Model-S完整model+AdamW moments。

## Added evidence

- checkpoint runner新增`--model/--context/--compare-binary`；
- 每个成功组保存两个rank的临时safetensors并比较57个Tensor/15,586,176个值，随后删除；
- resumed-final与uninterrupted-final完整checkpoint继续逐字节比较；
- worker回报checkpoint write、rank1 wait、read/verify和resume(load+restore)时间；
- summary记录大文件大小和三次成功写时间；
- ownership故障仍用tiny注入，因为它发生在共同barrier/写/marker层，不重复制造187MB失败文件。

## Pilot

固定Model-S、T32、两rank，`1+1`恢复对照不中断2步：

- 三个checkpoint均187,042,096 bytes（约178.4 MiB）并且final逐字节相等；
- 三组双rank参数Max/RMS均0；
- 写时间约1015–1048ms；rank1最大等待1056ms；
- checkpoint read/verify最大544ms；两个rank load+restore最大745ms；
- resume/不中断final step与optimizer step均2；
- rank0写3次、其他rank 0次；失败为peer−15/rank0 1；
- checkpoint/safetensors/ready/tmp/ID全部清理。

完整RCCL标签49/49，静态checkpoint合同和测试审计125通过。下一提交从干净revision生成正式
Model-S smoke；单次I/O时间只作当前环境资源证据，不推广为存储性能排名。
