# 2026-08-25 — ranked weighted-overlap matrix infrastructure

## 为什么不能直接开始计时

原context matrix只会测每张卡相同batch。Step 101需要测`[B1,B2]`，还必须回答两个比“程序
没崩”更严格的问题：

1. 同步weighted与overlap weighted最后得到的是不是同一组参数？
2. 三步CPU loss门应该用什么公开容差，失败时会不会残留几百MB临时权重？

## 新的测量契约

`ranked_overlap_context_matrix.py`现在接受：

- `--rank-batch-rows 1,2`；
- `--input-weighting token-weighted`；
- `--mean-loss-tolerance`，容差必须在命令与JSON里明示；
- 每条策略临时保留rank 0共识参数，成对逐项比较57个Tensor、15,586,176个值；
- 比较完成后删除临时参数，并在summary写
  `temporary_parameter_files_retained=false`。

launcher也修复了失败清理：即使CPU gate拒绝结果，rank与reference的临时safetensors也会先
删除。用`1e-12`故意触发Model-S门时，进程按预期失败且残留参数文件为0。

## pilot看见了什么

两张MI300X、Model-S T32、`[B1,B2]`、三步、一次交替pilot：

- 同步与overlap最终参数Max/RMS：0/0；
- rank间参数Max/RMS：0/0；
- CPU参数Max/RMS：`0.007758 / 4.261e-6`；
- 第三步加权loss使全程最大差达到`4.067e-4`；
- 同步steady step：8.132 ms；
- overlap steady step：8.520 ms；
- pilot速度比：0.9545x，未过1.01门。

旧`1e-4`是一步/等长路径上观察到的门，不足以覆盖uneven三步轨迹。正式实验将明确使用
`1e-3`，同时保留更严格的参数Max `1e-2`、RMS `1e-5`以及同步/overlap精确相等门。容差变化
被写进命令和JSON，不会静默放宽。

pilot只验证runner并暴露T32反例，不形成正式性能结论。正式测量从本节点推送后的干净revision
运行T128、每策略三轮、每轮丢弃第1步。
