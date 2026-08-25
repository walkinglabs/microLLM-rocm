# 2026-08-25 — ranked uneven-input weighting infrastructure

简单all-reduce average只在每rank有效token数相同时等价于global-batch loss。最后一个数据batch常会
不整齐；直接平均会让小batch拥有过高权重。

## Contract

- worker/launcher/matrix接受每rank `--rank-batch-rows`；默认全部1，旧路径不变；
- rank-local batch第一行保持历史数据，额外行按rank/row/token位置确定生成；
- CPU reference拼接所有rank全部行；
- 每rank在模型训练前用一个RCCL average交换local有效token数；
- `equal-only`发现local tokens不等于average时，在任何参数collective前明确失败；
- `token-weighted`将本地mean-loss gradient乘`local_tokens/average_tokens`，再做原有RCCL average；
- 数学结果为`sum(local_grad*local_tokens)/global_tokens`；
- weighted模式暂不接ready overlap，避免gradient已入队后再缩放；同步per-parameter/bucket/view可用；
- JSON回报rows、local/average tokens和gradient scale；launcher用rows加权local loss再比CPU。

## Pilot

tiny两rank、rows `[1,2]`：

- equal-only：两进程完成token-count collective后均返回1，错误进程2，3.08秒有界退出；
- token-weighted：rank0/1 scale约`0.6667/1.3333`；
- 三步rank参数Max/RMS 0；
- CPU Max/RMS `8.18e-8/8.79e-9`；
- local-loss加权与CPU global loss最大差`1.94e-7`；
- 全部7条ranked CTest与测试审计125通过。

下一提交从干净revision保存拒绝与weighted三步正式证据，再决定Model-S smoke。
