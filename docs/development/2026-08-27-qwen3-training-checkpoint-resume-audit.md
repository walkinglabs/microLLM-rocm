# Qwen3 官方 checkpoint 中断恢复

日期：2026-08-27

正确实验必须让两条路从完全相同的step1状态分叉。第一版让另一个进程重新计算step1，出现Parameter `2.38e-7`、Moment `5.36e-7`差异，混入GPU调度，因此bitwise解释被拒绝。

最终实验由控制进程在step1后保存checkpoint并继续到step3；新进程加载同一个checkpoint再做两步。FP32/BF16的后两步loss都bitwise相同，310参数Max分别`1.86e-9/1.49e-8`，620 moments Max `2.98e-8/1.79e-7`，step/global step严格为3。

参数和moment不宣称bitwise：tied embedding及原子累加会因进程重启改变调度。固定恢复门为loss `1e-7`、参数`1e-7`、moment`1e-5`，两种精度5/5通过。checkpoint v2配置、optimizer state和BF16 mirrors恢复路径均实际执行。
