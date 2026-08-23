# Experiment 144 data

这份证据回答“头文件有outer-vector，当前MI300运行时是否真的执行”。

- fresh app构建50步和CLI合同；
- 同一fresh目录打开tests后，只增量构建HIP test 12步；
- GTest JSON保存`output_column_native_status=0`与post-scale=1；
- 两模型T512的6个worker证明状态被带进真实CLI；
- `verification.json`锁定Qwen 336、DeepSeek 394次post-scale；
- 所有命令、GPU预检、退出码和空stderr均保留。

正式模型只跑单进程、1次warm-up和1次测量，因此TPS不是性能证据。
