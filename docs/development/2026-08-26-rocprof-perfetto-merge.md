# 2026-08-26 — rocprof marker/kernel合并为Perfetto

最初版本把marker和kernel中相同的`Correlation_Id`直接当作关系。开启HIP API trace后，这个解释被
推翻：marker的ID属于marker调用，Kernel的ID属于`hipLaunchKernel`或copy API；在较小trace中数值
相同可能只是巧合。

修正后的`merge_rocprof_perfetto(..., hip_api_csv=...)`使用两段证据：ROCTX range在host时间上包含
launch API；launch API与Kernel再共享精确`Correlation_Id`。Kernel可以在range结束后才启动或完成，
因此不要求异步Kernel时间戳被host range包含。没有HIP API CSV时仍输出marker/kernel事件，但不生成
未经证明的flow。

导出原子替换；空、负时长、重复marker ID或schema不兼容CSV拒绝。正式C++和Python证据均已使用
`--hip-trace --marker-trace --kernel-trace`重跑。
