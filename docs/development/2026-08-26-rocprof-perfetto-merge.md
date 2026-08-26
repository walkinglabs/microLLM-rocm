# 2026-08-26 — rocprof marker/kernel合并为Perfetto

`merge_rocprof_perfetto(marker_csv, kernel_csv, output)`严格校验rocprof CSV，统一时间原点，将ROCTX与GPU
Kernel分别写成`X`事件，并用共同`Correlation_Id`生成`s/f` flow。不能靠range包含关系猜关联：当前pilot
中marker结束与GPU timestamp有约8µs偏差，但双方correlation=2。

导出原子替换；空或schema不兼容CSV拒绝。基础设施提交后从干净revision重跑正式marker+kernel trace。
